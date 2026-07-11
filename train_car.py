import os
import sys
import argparse
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import yaml
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import model
import utils
from dataset import DATASET_DICT 
from model import MODEL_DICT

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="poisson", help="config of task to run")
parser.add_argument("--config", type=str,default="config.yaml", help="path to config file")
parser.add_argument("--evaluate", default=False, action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--load_model", type=str, help="path to load a pretrained model from")
parser.add_argument("--distributed", default=False, action="store_true", help="use distributed training")
parser.add_argument("--local_rank", type=int, default=None, help="local rank for distributed training (automatically set by torchrun)")
args = parser.parse_args()
args=vars(args) # convert to dictionary

# paths
base_path = os.path.dirname(__file__)
dataset_path = os.path.join(base_path, "data", args['task'])
args['config'] = os.path.join(base_path, "config", args['task'], args['config'])
model_save_path = os.path.join(base_path,"checkpoints", args['task'])
args['log_dir']=os.path.join(base_path,"logs")
args=utils.load_config(args) # TODO load config from pretrained model

if args['system']['log']:
    saving_path, saving_name = utils.set_up_logger(args["model"]["name"], args["task"], args['log_dir']) #  (TODO: add more details on names)
else:
    saving_path, saving_name = utils.get_dir_path(args["model"]["name"], args["task"], args['log_dir'])
args['saving_path'] = saving_path
args['saving_name'] = saving_name
if args['system']['log']:
    utils.save_config(args, saving_path)

# initialize distributed training
if args["distributed"]:# TODO check distrubuted
    utils.setup_distributed(args)
    args['system']['verbose'] = False if args['rank']!=0 else args['system']['verbose']

# system info logging
if args['system']['verbose']:
    logger = logging.info if args['system']['log'] else print
    logger(args)

# system settings
device = torch.device(f'cuda:{args["local_rank"]}' if torch.cuda.is_available() else 'cpu')
args["model"]["device"]=device
dtype = torch.float32

# === Load datasets
if not args["evaluate"]: 
    train_dataset = DATASET_DICT[args["dataset"]["name"]](mode='train', dataset_path=dataset_path)
    if args["dataset"]["create_val"]:
        train_list,val_list=utils.split_train_val(len(train_dataset),ratio=0.05)
        train_dataset.split_val(train_list)
    if args["distributed"]:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args["world_size"], rank=args["rank"], shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args["dataset"]["train_batch_size"], sampler=train_sampler, shuffle=(train_sampler is None), collate_fn=utils.concat_collate_fn if args['dataset']['contact_collate'] else None)

test_dataset = DATASET_DICT[args["dataset"]["name"]](mode='test', dataset_path=dataset_path)
if args["distributed"]:
    test_sampler = DistributedSampler(test_dataset, num_replicas=args["world_size"], rank=args["rank"], shuffle=False)
else:
    test_sampler = None
test_loader = DataLoader(test_dataset, batch_size=args["dataset"]["test_batch_size"], sampler=test_sampler, shuffle=False, collate_fn=utils.concat_collate_fn if args['dataset']['contact_collate'] else None)

valid_dataset = DATASET_DICT[args["dataset"]["name"]](mode='val', dataset_path=dataset_path)
if args["dataset"]["create_val"] and not args["evaluate"]:
    valid_dataset.split_val(val_list)
if args["distributed"]:
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=args["world_size"], rank=args["rank"], shuffle=False)
else:
    valid_sampler = None
valid_loader = DataLoader(valid_dataset, batch_size=args["dataset"]["test_batch_size"], sampler=valid_sampler, shuffle=False, collate_fn=utils.concat_collate_fn if args['dataset']['contact_collate'] else None)

if args['system']['verbose']:
    logger("Train dataset size: {}".format(len(train_dataset)))
    logger("Vali dataset size: {}".format(len(valid_dataset)))
    logger("Test dataset size: {}".format(len(test_dataset)))

# === Create the model
if args['model']['name'] not in MODEL_DICT.keys():
    raise NotImplementedError("Model {} not implemented".format(args['model']['name']))
model = MODEL_DICT[args["model"]["name"]](config=args["model"])

# model trainable parameters number
if args['system']['verbose']:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger(f"Total params: {total / 1e6:.2f} M | Trainable: {trainable / 1e6:.2f} M")

model = model.to(device)

# data distribution initialization
if args["distributed"]:
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

if args["load_model"]: # TODO check distributed model loading and model loading from checkpoint
    if args['system']['verbose']:
        logger("Loading pretrained model from: " + str(args.load_model))
    if args["distributed"]:
        if args["rank"] == 0:
            checkpoint = torch.load(args.load_model)
        else:
            checkpoint = None
       
        checkpoint_list = [checkpoint]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint = checkpoint_list[0]  
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(args.load_model))
    if args['system']['verbose']:
        logger("...done")

if args["evaluate"] and not args["load_model"]:
    raise ValueError("Called with --evaluate but not --load_model. This will evaluate on a randomly initialized model, which is probably not what you want to do.")

# training settings
n_epoch = args["training"]["epoch"]
lr = args["training"]["lr"]
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = utils.build_scheduler(optimizer, args['training']['scheduler'])

if args["training"]["loss"]=="L2relative":
    myloss = utils.LpLoss(size_average=True)
elif args["training"]["loss"]=="cross_entropy":
    myloss = utils.classify_nll_loss()
elif args["training"]["loss"]=="MultiL2relative":
    myloss = utils.MultipleLoss(size_average=True)
elif args['training']['loss']=='CarLoss':
    myloss = utils.CarCFDLoss()
else:
    raise NotImplementedError("Loss {} not implemented".format(args["training"]["loss"]))

# Do a training epoch
def train_process(epoch):
    model.train()
    optimizer.zero_grad()

    if args["distributed"]:
        train_loader.sampler.set_epoch(epoch)

    for data in train_loader:
        pos,x,m,eval,evec,y,disnorm=data
        pos=pos.to(device)
        x=x.to(device)
        m=m.to(device)
        eval=eval.to(device)
        evec=evec.to(device)
        y=y.to(device)
        disnorm=disnorm.to(device)
        features = torch.cat([x,pos],dim=-1)

        preds = model(features, m, evals=eval, evecs=evec,
                      curv=disnorm)

        # Evaluate loss
        labels=data['labels'].to(device)
        loss_result =myloss(preds,labels)

        if isinstance(loss_result, (list, tuple)): #multiple losses
            loss_result[0].backward()
            loss_values = [loss_item.item() if hasattr(loss_item, 'item') else loss_item  for loss_item in loss_result]
        else: #simgle loss               
            loss_result.backward()                                                
            loss_values = [loss_result.item()]
        
        if 'loss_history' not in locals():
            loss_history = [[] for _ in range(len(loss_values))]
        
        for i, loss_val in enumerate(loss_values):
            loss_history[i].append(loss_val)

        # Step the optimizer
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()

    if scheduler is not None:
        scheduler.step()

    # collect all process losses
    if args["distributed"]:
        loss_tensors = []
        for loss_component in loss_history:
            avg_loss = np.mean(loss_component)
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensors.append(loss_tensor / args.world_size)
        
        train_losses = [tensor.item() for tensor in loss_tensors]
    else:
        train_losses = [np.mean(loss_component) for loss_component in loss_history]

    return train_losses

# Do an evaluation on the test/validate dataset
def evaluate_process(dataloader):
    model.eval()

    if args["distributed"]:
        dataloader.sampler.set_epoch(epoch)

    with torch.no_grad():

        for data in dataloader:
            # Apply the model
            pos,x,m,eval,evec,y,disnorm=data
            pos=pos.to(device)
            x=x.to(device)
            m=m.to(device)
            eval=eval.to(device)
            evec=evec.to(device)
            y=y.to(device)
            disnorm=disnorm.to(device)
            features = torch.cat([x,pos],dim=-1)
            
            preds = model(features, m, evals=eval, evecs=evec,
                        curv=disnorm)
            
            labels=data['labels'].to(device)
            loss_result =myloss(preds,labels)

            if isinstance(loss_result, (list, tuple)): #multiple losses
                loss_values = [loss_item.item() if hasattr(loss_item, 'item') else loss_item for loss_item in loss_result]
            else: #simgle loss                                                           
                loss_values = [loss_result.item()]
            
            if 'loss_history' not in locals():
                loss_history = [[] for _ in range(len(loss_values))]
            
            for i, loss_val in enumerate(loss_values):
                loss_history[i].append(loss_val)

    # collect all process losses
    if args["distributed"]:
        loss_tensors = []
        for loss_component in loss_history:
            avg_loss = np.mean(loss_component)
            loss_tensor = torch.tensor([avg_loss], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            loss_tensors.append(loss_tensor / args.world_size)
        
        train_losses = [tensor.item() for tensor in loss_tensors]
    else:
        train_losses = [np.mean(loss_component) for loss_component in loss_history]

    return train_losses

# Training loop
if not args["evaluate"]:
    if args['system']['verbose']:
        logger("Start training...")

    best_epoch = 0
    best_loss = None
    counter = 0

    start_time = time.time()
    for epoch in range(n_epoch):
        epoch_start_time = time.time()
        
        lr=optimizer.param_groups[0]["lr"]
        train_loss = train_process(epoch)

        # time cost (total time per epoch and estimated remaining time)
        epoch_time = time.time() - epoch_start_time
        elapsed_time = time.time() - start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        remaining_epochs = n_epoch - epoch - 1
        estimated_remaining_time = avg_time_per_epoch * remaining_epochs

        if args['system']['verbose']:
            output = "Epoch {} | lr: {:.6f} | training loss: ".format(epoch, lr)
            output+=utils.output_loss_(train_loss, args["training"]["loss_meaning"])
            output += " | epoch time: {:.2f}s".format(epoch_time)
            output += " | remaining: {}".format(utils.format_time(estimated_remaining_time))
            logger(output)

        # save checkpoint if needed (ATTENTION: including model, optimizer, scheduler states)
        if args['training']['saving_checkpoint'] and (epoch + 1) % args['training']['checkpoint_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.cpu().state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss_record': train_loss,
                }, os.path.join(args['saving_path'], "checkpoint_{}_{:.4f}.pth".format(epoch,train_loss[0])))
            model=model.to(device)
            if args['system']['verbose']:
                logger("Epoch {} | save checkpoint in {}".format(epoch, args['saving_path']))
        
        # evaluate on validation set 
        if (epoch + 1) % args['training']['eval_freq'] == 0:
            val_loss = evaluate_process(valid_loader)
            test_loss= evaluate_process(test_loader)
            if args['system']['verbose']:
                output="Epoch {} | validation loss: ".format(epoch)
                output+=utils.output_loss_(val_loss, args["training"]["loss_meaning"])
                logger(output)
                output="Epoch {} | test loss: ".format(epoch)
                output+=utils.output_loss_(test_loss, args["training"]["loss_meaning"])
                logger(output)
            
            # save the best model on validation set 
            index=args["training"]['best_index']
            if index==-1:
                continue
            if not best_loss or val_loss[index] < best_loss[index]:
                counter = 0
                best_epoch = epoch
                best_loss = val_loss 
                torch.save(model.cpu().state_dict(), os.path.join(args['saving_path'], "best_model.pth")) 
                model=model.to(device)
                if args['system']['verbose']:
                    logger("Epoch {} | save best models in {}".format(epoch, args['saving_path']))
            # early stopping
            elif args['training']['patience'] != -1:
                counter += 1
                if counter >= args['training']['patience']:
                    if args['system']['verbose']:
                        logger("Early stop at epoch {}".format(epoch))
                    break

    logger("Training Finished!")
    
    # load best model on validation set or last model
    if not best_loss:
        torch.save(model.cpu().state_dict(), os.path.join(args['saving_path'], "best_model.pth"))
    else:
        model.load_state_dict(torch.load(os.path.join(args['saving_path'], "best_model.pth")))
        logger("Load best models at epoch {} from {}".format(best_epoch, args['saving_path']))        
    model=model.to(device)
    
    valid_loss_record = evaluate_process(valid_loader)
    if args['system']['verbose']:
        output="Validation loss: "
        output += utils.output_loss_(valid_loss_record, args["training"]["loss_meaning"])
        logger(output)

# valuate on test set
test_loss_record = evaluate_process(test_loader)

output="Test loss: "
output += utils.output_loss_(test_loss_record, args["training"]["loss_meaning"])

if args['system']['verbose']:
    logger(output)
else :
    print(output)

if args["distributed"]:
    dist.destroy_process_group()