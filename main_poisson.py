import os
import sys
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  # 添加matplotlib用于绘图

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../src/"))  # add the path to the DiffusionNet src
import diffusion_net
from diffusion_net.utils import set_device,LpLoss,MultipleLoss,convert_to_one_hot,toNP
from torch.utils.data import DataLoader
from glno_dataset import CylinderFlowBase,PoissonBase
import yaml

# === Options
# system things

'''
tmux new-session -s train_session5
screen -S train_session5

export CUDA_VISIBLE_DEVICES=0,3
conda activate zm_AMG
cd ICLR/AMG-main/GLNO
torchrun --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29505 main_poisson.py --input_features xyz --load_model Best_models/poisson_val42_xyz_poisson-GLNOM-pole1-sigma0-lr0.001-channel64-block-4-decay-100x0.5-L2loss.pth
python main_poisson.py --input_features xyz
'''

# Parse a few args
parser = argparse.ArgumentParser()
parser.add_argument("--evaluate", default=False, action="store_true", help="evaluate using the pretrained model")
parser.add_argument("--input_features", type=str, help="what features to use as input ('xyz' or 'hks') default: hks", default = 'hks')
parser.add_argument("--load_model", type=str, help="path to load a pretrained model from")
# 添加单机多卡参数
parser.add_argument("--local_rank", type=int, default=None, help="local rank for distributed training (automatically set by torchrun)")
args = parser.parse_args()

# 初始化分布式训练
def setup_distributed():
    # 检查是否在分布式环境中运行
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.distributed = True
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.world_size = 1
        args.rank = 0
        args.local_rank = 0
        return

    # 设置分布式训练
    if args.distributed:
        # 设置设备
        torch.cuda.set_device(args.local_rank)
        
        # 初始化进程组
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        
        print(f"Initialized distributed training with {args.world_size} GPUs, rank {args.rank}, local rank {args.local_rank}")

# 调用初始化
setup_distributed()

# system things
# device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')
device =torch.device("cuda:2")
dtype = torch.float32
with open('config_poisson.yaml', 'r') as f:
    config = yaml.safe_load(f)

save_index = config["system"]["save_index"]
# 初始化记录列表
train_loss_history = []
test_loss_history = []
val_loss_history = []

# problem/dataset things
n_class = 1

# model
input_features = args.input_features  # args.input_features # one of ['xyz', 'hks']

# training settings
train = True  # not args.evaluate
n_epoch = config["training"]["epoch"]
lr = config["training"]["lr"]  # 0.00015
decay_every = config["training"]["decay_interval"]
decay_rate = config["training"]["decay_rate"]
augment_random_rotate = (input_features == 'xyz')
k_eig=config['dataset']['k_eig']

# Important paths
base_path = os.path.dirname(__file__)
op_cache_dir = os.path.join(base_path, "data","poisson", "op_cache")
# pretrain_path = os.path.join(base_path, "pretrained_models/human_seg_{}_4x32.pth".format(input_features))
model_save_path = os.path.join(base_path,"Best_models")
dataset_path = os.path.join(base_path, "../data/poisson")

# === Load datasets

# process_path = dataset_path + '/poisson_data_processed.pt'
# print('Loading processed data from ', process_path)
# train_data, valid_data, test_data = torch.load(process_path)
# if args.rank == 0:
#     train_dataset = PoissonBase(data=train_data, root_dir=dataset_path, mode='train', k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
# if args.rank==1:
#     test_dataset = PoissonBase(data=test_data, root_dir=dataset_path, mode='test', k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
# if args.rank==2:
#     valid_dataset = PoissonBase(data=valid_data, root_dir=dataset_path, mode='val',k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    
# === Load datasets

if not args.evaluate: 
    train_dataset = PoissonBase(root_dir=dataset_path, mode='train', k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    else:
        train_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=20, sampler=train_sampler, shuffle=(train_sampler is None))

test_dataset = PoissonBase(root_dir=dataset_path, mode='test', k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
if args.distributed:
    test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
else:
    test_sampler = None
test_loader = DataLoader(test_dataset, batch_size=20, sampler=test_sampler, shuffle=False)

valid_dataset = PoissonBase(root_dir=dataset_path, mode='val',k_eig=k_eig, use_cache=True, op_cache_dir=op_cache_dir)
if args.distributed:
    valid_sampler = DistributedSampler(valid_dataset, num_replicas=args.world_size, rank=args.rank, shuffle=False)
else:
    valid_sampler = None
valid_loader = DataLoader(valid_dataset, batch_size=20, sampler=valid_sampler, shuffle=False)

if args.rank == 0:
    print("Train dataset size: {}".format(len(train_dataset)))
    print("Vali dataset size: {}".format(len(valid_dataset)))
    print("Test dataset size: {}".format(len(test_dataset)))

# === Create the model

C_in = {'xyz': 3, 'hks': 16}[input_features]  # dimension of input features

model = diffusion_net.layers_fno.DiffusionNet(config=config["model"],
                                              C_in=C_in,
                                              C_out=n_class,
                                              last_activation=None #lambda x: torch.nn.functional.log_softmax(x, dim=-1),
                                              )

if args.rank==0:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total / 1e6:.2f} M | Trainable: {trainable / 1e6:.2f} M")

model = model.to(device)

# 使用分布式数据并行
if args.distributed:
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)


if args.load_model:
    if args.rank == 0:
        print("Loading pretrained model from: " + str(args.load_model))
    if args.distributed:
        if args.rank == 0:
            checkpoint = torch.load(args.load_model)
        else:
            checkpoint = None
        # 创建列表并广播（会原地修改列表）
        checkpoint_list = [checkpoint]
        dist.broadcast_object_list(checkpoint_list, src=0)
        checkpoint = checkpoint_list[0]  # 从列表获取广播后的值
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(torch.load(args.load_model))
    if args.rank == 0:
        print("...done")

if args.evaluate and not args.load_model:
    if args.rank == 0:
        raise ValueError("Called with --evaluate but not --load_model. This will evaluate on a randomly initialized model, which is probably not what you want to do.")


# if not train:
#     # load the pretrained model
#     print("Loading pretrained model from: " + str(pretrain_path))
#     model.load_state_dict(torch.load(pretrain_path))

# === Optimize
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

myloss = LpLoss(size_average=True)

decay_epoch=[]

def train_epoch(epoch):
    # Implement lr decay
    if epoch > 0 and (epoch % decay_every == 0 or epoch in decay_epoch):
        global lr
        lr *= decay_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        # print("LR decay!")
            # Set model to 'train' mode
    model.train()
    optimizer.zero_grad()

    correct = 0
    total_num = 0
    losses1=[]
    losses2=[]
    losses3=[]

    if args.distributed:
        train_loader.sampler.set_epoch(epoch)

    for data in tqdm(train_loader) if args.rank == 0 else train_loader:

        # Get data
        # data=data.to(device)
        verts, x, faces, mass, evals, evecs, labels, edges, disnorm = data  # evals, evecs, gradX, gradY比较重要
        verts=verts.to(device)
        x=x.to(device)
        faces=faces.to(device)
        mass=mass.to(device)
        # L=L.to(device)  
        evals=evals.to(device)
        evecs=evecs.to(device)
        labels=labels.to(device)
        edges=edges.to(device)
        disnorm=disnorm.to(device)
        # print(curv.shape)

        # Randomly rotate positions
        # if augment_random_rotate:
        #     verts = diffusion_net.utils.random_rotate_points(verts)

        # Construct features
        if input_features == 'xyz':
            features = torch.cat([x,verts[:,:,:2]],dim=-1)
        # elif input_features == 'hks':
        #     features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

        # Apply the model

        preds = model(features, mass, evals=evals, evecs=evecs, faces=faces, edges=edges,
                      curv=disnorm)

        # Evaluate loss
        # loss = torch.nn.functional.nll_loss(preds, labels)
        lossl2=myloss(preds,labels)
        losses1.append(lossl2.item())
        # losses2.append(lossl2[2].item())
        # losses3.append(lossl2[3].item())
        lossl2.backward()

        # Step the optimizer
        # if (epoch%10==0 or epoch==n_epoch-1) and epoch>0:
        optimizer.step()
        optimizer.zero_grad()

    # 收集所有进程的损失
    if args.distributed:
        losses_tensor = torch.tensor([np.mean(losses1)], device=device)
        dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
        train_loss = losses_tensor / args.world_size
        train_loss=train_loss.item()
    else:
        train_loss = np.mean(losses1)

    return train_loss

# Do an evaluation pass on the test dataset
def test_val(dataloader):
    model.eval()

    correct = 0
    total_num = 0
    losses1=[]
    losses2=[]
    losses3=[]

    if args.distributed:
        dataloader.sampler.set_epoch(epoch)

    with torch.no_grad():

        for data in tqdm(dataloader) if args.rank == 0 else dataloader:

            # Get data
            # data=data.to(device)
            verts, x, faces, mass, evals, evecs, labels, edges, disnorm = data  # evals, evecs, gradX, gradY比较重要
            verts=verts.to(device)
            x=x.to(device)
            faces=faces.to(device)
            mass=mass.to(device)
            # L=L.to(device)  
            evals=evals.to(device)
            evecs=evecs.to(device)
            labels=labels.to(device)
            edges=edges.to(device)
            disnorm=disnorm.to(device)
            # print(curv.shape)

            # Randomly rotate positions
            # if augment_random_rotate:
            #     verts = diffusion_net.utils.random_rotate_points(verts)

            # Construct features
            if input_features == 'xyz':
                features = torch.cat([x,verts[:,:,:2]],dim=-1)
            # elif input_features == 'hks':
            #     features = diffusion_net.geometry.compute_hks_autoscale(evals, evecs, 16)

            # Apply the model

            preds = model(features, mass, evals=evals, evecs=evecs, faces=faces, edges=edges,
                        curv=disnorm)

            lossl2=myloss(preds,labels)
            losses1.append(lossl2.item())
            # losses2.append(lossl2[2].item())
            # losses3.append(lossl2[3].item())

    # 收集所有进程的损失
    # if args.distributed:
        # print("?")
        # losses_tensor = torch.tensor([np.mean(losses1)], device=device)
        # print(">>")
        # dist.all_reduce(losses_tensor, op=dist.ReduceOp.SUM)
        # print(">>>")
        # train_loss = losses_tensor / args.world_size
        # print(">>>>")
        # print(train_loss)
        # train_loss=train_loss.item()
        # print(">>>>>")
    # else:
    train_loss = np.mean(losses1)
    # print("!!!")
    return train_loss

if train:
    if args.rank == 0:
        print("Start training...")

    best_loss=1.0
    last_epoch=-4
    for epoch in range(n_epoch):
        train_l2loss = train_epoch(epoch)
        val_l2loss = test_val(valid_loader)
        train_loss_history.append(train_l2loss)
        val_loss_history.append(val_l2loss.item())
        if args.rank==0:
            print("Epoch {} - Train L2: {} Val L2: {}".format(epoch, train_l2loss, val_l2loss))
        if val_l2loss<best_loss and epoch>=last_epoch+5:
            last_epoch=epoch
            best_loss=val_l2loss
            save_path=os.path.join(model_save_path,"poisson_val{}_{}_{}.pth".format(int(1000*val_l2loss),input_features, save_index))
            print(" ==> saving last model to " + save_path)
            torch.save(model.state_dict(), save_path)

# if args.rank == 0:
print(f"Test_Loss on {args.rank}: {test_val(test_loader)}")

if args.rank == 0:
    # === 可视化设置 ===
    # 创建保存结果的目录
    current_directory = os.getcwd()
    case = "Case_"
    folder_index = str(save_index)

    results_dir = "/" + case + folder_index + "/"
    save_results_to = current_directory + results_dir
    if not os.path.exists(save_results_to):
        os.makedirs(save_results_to)  # 创建结果目录

    # 保存训练数据到文本文件
    np.savetxt(os.path.join(save_results_to, 'train_acc.txt'), train_loss_history)
    np.savetxt(os.path.join(save_results_to, 'test_acc.txt'), val_loss_history)
    # if train_loss_history:
    #     np.savetxt(os.path.join(save_results_to, 'train_loss.txt'), train_loss_history)

    # 保存训练数据到文本文件
    np.savetxt(os.path.join(save_results_to, 'train_acc.txt'), train_loss_history)
    np.savetxt(os.path.join(save_results_to, 'test_acc.txt'), val_loss_history)
    # if train_loss_history:
    #     np.savetxt(os.path.join(save_results_to, 'train_loss.txt'), train_loss_history)

    # 训练结束后绘制训练曲线
    num_epoch = n_epoch
    epochs = np.linspace(1, num_epoch, num_epoch)

    # 创建图表
    fig = plt.figure(constrained_layout=False, figsize=(8, 10))

    # 准确率曲线
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.plot(epochs, train_loss_history, color='blue', label='Train Accuracy')
    ax1.plot(epochs, val_loss_history, color='red', label='Test Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.legend(loc='lower right')
    ax1.grid(True)
    ax1.set_title(f'Accuracy Curve - Input Features: {input_features}')

    plt.tight_layout()
    plt.savefig(os.path.join(save_results_to, f'training_curves_{input_features}.png'))
    plt.close()

    print(f"Training curves saved to {save_results_to}")

# Test
    # print("Overall test accuracy: {}".format(test_loss11))

# 清理分布式进程组
if args.distributed:
    dist.destroy_process_group()
