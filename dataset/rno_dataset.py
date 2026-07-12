import os
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import potpourri3d as pp3d

from tqdm import tqdm
import h5py


class CarCFD(Dataset):
    def __init__(self, mode='train', dataset_path=None, preload=False, features=None, device='cpu', **kwargs):
        self.mode = mode
        self.dataset_path = dataset_path
        self.device=device
        self.preload=preload
        self.features=features

        train_cache = os.path.join(self.dataset_path, "train.h5")
        val_cache=os.path.join(self.dataset_path, "val.h5")
        test_cache = os.path.join(self.dataset_path, "test.h5")
        self.len_list={'train':690,'val':99,'test':100}
        if mode=='train':
            self.load_cache = train_cache
        elif mode=='val':
            self.load_cache = val_cache
        elif mode=='test':
            self.load_cache = test_cache
        else:
            raise ValueError("Invalid mode: {}".format(mode))
        
        with h5py.File(self.load_cache, 'r') as f:
            # 获取所有样本组名（假设元数据组为 'metadata'，其余都是样本组）
            # 或者从 metadata/sample_names 读取，这里采用简单规则：排除 'metadata'
            self.sample_groups = [name for name in f.keys() if name.startswith('sample_')]
            self.num_samples = len(self.sample_groups)

            if self.num_samples == 0:
                raise RuntimeError(f"No sample groups found in {self.load_cache}")

            # 获取第一个样本组，用于确定可用的特征
            first_grp = f[self.sample_groups[0]]
            available = set(first_grp.keys())

            # 处理稀疏矩阵 L：如果存在 L_data, L_indices, L_indptr, L_shape，则视为一个特征 'L'
            sparse_L_components = {'L_data', 'L_indices', 'L_indptr', 'L_shape'}
            if sparse_L_components.issubset(available):
                self.has_sparse_L = True
                available = available - sparse_L_components
                available.add('L')
            else:
                self.has_sparse_L = False

            # 确定最终要返回的特征列表
            if self.features is None:
                self.feature_list = sorted(available)  # 按字母排序保持稳定
            else:
                # 检查用户请求的特征是否都存在
                for feat in self.features:
                    if feat == 'L':
                        if not self.has_sparse_L:
                            raise ValueError(f"Feature 'L' requested but sparse L components not found in file.")
                    elif feat not in available:
                        raise ValueError(f"Feature '{feat}' not found in file (available: {available})")
                self.feature_list = self.features
        print(self.feature_list)

    def __len__(self):
        return self.len_list[self.mode]
    
    def __getitem__(self, idx):
        with h5py.File(self.load_cache, 'r') as f:
            grp = f[self.sample_groups[idx]]
            sample = {}
            for feat in self.feature_list:
                if feat == 'L' and self.has_sparse_L:
                    continue
                    # L = sparse_np_to_torch(read_sparse_L(grp))
                    # sample['L'] = L
                else:
                    arr = grp[feat][:]
                    if arr.dtype.kind in {'U', 'S', 'O'}:
                        sample[feat] = arr
                    else:
                        # if feat=='curv':
                        #     arr=enhance_curvature_values(arr)
                        sample[feat] = torch.from_numpy(arr).to(self.device)
                        if sample[feat].dtype==torch.double:
                            # print(":!",feat)
                            sample[feat]=sample[feat].to(torch.float)
                        elif sample[feat].dtype==torch.int32:
                            sample[feat]=sample[feat].to(torch.int64)
        # print(torch.max(sample['eval']))
        # print(torch.max(sample['curv']))
        # curv=sample['curv']/torch.max(sample['curv'])
        dist=sample['distance']/torch.max(sample['distance'])
        # print(sample['input'].shape)
        print(torch.max(sample['input'],dim=0),torch.min(sample['input'],dim=0))
        return {"input":sample['input'],"vertices":sample['vertice'], \
                "mass":sample['mass'], "evals":sample['eval'].view(-1), \
                "evecs":sample['evec'], "labels":sample['output'],\
                "geo_feat":dist}
