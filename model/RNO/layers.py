import sys
import os
import random

from model.GLNO.pointnet import FPSPointNetModule
import scipy
import scipy.sparse.linalg as sla
# ^^^ we NEED to import scipy before torch, or it crashes :(
# (observed on Ubuntu 20.04 w/ torch 1.6.0 and scipy 1.5.2 installed via conda)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry import to_basis, from_basis, rotate
from einops.layers.torch import Rearrange

DEBUG=True#os.getenv('DEBUG_MODE', '0') == '1'

class Laplace_Transform_Layer(nn.Module):
    """
    Laplace Transform with learnable non-linear basis and learnable system poles and residues on spectral domain.

    Inputs:
      - values: (V,C) in the spectral domain
      - L: (V,V) sparse laplacian
      - evals: (K) eigenvalues
      - mass: (V) mass matrix diagonal

      (note: L/evals may be omitted as None depending on method)
    Outputs:
      - (V,C) diffused values
    """

    def __init__(self, config):
        super(Laplace_Transform_Layer, self).__init__()
        self.C_inout = config["C_width"]
        self.in_channels = config["C_width"]
        self.out_channels = config["C_width"]
        self.num_sigma = config["glno_sigma"]
        self.num_poles = config["glno_poles"]
        
        # self.exp_x2_appr=config['exp_x2_appr']
        # self.gaussian=config['gaussian']
        # self.gaussian_factor=config.get('gaussian_factor',1)

        self.LT_kmin = config.get('k_min',0)
        self.LT_kmax = config.get('k_max',config['k_eig'])

        self.period_norm=config['glno_period_norm']
        self.aperiod_norm= config['glno_aperiod_norm']
        self.glno_norm=config['glno_norm']
        self.period=config.get("glno_period",True)
        self.aperiod=config.get("glno_aperiod",True)
        self.norm_period=NormLayer(self.period_norm,self.C_inout)
        self.norm_aperiod=NormLayer(self.aperiod_norm,self.C_inout)
        self.norm_glno=NormLayer(self.glno_norm,self.C_inout)

        self.pole_scale=config.get("pole_scale",None)
        self.residue_scale=config.get("residue_scale",None)
        self.system_poles = nn.Parameter(torch.rand(self.num_poles, self.C_inout, dtype=torch.cfloat)) #*5-2.5for car
        self.system_residues = nn.Parameter(torch.randn(self.num_poles, self.C_inout, dtype=torch.cfloat))
        if self.num_sigma:
            self.sigma = nn.Parameter(-(torch.zeros(self.num_sigma, self.C_inout, dtype=torch.float))) #default 0.02

        if self.pole_scale:
            self.system_poles.imag.data *= self.pole_scale
            if self.num_sigma:
                self.sigma.data *= self.pole_scale

        if self.residue_scale:
            self.system_residues.data *= self.residue_scale
               
        self.normalize_evals=config.get("normalize_evals",None)
        self.scale_evals=config.get("scale_evals",None)
        self.sqrt_evals=config.get("sqrt_evals",True)
        self.normalize_k=self.LT_kmax
        self.normalize_basis=config.get("normalize_basis",False)
        self.safe_mode=config.get("safe_mode",True)
        self.basis_norm=config.get("basis_norm",1)
         
    def pole_to_operator(self, pole, evecs, residue, geo_feat, pos=None, eigen_gate=None, **kwargs):
        """
        参数:
            pole: 复数极点，实部x表示衰减，虚部y表示频率[c_in,c_out,num_pole]
            evecs: 特征向量[batch,num_vectices,num_eig]
            residue: 留数[batch,c_in,c_out,num_pole]
            geo_feat: 几何特征，形状为 (batch_size, vectices)
        """
        if eigen_gate is None:
            term1=torch.exp(torch.einsum("pc,bn->bcnp",pole,geo_feat))
            if self.num_sigma:
                function=torch.einsum("bpskc,bcnp->bnck",residue,term1).real
            else:
                function=torch.einsum("bpkc,bcnp->bnck",residue,term1).real
            return torch.einsum("bnck,bnk->bnc",function,evecs)
        
        evecs=evecs*eigen_gate
        if self.num_sigma:
            function=torch.einsum("bpskc,pc->bck",residue,pole).real
        else:
            function=torch.einsum("bpkc,pc->bck",residue,pole).real
        return torch.einsum("bck,bnk->bnc",function,evecs)

    def PoleRes(self, evals, x_spec, system_poles, system_residues):
        """
        参数:
            evals: (batch, num_eigenvectors)
            x_spec=[batch,sigma, num_eig,inc] 
            system_poles: (num_poles, in_channels)
            system_residues: (num_poles, in_channels)
        返回:
            output_residue1: (batch, K, in_channels)   # periodic
            output_residue2: (batch, num_poles, sigma, num_eig, in_channels)  # aperiodic
        """
        # 1. 计算 pole_in: (batch, K, in_channels)
        if self.sqrt_evals:
            evals_abs_sqrt = torch.sqrt(torch.abs(evals))  # (batch, num_eig)
        else:
            evals_abs_sqrt = torch.abs(evals)  # (batch, num_eig)
        if self.num_sigma:
            evals_ext = evals_abs_sqrt[:, None, :, None]   # (batch, 1, num_eig,  1)
            sigma_ext = self.sigma[None, :, None, :]            # (1, sigma, 1, inchannels)
            pole_in = (evals_ext * 1j + sigma_ext)#.reshape(batch_size, -1, in_channels)

            # 2. 计算 Hw = residues / (pole_in - poles)
            # Hw = residue / (eval - pole)
            # pole_in(b,s,k,1,c) - system_poles(1,1,num_pole,c) -> (b,s,k,num_pole,c)
            term1 = torch.div(1, torch.sub(pole_in[:, :, :, None, :], system_poles[None, None, None, :, :]))
        else:
            pole_in=evals_abs_sqrt*1j
            term1 = torch.div(1, torch.sub(pole_in[:, :, None, None], system_poles[None, None, :, :]))
            
        if self.safe_mode:
            term1_real = torch.nan_to_num(term1.real, nan=0.0)
            term1_imag = torch.nan_to_num(term1.imag, nan=0.0)
            term1 = torch.complex(term1_real, term1_imag)
        if DEBUG:
            assert torch.isnan(term1).any()==False, "term1 contains NaN"

        if self.num_sigma:
            # term1 [batch, sigma, num_eig, numpole, channels]
            Hw = torch.einsum("pc,bskpc->bskpc", system_residues, term1)
            # [batch, sigma*num_eig, numpole, channels]
            
            # output_residue1 = α * H(λ) (periodic)
            # output_residue1 = torch.einsum("bkc,bkpc->bkc", x_spec, Hw)
            # output_residue2 = α * H(μ) (aperiodic)
            output_residue2 = torch.einsum("bskc,bskpc->bpskc", x_spec, -Hw)
            output_residue1= -output_residue2.sum(dim=1)
        else:
            Hw = torch.einsum("pc,bkpc->bkpc", system_residues, term1)
            output_residue2 = torch.einsum("bkc,bkpc->bpkc", x_spec, -Hw)
            output_residue1= -output_residue2.sum(dim=1)
        return output_residue1, output_residue2

    def cal(self, x_spec, evals, evecs, basis, pos, geo_feat, system_poles, system_residues, eigen_gate=None):
        #x_spec=[batch,sigma*num_eig,inc]

        # Compute output residues
        output_residue1, output_residue2 = self.PoleRes(
            evals, x_spec, system_poles, system_residues
        )
        
        if DEBUG:
            assert torch.isnan(output_residue1).any()==False, "output_residue1 contains NaN"
            assert torch.isnan(output_residue2).any()==False, "output_residue2 contains NaN"

        # Transform back to per-vertex
        if self.period:
            # output_residue1=[batch,sigma*num_eig,outchannel]
            x1_spec = torch.real(output_residue1)
            x1 = from_basis(x1_spec, evecs)
            # x1=[batch,sigma,num_vec,C]
            if DEBUG:
                assert torch.isnan(x1).any()==False, "x1 contains NaN"

            x_period = x1
            if self.num_sigma:
                x_period = torch.einsum("bsnc,bsnc->bnc", x1, basis)

            if DEBUG:
                assert torch.isnan(x1_spec).any()==False, "x1_spec contains NaN"
                assert torch.isnan(x_period).any()==False, "x_period contains NaN"

            if self.period_norm:
                x_period=self.norm_period(x_period)
            elif self.num_sigma:
                x_period=x_period/self.num_sigma
            
        if self.aperiod:
            x_aperiod = self.pole_to_operator(system_poles, evecs, output_residue2, geo_feat, pos=pos, eigen_gate=eigen_gate)
            if DEBUG:
                assert torch.isnan(x_aperiod).any()==False, "x_aperiod contains NaN"
        
            if self.aperiod_norm:
                x_aperiod=self.norm_aperiod(x_aperiod)
            elif self.num_sigma:
                x_aperiod=x_aperiod/self.num_sigma #原来没有这个
        
        x_glno = (x_period if self.period else 0) + (x_aperiod if self.aperiod else 0)
        # x_glno = torch.nan_to_num(x_glno, nan=0.0)
        if DEBUG:
            assert torch.isnan(x_glno).any()==False
        
        return x_glno
    
    def forward(self, x, mass, evals, evecs, geo_feat, pos=None):
        if x.shape[-1] != self.C_inout:
            raise ValueError(
                "Tensor has wrong shape = {}. Last dim shape should have number of channels = {}".format(
                    x.shape, self.C_inout))

        # Transform to spectral
        batch_size = x.shape[0] # B N C

        if evals.shape[-1]<self.LT_kmax:
            raise ValueError("evals.shape[-1]<self.LT_kmax")

        if self.normalize_evals:
            evals = (evals / evals[...,self.normalize_k-1].unsqueeze(-1)) * self.normalize_evals
        elif self.scale_evals:
            evals = evals * self.scale_evals

        # mask = (evals >= self.LT_kmin) & (evals <= self.LT_kmax)
        # mask = mask.squeeze(0)
        # evals = evals[..., mask]
        # evecs = evecs[..., mask]

        evals=evals[...,self.LT_kmin:self.LT_kmax]
        evecs=evecs[...,self.LT_kmin:self.LT_kmax]       
        
        x_scale=x.unsqueeze(1) # (B, 1, V, C) 直接与 basis 广播
        basis=None
        if self.num_sigma:
            basis = torch.exp(torch.einsum("bn,sc->bsnc",geo_feat,self.sigma)) #非周期basis
        
            if self.normalize_basis:
                basis = basis / torch.norm(basis, dim=-2, keepdim=True) #normalize basis
                
            # x: (B, V, C) → 扩展为 (B, 1, V, C) 直接与 basis 广播
            # x_scale = x[:, None, :, :] * basis   # 结果 (B, sigma, V, C)
            x_scale=torch.einsum("bnc,bsnc->bsnc",x,basis)
        x_spec = to_basis(x_scale, evecs, mass)
        if self.num_sigma is None:
            x_spec=x_spec.squeeze(1) #如果没有sigma维度，就去掉这个维度，变成(batch,num_eig,inc)
        # x_spec=[batch,sigma,num_eig,inc]

        if DEBUG:
            assert torch.isnan(x_scale).any()==False, "x_scale contains NaN"
            # assert torch.isnan(basis).any()==False, "basis contains NaN"
            assert torch.isnan(x_spec).any()==False, "x_spec contains NaN"
        
        x_out=self.cal(x_spec, evals, evecs, basis, pos, geo_feat, self.system_poles, self.system_residues)
            
        if self.glno_norm:
            x_out=self.norm_glno(x_out)
            
        return x_out  ##分开来并不能变好
    
class MLP(nn.Sequential):
    '''
    A simple MLP with configurable hidden layer sizes.
    '''
    def __init__(self, layer_sizes, dropout=False, norm='layer', act='relu', name="MLP"):
        super(MLP, self).__init__()
        activation = {'gelu': nn.GELU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh(),
                      'sigmoid': nn.Sigmoid(), 'leaky_relu': nn.LeakyReLU(),
                      'elu': nn.ELU(), 'softplus': nn.Softplus(), 'silu': nn.SiLU()}.get(act, None)
        
        for i in range(len(layer_sizes) - 1):
            is_last = (i + 2 == len(layer_sizes))

            if dropout and i > 0:
                self.add_module(
                    name + "_layer_dropout_{:03d}".format(i),
                    nn.Dropout(p=dropout)
                )

            self.add_module(
                name + "_layer_{:03d}".format(i),
                nn.Linear(layer_sizes[i], layer_sizes[i + 1]),
            )

            if not is_last and norm: ##增加了not is_last
                self.add_module(
                    name + f"_{norm}norm_{i:03d}",
                    NormLayer(norm, layer_sizes[i + 1])
                )   #原来是layer

            # Nonlinearity
            # (but not on the last layer)
            if not is_last and activation is not None:
                self.add_module(
                    name + "_act_{:03d}".format(i),
                    activation
                )
    
class NormLayer(nn.Module):
    def __init__(self, method, normalized_shape, affine=True):
        super(NormLayer, self).__init__()
        self.method = method
        self.normalized_shape = normalized_shape   # 可以是 int 或 tuple

        if method == 'layer':
            # LayerNorm 直接支持任意形状的 normalized_shape
            self.norm = nn.LayerNorm(normalized_shape, elementwise_affine=affine)

        elif method in ('batch', 'instance', 'group'):
            # 对于这些方法，要求 normalized_shape 是整数（或只包含一个整数的元组）
            if isinstance(normalized_shape, (tuple, list)):
                if len(normalized_shape) != 1:
                    raise ValueError(f"{method} norm only supports single channel dimension, got {normalized_shape}")
                num_features = normalized_shape[0]
            else:
                num_features = normalized_shape
            self.num_features = num_features

            # 公共的通道置换层（因为 BatchNorm1d/InstanceNorm1d/GroupNorm 要求形状为 [B, C, L]）
            self.rearrange_to_bcn = Rearrange('b n c -> b c n')
            self.rearrange_to_bnc = Rearrange('b c n -> b n c')

            if method == 'batch':
                self.norm = nn.Sequential(
                    self.rearrange_to_bcn,
                    nn.BatchNorm1d(num_features, track_running_stats=True, affine=affine),
                    self.rearrange_to_bnc
                )
            elif method == 'instance':
                self.norm = nn.Sequential(
                    self.rearrange_to_bcn,
                    nn.InstanceNorm1d(num_features, affine=affine),
                    self.rearrange_to_bnc
                )
            elif method == 'group':
                # GroupNorm 需要指定组数，这里保持原逻辑（num_groups=16）
                self.norm = nn.Sequential(
                    self.rearrange_to_bcn,
                    nn.GroupNorm(num_groups=16, num_channels=num_features, affine=affine),
                    self.rearrange_to_bnc
                )
        elif method is None:
            self.norm = None
        else:
            raise ValueError("invalid norm type")

    def forward(self, x):
        # 处理输入维度不足3维的情况（如 [N, C] -> 添加 batch 维）
        flag = (x.dim() < 3)
        if flag:
            x = x.unsqueeze(0)

        if self.norm is not None:
            if self.method == 'layer':
                # LayerNorm：直接应用，它会自动处理最后 len(normalized_shape) 维
                x = self.norm(x)
            else:
                # BatchNorm/InstanceNorm/GroupNorm：输入必须为 [B, N, C]，且最后一维等于 num_features
                if x.shape[-1] != self.num_features:
                    # 如果最后一维不匹配但倒数第二维匹配，尝试转置（兼容旧代码）
                    if x.shape[-2] == self.num_features:
                        x = x.transpose(-1, -2)
                    else:
                        raise ValueError(f"Expected last dim {self.num_features}, got {x.shape[-1]}")
                x = self.norm(x)

        if flag:
            x = x.squeeze(0)
        return x

class SpatialGradientFeatures(nn.Module):
    """
    Compute dot-products between input vectors.
    Uses a learned complex-linear layer to keep dimension down.
    """
    def __init__(self, in_channels, with_gradient_rotations=True):
        """
        Parameters:
            in_channels (int): number of input channels.
            with_gradient_rotations (bool): whether with gradient rotations. Default True.
        """
        super(SpatialGradientFeatures, self).__init__()

        self.in_channels = in_channels
        self.with_gradient_rotations = with_gradient_rotations

        if self.with_gradient_rotations:
            self.A_re = nn.Linear(self.in_channels, self.in_channels, bias=False)
            self.A_im = nn.Linear(self.in_channels, self.in_channels, bias=False)
        else:
            self.A = nn.Linear(self.in_channels, self.in_channels, bias=False)

    def forward(self, feat_in):
        """
        Input:
            feat_in (B,Nv,C,2)
        Output:
            feat_out (B,Nv,C)
        """
        feat_a = feat_in

        if self.with_gradient_rotations:
            feat_real_b = self.A_re(feat_in[..., 0]) - self.A_im(feat_in[..., 1])
            feat_img_b = self.A_re(feat_in[..., 0]) + self.A_im(feat_in[..., 1])
        else:
            feat_real_b = self.A(feat_in[..., 0])
            feat_img_b = self.A(feat_in[..., 1])

        feat_out = feat_a[..., 0] * feat_real_b + feat_a[..., 1] * feat_img_b

        return torch.tanh(feat_out)

class GLNOBlock(nn.Module):
    def __init__(self, idx, config):
        super(GLNOBlock, self).__init__()

        self.C_width = config["C_width"]
        self.idx=idx

        # Operator
        self.LT = Laplace_Transform_Layer(config=config)

        self.learned_geo_feat=config['learned_geo_feat']
        if self.learned_geo_feat:
            self.geo_feat_dim=config['geo_feat_dim']
            self.geo_hidden_layers=config.get('geo_feat_hidden_layers',[self.C_width*2,self.C_width*2])
            self.geo_dropout=config.get('geo_feat_dropout',0.0)
            self.geo_norm=config.get('geo_feat_norm',None)
            self.geo_activation=config.get('geo_feat_activation',None)
            self.mlp_geo=MLP([self.geo_feat_dim]+self.geo_hidden_layers+[1], dropout=self.geo_dropout, norm=self.geo_norm,act=self.geo_activation)
            self.last_geo_norm=NormLayer(config.get('geo_feat_last_norm',None),1)

        self.high_fre=config['high_fre']
        if self.high_fre:
            self.k_eig_high_fre=config['k_eig_high_fre']
            self.high_fre_hidden_dims=config.get("high_fre_hidden_layers",[self.C_width])
            self.high_fre_dropout=config.get("high_fre_dropout",0.0)
            self.high_fre_norm=config.get("high_fre_norm",None)
            self.high_fre_activation=config.get("high_fre_activation",None)
            self.mlp_high=MLP([self.C_width]+ self.high_fre_hidden_dims + [self.C_width], dropout=self.high_fre_dropout, norm=self.high_fre_norm,act=self.high_fre_activation)

        self.connect_method = config["connect_method"]

        if self.connect_method:
            self.mlp_hidden_dims = config.get("mlp_hidden_layers",[self.C_width,self.C_width])
            self.mlp_dropout = config.get("mlp_dropout",0.0)
            self.mlp_norm = config.get("mlp_norm",None)
            if config.get('mlp_norm_last_disable',None)==True and idx < config['blocks']:
                self.mlp_norm=None
            self.mlp_activation = config.get("mlp_activation",None)
            self.MLP_C = 2* self.C_width
            # MLPs
            if self.connect_method=='mlp':
                self.mlp = MLP([self.MLP_C] + self.mlp_hidden_dims + [self.C_width], dropout=self.mlp_dropout, norm=self.mlp_norm,act=self.mlp_activation)
            else:
                raise ValueError("invalid connect_method")

        self.norm_feature=NormLayer(config["norm_feature"], self.MLP_C)        
        
        self.skip=config["skip"]
        
        self.diffuse_type=config.get("diffuse_type",None)
        if self.diffuse_type=='hks':
            self.t=nn.Parameter(torch.rand(self.C_width, dtype=torch.float))
        elif self.diffuse_type=='wks':
            self.e_N=nn.Parameter(torch.rand(self.C_width, dtype=torch.float))
            self.sigma=nn.Parameter(torch.rand(self.C_width, dtype=torch.float))
        elif self.diffuse_type is not None:
            raise ValueError("invalid diffuse_type")

        self.with_gradient_features=config.get("with_gradient_features",False)
        if self.with_gradient_features:
            self.gradient_features = SpatialGradientFeatures(self.C_width, with_gradient_rotations=config.get("with_gradient_rotations",True))

        self.k_max=config.get('k_max',128)
        self.k_min=config.get('k_min',0)

    def spectral_diffuse(self, x, evecs, evals, mass, filter_type, **kwargs):
        """
        对顶点特征 x 进行谱域扩散（HKS 或 WKS）。

        参数:
            x: (N, C) 或 (N, 1) 顶点特征，N 顶点数，C 通道数
            evecs: (N, K) 前 K 个特征向量（列向量）
            evals: (K,) 对应的特征值
            mass: (N,) 顶点质量（对角质量矩阵，用于 L2 内积）
            filter_type: 字符串，'hks' 或 'wks'
            **kwargs: 
                - t: 浮点数（HKS 时间参数）
                - e_N: 浮点数（WKS 对数能量中心）
                - sigma: 浮点数（WKS 高斯带宽）
        返回:
            x_filtered: (N, C) 扩散后的特征
        """
        evals=evals[...,self.k_min:self.k_max]
        evecs=evecs[...,self.k_min:self.k_max]
        if filter_type == 'hks':
            t = kwargs.get('t', 1.0)
            weights = torch.exp(-evals[:,:,None] * t[None,None,:])          # (K_use,)
        elif filter_type == 'wks':
            e_N = kwargs.get('e_N', 0.0)
            sigma = kwargs.get('sigma', 0.1)
            # 防止 log(0)
            log_evals = torch.log(evals + 1e-12)
            weights = torch.exp(-((e_N[None,None,:] - log_evals[:,:,None]) ** 2) / (2 * sigma[None,None,:] ** 2))
            # 可选：归一化（与 WKS 一致，但作为滤波器可不做）
            # weights = weights / weights.sum()
        else:
            raise ValueError("filter_type must be 'hks' or 'wks'")

        # 对每个通道进行谱滤波
        return from_basis(weights*to_basis(x, evecs, mass),evecs)
        
    def forward(self, x_in, mass, evals, evecs, geo_feat, pos=None, edges=None, **kwargs):
        if x_in.shape[-1] != self.C_width:
            raise ValueError(
                "Tensor has wrong shape = {e}. Last dim shape should have number of channels = {}".format(
                    x_in.shape, self.C_width))
        if self.learned_geo_feat:
            if len(geo_feat.shape)==2:
                geo_feat=geo_feat.unsqueeze(-1)
            geo_feat=self.mlp_geo(geo_feat)
            if self.last_geo_norm:
                geo_feat=self.last_geo_norm(geo_feat)
            geo_feat=geo_feat.squeeze(-1)

        x_glno = self.LT(x_in, mass, evals, evecs, geo_feat,pos)
        
        if self.high_fre:
            if evecs.shape[-1]<self.k_eig_high_fre:
                raise ValueError("k_eig_high_fre should be less than or equal to the number of eigenvectors")
            evecs=evecs[...,:self.k_eig_high_fre]
            x_low=from_basis(to_basis(x_in.unsqueeze(1), evecs, mass), evecs).squeeze(1)
            x_high=x_in-x_low
            x_high=self.mlp_high(x_high)
            x_glno+=x_high

        if self.diffuse_type:
            x_glno=self.spectral_diffuse(x_glno, evecs, evals, mass, self.diffuse_type, 
                                         t=self.t if self.diffuse_type=='hks' else None, 
                                         e_N=self.e_N if self.diffuse_type=='wks' else None, sigma=self.sigma if self.diffuse_type=='wks' else None)
        
        if DEBUG:
            assert torch.isnan(x_glno).any()==False, "LT output contains NaN"
        
        if len(geo_feat.shape)==2:
            geo_feat=geo_feat.unsqueeze(-1)
            
        if self.with_gradient_features:
            # Compute gradient
            feat_grads = []
            B = x_in.shape[0]
            gradX=kwargs.get("gradX",None)
            gradY=kwargs.get("gradY",None)
            if gradX is None or gradY is None:
                raise ValueError("gradX and gradY must be provided when with_gradient_features is True")
            for b in range(B):
                # gradient after diffusion
                feat_gradX = torch.mm(gradX[b, ...], x_glno[b, ...])
                feat_gradY = torch.mm(gradY[b, ...], x_glno[b, ...])

                feat_grads.append(torch.stack((feat_gradX, feat_gradY), dim=-1))
                
            feat_grad = torch.stack(feat_grads, dim=0) # [B, V, C, 2]

            # Compute gradient features
            feat_grad_features = self.gradient_features(feat_grad)

            # Stack inputs to MLP
            feature_combined = torch.cat((x_in, feat_grad_features), dim=-1)
        else:
            feature_combined = torch.cat((x_in, x_glno), dim=-1) #high还是in好像也no difference， xin比较good #这里加x_high是useless
        feature_combined = self.norm_feature(feature_combined)         
        
        # Apply the connect method
        if self.connect_method=='mlp':
            x0_out = self.mlp(feature_combined)
        else:
            raise ValueError("invalid connect_method")

        # Skip connection
        if self.skip:
            x0_out = x0_out + x_in

        return x0_out


class GLNONet(nn.Module):
    def __init__(self, config):
        """
        Construct a GLNONet.

        Parameters loaed from config:
            C_in (int):                     input dimension
            C_out (int):                    output dimension
            last_activation (str)          a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax (default: None)
            outputs_at (string)             produce outputs at various mesh elements by averaging from vertices. One of ['vertices', 'edges', 'faces', 'global_mean']. (default 'vertices', aka points for a point cloud)
            C_width (int):                  dimension of internal DiffusionNet blocks (default: 128)
            N_block (int):                  number of DiffusionNet blocks (default: 4)
            mlp_hidden_dims (list of int):  a list of hidden layer sizes for MLPs (default: [C_width, C_width])
            dropout (bool):                 if True, internal MLPs use dropout (default: True)
            diffusion_method (str):      how to evaluate diffusion, one of ['spectral', 'implicit_dense']. If implicit_dense is used, can set k_eig=0, saving precompute.
            """

        super(GLNONet, self).__init__()

        # Load parameters
        self.device=config["device"]
        
        self.C_in = config["C_in"]
        self.C_width = config["C_width"]
        self.N_block = config["blocks"]
        self.C_out = config["C_out"]
        self.vertices_dim = config["vertices_dim"]
        self.rotate = config["rotate"]
        self.geo_pre = config.get("geo_preprocess",None)
        self.k_eig=config['k_eig']

        if config["last_activation"] is None:
            self.last_activation = None
        elif config["last_activation"] == "log_softmax":
            self.last_activation = nn.LogSoftmax(dim=-1)
        else:
            raise ValueError("invalid setting for last_activation")
        
        self.outputs_at = config["outputs_at"]
        if self.outputs_at not in ['vertices', 'edges', 'faces', 'global_mean']:
            raise ValueError("invalid setting for outputs_at")
        
        ## Set up the network
        self.encoder_method=config.get('encoder',None)
        if self.encoder_method is None:
            self.encoder = nn.Linear(self.C_in, self.C_width)
        else:
            self.encoder_dropout=config["encoder_dropout"]
            self.encoder_norm=config["encoder_norm"]
            self.encoder_hidden_dims=config["encoder_hidden_layers"]
            self.encoder_activation=config["encoder_activation"]
            if config["encoder"]=='mlp':
                self.encoder = MLP([self.C_in] + self.encoder_hidden_dims + [self.C_width], dropout=self.encoder_dropout, norm=self.encoder_norm,act=self.encoder_activation)
            else:
                raise ValueError("invalid encoder method")
        
        if config["decoder"]=='mlp':
            self.decoder_dropout=config["decoder_dropout"]
            self.decoder_norm=config["decoder_norm"]
            self.decoder_hidden_dims=config["decoder_hidden_layers"]
            self.decoder_activation=config["decoder_activation"]
            self.decoder = MLP([self.C_width*self.N_block] + self.decoder_hidden_dims + [self.C_out], dropout=self.decoder_dropout, norm=self.decoder_norm,act=self.decoder_activation)
        elif config['decoder'] is None:
           self.decoder=nn.Linear(self.C_width, self.C_out)
        else: 
            raise ValueError("invalid decoder method")
        
        self.blocks = []
        for i_block in range(self.N_block):
            block = GLNOBlock(i_block,config=config)
            self.blocks.append(block)
            self.add_module("block_" + str(i_block), self.blocks[-1])

    def forward(self, data):
        """
        In the notation in this document, dimension are:
            - C: channel dimension (C_in/C_out on construction)
            - N: the number of vertices/points, which CAN be different for each forward pass
            - B: is an OPTIONAL batch dimension
            - K/K_EIG: is the number of eigenvalues used for spectral acceleration
            - V is vertices_dim at most 3

        Parameters:
            data should be a dictionary with the following keys:
                vertice (required):   tensor of vertex positions, dimension [N,V] or [B,N,V]
                input:     tensor of input features, dimension [N,C] or [B,N,C]
                faces:     tensor of face indices, dimension [F,3] or [B,F,3]
                edges:     tensor of edge indices, dimension [E,2] or [B,E,2]
                mass (required):      Mass vector, dimension [N] or [B,N]
                evals (required):     Eigenvalues of Laplace matrix, dimension [K_EIG] or [B,K_EIG]
                evecs (required):     Eigenvectors of Laplace matrix, dimension [N,K_EIG] or [B,N,K_EIG]
                geo_feat (required):   tensor of geometric features, dimension [N,C] or [B,N,C] (C>0 if learned geometric features block is used)
            
        Returns:
            x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """
        ## Real data
        x_in=None
        if "input" in data:
            x_in=data["input"].to(self.device)
        if "mass" in data:
            mass = data["mass"].to(self.device)
        else:
            raise ValueError("input data must contain'mass' key")
        if "evals" in data:
            evals = data["evals"].to(self.device)
            if evals.shape[-1]<self.k_eig:
                raise ValueError("input evals must have at least k_eig elements")
        else:
            raise ValueError("input data must contain 'evals' key")
        if "evecs" in data:
            evecs = data["evecs"].to(self.device)
        else:
            raise ValueError("input data must contain 'evecs' key")
        if "geo_feat" in data:
            geo_feat = data["geo_feat"].to(self.device)
            if self.geo_pre=='exp':
                geo_feat=torch.exp(-geo_feat)
            elif self.geo_pre=='max':
                geo_feat=geo_feat/torch.max(geo_feat,dim=-1,keepdim=True)
            elif self.geo_pre is not None:
                raise ValueError("geometry feature preprocessing method not recognized")
        else:
            raise ValueError("input data must contain 'geo_feat' key")
        
        edges,faces,vertices,gradX,gradY=None,None,None,None,None
        if "edges" in data:
            edges = data["edges"].to(self.device)
        if "faces" in data:
            faces = data["faces"].to(self.device)
        if "vertices" in data:
            vertices = data["vertices"].to(self.device)
            vertices = vertices[...,:self.vertices_dim]

            if self.rotate:
                vertices=rotate(vertices)
            if x_in is not None:
                x_in=torch.cat([x_in,vertices],dim=-1)
            else:
                x_in=vertices
        if "gradX" in data and "gradY" in data:
            gradX = data["gradX"].to(self.device)
            gradY = data["gradY"].to(self.device)

        if x_in is None:
            raise ValueError("input data must contain 'input' or 'vertices' key")
        
        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.C_in:
            raise ValueError(
                "DiffusionNet was constructed with C_in={}, but x_in has last dim={}".format(self.C_in, x_in.shape[-1]))

        if len(x_in.shape) == 2: # add a batch dim
            appended_batch_dim = True
            x_in = x_in.unsqueeze(0)
            mass = mass.unsqueeze(0)
            evals = evals.unsqueeze(0)
            evecs = evecs.unsqueeze(0)
            geo_feat = geo_feat.unsqueeze(0)
            if vertices is not None: vertices = vertices.unsqueeze(0)
            if edges is not None: edges = edges.unsqueeze(0)
            if faces is not None: faces = faces.unsqueeze(0)
            if gradX is not None: gradX = gradX.unsqueeze(0)
            if gradY is not None: gradY = gradY.unsqueeze(0)
        elif len(x_in.shape) == 3:
            appended_batch_dim = False
        else:
            raise ValueError("x_in should be tensor with shape [N,C] or [B,N,C]")

        ## Forward pass through the network
        if DEBUG:
            assert torch.isnan(x_in).any() == False, "input contains NaNs"
            assert torch.isnan(mass).any() == False, "mass contains NaNs"
            assert torch.isnan(evals).any() == False, "evals contains NaNs"
            assert torch.isnan(evecs).any() == False, "evecs contains NaNs"
            assert torch.isnan(geo_feat).any() == False, "geo_feat contains NaNs"
        
        x = self.encoder(x_in)
        if DEBUG:
            assert torch.isnan(x).any() == False, "encoder output contains NaNs"
        for b in self.blocks:
            x = b(x, mass, evals, evecs, geo_feat, vertices, edges, gradX=gradX, gradY=gradY)
        if DEBUG:
            assert torch.isnan(x).any() == False, "block output contains NaNs"
        x = self.decoder(x)
        if DEBUG:
            assert torch.isnan(x).any() == False, "decoder output contains NaNs"

        ## Remap output to requested output type
        if self.outputs_at == 'vertices':
            x_out = x

        elif self.outputs_at == 'edges':
            if edges is None:
                raise ValueError("edges must be provided for outputs_at='edges'")
            # Remap to edges
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 2)
            edges_gather = edges.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xe = torch.gather(x_gather, 1, edges_gather)
            x_out = torch.mean(xe, dim=-1)

        elif self.outputs_at == 'faces':
            if faces is None:
                raise ValueError("faces must be provided for outputs_at='faces'")
            # Remap to faces
            x_gather = x.unsqueeze(-1).expand(-1, -1, -1, 3)
            faces_gather = faces.unsqueeze(2).expand(-1, -1, x.shape[-1], -1)
            xf = torch.gather(x_gather, 1, faces_gather)
            x_out = torch.mean(xf, dim=-1)

        elif self.outputs_at == 'global_mean':
            # Produce a single global mean ouput using a weighted mean according to the point mass/area which is discretization-invariant.
            x_out = torch.sum(x * mass.unsqueeze(-1), dim=-2) / torch.sum(mass, dim=-1, keepdim=True)

        # Apply last nonlinearity if specified
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0)

        return x_out

