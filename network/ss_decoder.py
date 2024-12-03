#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDecoder(nn.Module):
    def __init__(self, latent_size, hidden_dims, uv_input_dim=2, output_dim=4):
        """
        条件解码器，用于学习显式曲面和隐式裁剪掩码的联合表示。

        参数:
            latent_size (int): 从编码器生成的潜在向量的大小。
            hidden_dims (list): 隐藏层的维度列表。
            uv_input_dim (int): (u, v) 输入坐标的维度，默认为2。
            output_dim (int): 输出的维度，默认为4（[x, y, z, d]）。
        """
        super().__init__()
        self.latent_size = latent_size
        self.uv_input_dim = uv_input_dim
        self.output_dim = output_dim

        # 输入为潜在向量和 (u, v) 坐标，大小为 latent_size + uv_input_dim
        input_dim = latent_size + uv_input_dim

        # 构建全连接层
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # 隐藏层使用ReLU激活
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    def forward(self, uv_coords, latent_vector):
        """
        前向传播函数。
    
        参数:
            uv_coords (torch.Tensor): (u, v) 坐标，形状为 (batch_size, uv_input_dim)。
            latent_vector (torch.Tensor): 编码器的输出潜在向量，形状为 (latent_size)。
    
        返回:
            torch.Tensor: 解码器输出，形状为 (batch_size, output_dim)，
                          包括显式3D曲面坐标 (x, y, z) 和 SDF 值 (d)。
        """
        # 扩展 latent_vector 的维度并重复，使其匹配 uv_coords 的第一个维度
        latent_vector = latent_vector.unsqueeze(0).repeat(uv_coords.shape[0], 1)
    
        # 将潜在向量和 (u, v) 坐标拼接
        x = torch.cat([latent_vector, uv_coords], dim=-1)
    
        # 通过神经网络
        output = self.network(x)
        return output

