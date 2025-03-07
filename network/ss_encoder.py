#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch_scatter
from torch.nn import Linear, Sequential, ModuleList, BatchNorm1d, Dropout, LeakyReLU, ReLU
import torch_geometric as tg

class CustomBRepEncoder(torch.nn.Module):
    def __init__(self, v_in_width, e_in_width, f_in_width, out_width, num_layers, use_attention=False):
        super().__init__()
        self.use_attention = use_attention

        # Initial feature embedding layers for vertices, edges, and faces
        self.embed_v_in = LinearBlock(v_in_width, out_width)
        self.embed_e_in = LinearBlock(e_in_width, out_width)
        self.embed_f_in = LinearBlock(f_in_width, out_width)

        # Message passing layers for encoding hierarchical structure
        self.V2E = BipartiteResMRConv(out_width)
        self.E2F = BipartiteResMRConv(out_width)

        # Additional message passing layers to refine features
        self.message_layers = ModuleList([BipartiteResMRConv(out_width) for _ in range(num_layers)])

        # Attention mechanism for handling varied neighborhood sizes
        if self.use_attention:
            self.attention_layers = ModuleList([tg.nn.GATConv(out_width, out_width//4, heads=4) for _ in range(num_layers)])

    def forward(self, data):
        x_v = self.embed_v_in(data.vertices)
        x_e = self.embed_e_in(data.edges)
        x_f = self.embed_f_in(data.faces)

        print("x_v shape:", x_v.shape)
        print("x_e shape:", x_e.shape)
        print("x_f shape:", x_f.shape)
        print("data.edge_to_vertex:", data.edge_to_vertex)
        print("data.face_to_edge:", data.face_to_edge)

        # 检查 face_to_edge 是否越界
        max_face_idx = data.face_to_edge[0].max().item()
        max_edge_idx = data.face_to_edge[1].max().item()

        print("Max face index in face_to_edge:", max_face_idx)
        print("Max edge index in face_to_edge:", max_edge_idx)

        if max_face_idx >= x_f.shape[0] or max_edge_idx >= x_e.shape[0]:
            raise IndexError(f"Index out of range: max_face_idx={max_face_idx}, max_edge_idx={max_edge_idx}")

        # Upward pass: propagate information from vertices to edges, and from edges to faces
        x_e = self.V2E(x_v, x_e, data.edge_to_vertex[[1, 0]])
        x_f = self.E2F(x_e, x_f, data.face_to_edge[[1, 0]])

        # Refinement through additional message passing layers
        for i, layer in enumerate(self.message_layers):
            if self.use_attention:
                x_f = self.attention_layers[i](x_f, data.face_to_face[:2, :])
            else:
                x_f = layer(x_f, x_f, data.face_to_face[:2, :])

        return x_f  # Return the final face embeddings

class BipartiteResMRConv(torch.nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = LinearBlock(2 * width, width)

    def forward(self, x_src, x_dst, e):
        diffs = torch.index_select(x_dst, 0, e[1]) - torch.index_select(x_src, 0, e[0])
        maxes, _ = torch_scatter.scatter_max(diffs, e[1], dim=0, dim_size=x_dst.shape[0])
        return x_dst + self.mlp(torch.cat([x_dst, maxes], dim=1))

# LinearBlock with flexibility for configurations
class LinearBlock(torch.nn.Module):
    def __init__(self, *layer_sizes, batch_norm=False, dropout=0.0, last_linear=False, leaky=True):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            c_in = layer_sizes[i]
            c_out = layer_sizes[i + 1]
            layers.append(Linear(c_in, c_out))
            if last_linear and i + 1 >= len(layer_sizes) - 1:
                break
            if batch_norm:
                layers.append(BatchNorm1d(c_out))
            if dropout > 0:
                layers.append(Dropout(p=dropout))
            layers.append(LeakyReLU() if leaky else ReLU())
        self.f = Sequential(*layers)

    def forward(self, x):
        return self.f(x)

