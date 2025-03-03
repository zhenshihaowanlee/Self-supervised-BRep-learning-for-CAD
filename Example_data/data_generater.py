#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import math

class CuboidData:
    def __init__(self, width, height, depth):
        """
        初始化长方体数据
        :param width: 长方体的宽度（x 方向）
        :param height: 长方体的高度（y 方向）
        :param depth: 长方体的深度（z 方向）
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.vertices = self.generate_vertices()  # 顶点特征
        self.edges = self.generate_edges()  # 边索引
        self.faces = self.generate_faces()  # 面特征
        self.edge_to_vertex = self.generate_edge_to_vertex()  # 边到顶点关系
        self.face_to_edge = self.generate_face_to_edge()  # 面到边关系
        self.face_to_face = self.generate_face_to_face()  # 面到面关系

    def generate_vertices(self):
        """
        生成长方体的顶点特征（3 维）
        :return: 顶点特征的张量
        """
        vertices = torch.tensor([
            [0, 0, 0], [self.width, 0, 0], [self.width, self.height, 0], [0, self.height, 0],  # 底面
            [0, 0, self.depth], [self.width, 0, self.depth], [self.width, self.height, self.depth], [0, self.height, self.depth]  # 顶面
        ], dtype=torch.float32)
        return vertices

    def generate_edges(self):
            """
            生成长方体的边特征（5 维）
            :return: 边特征张量
            """
            # 边的起点和终点索引
            edge_indices = torch.tensor([
                [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
                [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
                [0, 4], [1, 5], [2, 6], [3, 7]   # 垂直边
            ], dtype=torch.long)
    
            # 生成边的特征，直接替换 edges 为 5 维特征
            edges = torch.cat([
                edge_indices.float(),  # 前两列：起点和终点的索引（浮点型以兼容其他特征）
                torch.norm(self.vertices[edge_indices[:, 1]] - self.vertices[edge_indices[:, 0]], dim=1, keepdim=True),  # 边的长度 (1维)
                ((self.vertices[edge_indices[:, 1]] + self.vertices[edge_indices[:, 0]]) / 2),  # 边的中点位置 (3维)
            ], dim=1)  # Shape: [12, 5]
    
            return edges

    def generate_faces(self):
        """
        生成长方体的面特征（7 维）
        :return: 面特征张量
        """
        face_normals = torch.tensor([
            [0, 0, -1],  # 底面
            [0, 0, 1],   # 顶面
            [-1, 0, 0],  # 左面
            [1, 0, 0],   # 右面
            [0, -1, 0],  # 前面
            [0, 1, 0]    # 后面
        ], dtype=torch.float32)

        face_areas = torch.tensor([
            self.width * self.height,  # 底面面积
            self.width * self.height,  # 顶面面积
            self.depth * self.height,  # 左面面积
            self.depth * self.height,  # 右面面积
            self.width * self.depth,  # 前面面积
            self.width * self.depth   # 后面面积
        ], dtype=torch.float32).view(-1, 1)

        face_centers = torch.tensor([
            [self.width / 2, self.height / 2, 0],                  # 底面中心
            [self.width / 2, self.height / 2, self.depth],         # 顶面中心
            [0, self.height / 2, self.depth / 2],                 # 左面中心
            [self.width, self.height / 2, self.depth / 2],        # 右面中心
            [self.width / 2, 0, self.depth / 2],                 # 前面中心
            [self.width / 2, self.height, self.depth / 2]         # 后面中心
        ], dtype=torch.float32)

        faces = torch.cat([face_normals, face_areas, face_centers], dim=1)  # Shape: [6, 7]
        return faces

    def generate_edge_to_vertex(self):
        """
        生成边到顶点的关系
        :return: 边到顶点关系张量
        """
        edge_to_vertex = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3],
            [1, 2, 3, 0, 5, 6, 7, 4, 4, 5, 6, 7]
        ], dtype=torch.long)
        return edge_to_vertex

    def generate_face_to_edge(self):
        """
        生成面到边的关系
        :return: 面到边关系张量
        """
        face_to_edge = torch.tensor([
            [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5],  # 面索引
            [0, 1, 2, 3, 4, 5, 6, 7, 0, 4, 8, 11, 1, 5, 9, 10, 2, 6, 10, 8, 3, 7, 11, 9]  # 边索引
        ], dtype=torch.long)
        return face_to_edge

    def generate_face_to_face(self):
        """
        生成面到面的关系
        :return: 面到面关系张量
        """
        face_to_face = torch.tensor([
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],  # 起点面索引
            [2, 3, 4, 2, 3, 5, 0, 1, 4, 0, 1, 5, 0, 1, 2, 3, 4, 5],  # 终点面索引
        ], dtype=torch.long)
        return face_to_face


# In[ ]:


class CylinderData:
    def __init__(self, radius, height):
        """
        :param radius: 圆柱体的底面半径
        :param height: 圆柱体的高度
        :param num_segments: 圆周方向的分段数
        """
        self.radius = radius
        self.height = height
        self.vertices = self.generate_vertices()  # 顶点特征
        self.edges = self.generate_edges()  # 边特征
        self.faces = self.generate_faces()  # 面特征
        self.edge_to_vertex = self.generate_edge_to_vertex()  # 边到顶点关系
        self.face_to_edge = self.generate_face_to_edge()  # 面到边关系
        self.face_to_face = self.generate_face_to_face()  # 边到边关系

    def generate_vertices(self):
        """
        生成圆柱体的顶点特征
        :return: 顶点特征的张量
        """
        vertices = torch.tensor([
            [0, 0, 0], [0, 0, self.height]
        ], dtype=torch.float32)
        return vertices

    def generate_edges(self):
        """
        生成圆柱体的边特征
        :return: 边特征张量
        """
        edges = torch.tensor([
            [0, 0, 0, 1, 0, self.radius],[0, 0, 0, 1, 1, self.radius]
        ], dtype=torch.float32)
        return edges

    def generate_faces(self):
        """
        生成圆柱体的面特征
        :return: 面特征张量
        """
        faces = torch.tensor([
            [0, 0, 0, 0, 1, 0, self.radius],[0, 0, 0, 0, 1, 1, self.radius],[0, 0, 0, 1, 0, 0, 1]
        ], dtype=torch.float32)
        return faces

    def generate_edge_to_vertex(self):
        """
        生成边到顶点的关系
        :return: 边到顶点关系张量
        """
        edge_to_vertex = torch.tensor([
            [0, 1],
            [0, 1]
        ],dtype=torch.long)
        return edge_to_vertex

    def generate_face_to_edge(self):
        """
        生成面到边的关系
        :return: 面到边关系张量
        """
        face_to_edge = torch.tensor([
            [0, 1, 2, 2],
            [0, 1, 0, 1]
        ],dtype=torch.long)
        
        return face_to_edge

    def generate_face_to_face(self):

        face_to_face = torch.tensor([
            [0, 1, 2, 2],
            [2, 2, 0, 1]
        ],dtype=torch.long)
        return face_to_face


# In[ ]:


class ConeData:
    def __init__(self, radius, height):
        """
        :param radius: 圆柱体的底面半径
        :param height: 圆柱体的高度
        :param num_segments: 圆周方向的分段数
        """
        self.radius = radius
        self.height = height
        self.vertices = self.generate_vertices()  # 顶点特征
        self.edges = self.generate_edges()  # 边特征
        self.faces = self.generate_faces()  # 面特征
        self.edge_to_vertex = self.generate_edge_to_vertex()  # 边到顶点关系
        self.face_to_edge = self.generate_face_to_edge()  # 面到边关系
        self.face_to_face = self.generate_face_to_face()  # 边到边关系

    def generate_vertices(self):
        """
        生成圆柱体的顶点特征
        :return: 顶点特征的张量
        """
        vertices = torch.tensor([
            [0, 0, 0], [0, 0, self.height]
        ], dtype=torch.float32)
        return vertices

    def generate_edges(self):
        """
        生成圆柱体的边特征
        :return: 边特征张量
        """
        edges = torch.tensor([
            [0, 0, 0, 1, 1, self.radius]
        ], dtype=torch.float32)
        return edges

    def generate_faces(self):
        """
        生成圆柱体的面特征
        :return: 面特征张量
        """
        faces = torch.tensor([
            [0, 0, 0, 0, 1, 1, self.radius],[0, 0, 1, 0, 0, 0, 0]
        ], dtype=torch.float32)
        return faces

    def generate_edge_to_vertex(self):
        """
        生成边到顶点的关系
        :return: 边到顶点关系张量
        """
        edge_to_vertex = torch.tensor([
            [0],
            [1]
        ],dtype=torch.long)
        return edge_to_vertex

    def generate_face_to_edge(self):
        """
        生成面到边的关系
        :return: 面到边关系张量
        """
        face_to_edge = torch.tensor([
            [0, 1],
            [0, 0]
        ],dtype=torch.long)
        
        return face_to_edge

    def generate_face_to_face(self):

        face_to_face = torch.tensor([
            [0, 1],
            [1, 0]
        ],dtype=torch.long)
        return face_to_face


# In[ ]:


class SphereData:
    def __init__(self, radius):
        """
        :param radius: 圆柱体的底面半径
        :param height: 圆柱体的高度
        :param num_segments: 圆周方向的分段数
        """
        self.radius = radius
        self.vertices = self.generate_vertices()  # 顶点特征
        self.edges = self.generate_edges()  # 边特征
        self.faces = self.generate_faces()  # 面特征
        self.edge_to_vertex = self.generate_edge_to_vertex()  # 边到顶点关系
        self.face_to_edge = self.generate_face_to_edge()  # 面到边关系
        self.face_to_face = self.generate_face_to_face()  # 边到边关系

    def generate_vertices(self):
        """
        生成圆柱体的顶点特征
        :return: 顶点特征的张量
        """
        vertices = torch.tensor([
            [0, 0, 0]
        ], dtype=torch.float32)
        return vertices

    def generate_edges(self):
        """
        生成圆柱体的边特征
        :return: 边特征张量
        """
        edges = torch.tensor([
            [0, 0, 0, 0, 0, 0]
        ], dtype=torch.float32)
        return edges

    def generate_faces(self):
        """
        生成圆柱体的面特征
        :return: 面特征张量
        """
        faces = torch.tensor([
            [0, 1, 0, 0, 0, 0, self.radius]
        ], dtype=torch.float32)
        return faces

    def generate_edge_to_vertex(self):
        """
        生成边到顶点的关系
        :return: 边到顶点关系张量
        """
        edge_to_vertex = torch.tensor([
            [0],
            [0]
        ],dtype=torch.long)
        return edge_to_vertex

    def generate_face_to_edge(self):
        """
        生成面到边的关系
        :return: 面到边关系张量
        """
        face_to_edge = torch.tensor([
            [0],
            [0]
        ],dtype=torch.long)
        
        return face_to_edge

    def generate_face_to_face(self):

        face_to_face = torch.tensor([
            [0],
            [0]
        ],dtype=torch.long)
        return face_to_face

