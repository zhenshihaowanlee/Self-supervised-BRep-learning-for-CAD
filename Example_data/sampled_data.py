#!/usr/bin/env python
# coding: utf-8

# In[ ]:
def sample_uv_extended(resolution, extend=0.1):
    u = torch.linspace(-extend, 1 + extend, resolution)
    v = torch.linspace(-extend, 1 + extend, resolution)
    uu, vv = torch.meshgrid(u, v, indexing='ij')
    return torch.stack([uu.flatten(), vv.flatten()], dim=-1)

from scipy.spatial import KDTree

def compute_sdf(inside_points, outside_points):
    inside_tree = KDTree(outside_points)
    outside_tree = KDTree(inside_points)

    d_inside, _ = inside_tree.query(inside_points)
    d_outside, _ = outside_tree.query(outside_points)

    sdf_inside = d_inside
    sdf_outside = -d_outside

    sdf_points = torch.cat([inside_points, outside_points], dim=0)
    sdf_values = torch.cat([torch.tensor(sdf_inside), torch.tensor(sdf_outside)], dim=0)
    return sdf_points, sdf_values

def bias_sample_sdf(sdf_points, sdf_values, n_samples, boundary_ratio=0.4):
    sorted_indices = torch.argsort(torch.abs(sdf_values))
    n_boundary = int(n_samples * boundary_ratio)

    boundary_indices = sorted_indices[:n_boundary]
    random_indices = sorted_indices[n_boundary:]
    random_indices = random_indices[torch.randperm(len(random_indices))[:n_samples - n_boundary]]

    selected_indices = torch.cat([boundary_indices, random_indices], dim=0)
    return sdf_points[selected_indices], sdf_values[selected_indices]

def compute_loss(predicted, target_xyz, target_sdf):
    pred_xyz, pred_sdf = predicted[:, :3], predicted[:, 3]
    xyz_loss = torch.nn.functional.mse_loss(pred_xyz, target_xyz)
    sdf_loss = torch.nn.functional.mse_loss(pred_sdf, target_sdf)
    return xyz_loss + sdf_loss

def compute_xyz_from_uv(uv_coords):
    """
    将 UV 参数坐标映射到简单的 3D 平面坐标 (x, y, z)。

    参数:
        uv_coords (torch.Tensor): UV 坐标，形状为 (n_samples, 2)。

    返回:
        torch.Tensor: 对应的 3D 坐标，形状为 (n_samples, 3)。
    """
    # 假设曲面是一个简单的平面，并使用 (u, v) 作为 (x, y)，z = 0
    x = uv_coords[:, 0]  # u 对应 x
    y = uv_coords[:, 1]  # v 对应 y
    z = torch.zeros_like(x)  # z 固定为 0

    return torch.stack([x, y, z], dim=-1)


from OCC.Core.gp import gp_Pnt2d, gp_Dir2d, gp_Ax2d
from OCC.Core.Geom2d import Geom2d_Line
from OCC.Core.Geom2dAPI import Geom2dAPI_ProjectPointOnCurve

def query_cad_kernel(uv_samples):
    """
    使用 OpenCASCADE 判断 UV 点是否在裁剪区域内部或外部
    这里假设裁剪区域是 2D 直线的一侧
    """
    # 定义直线作为裁剪区域
    origin = gp_Pnt2d(0.5, 0.5)  # 直线的起点
    direction = gp_Dir2d(1, 0)  # 直线的方向
    axis = gp_Ax2d(origin, direction)
    curve = Geom2d_Line(axis)  # 创建直线

    # 判断点是否在直线的“左侧”或“右侧”
    inside_mask = []
    for uv in uv_samples:
        point = gp_Pnt2d(uv[0].item(), uv[1].item())
        proj = Geom2dAPI_ProjectPointOnCurve(point, curve)  # 投影点到曲线
        dist = proj.LowerDistance()  # 获取点到曲线的最近距离
        inside = dist < 0.4  # 假设 0.4 为“内部”的阈值
        inside_mask.append(inside)

    inside_mask = torch.tensor(inside_mask)
    inside_points = uv_samples[inside_mask]
    outside_points = uv_samples[~inside_mask]
    return inside_points, outside_points



# 数据预处理函数
def preprocess_data(encoder, data, num_samples=5000, n_samples=500):
    embeddings = encoder(data)  # 从 encoder 获取嵌入
    preprocessed_data = []

    for embedding in embeddings:
        uv_samples = sample_uv_extended(num_samples).float()
        inside_points, outside_points = query_cad_kernel(uv_samples)  # CAD kernel
        sdf_points, sdf_values = compute_sdf(inside_points, outside_points)  # SDF计算
        sampled_points, sampled_sdf = bias_sample_sdf(sdf_points, sdf_values, n_samples=n_samples)

        # 转换为 float 并保存
        sampled_points = sampled_points.float()
        sampled_sdf = sampled_sdf.float()

        preprocessed_data.append((embedding, sampled_points, sampled_sdf))

    # 保存为文件
    torch.save(preprocessed_data, "preprocessed_data.pt")
    print("Preprocessed data saved to preprocessed_data.pt")
    return preprocessed_data



# In[ ]:


def preprocess_data_cylinder(encoder, data, num_samples=5000, n_samples=500):
    embeddings = encoder(data2)  # 从 encoder 获取嵌入
    preprocessed_data_cylinder = []

    for embedding in embeddings:
        uv_samples = sample_uv_extended(num_samples).float()
        inside_points, outside_points = query_cad_kernel(uv_samples)  # CAD kernel
        sdf_points, sdf_values = compute_sdf(inside_points, outside_points)  # SDF计算
        sampled_points, sampled_sdf = bias_sample_sdf(sdf_points, sdf_values, n_samples=n_samples)

        # 转换为 float 并保存
        sampled_points = sampled_points.float()
        sampled_sdf = sampled_sdf.float()

        preprocessed_data_cylinder.append((embedding, sampled_points, sampled_sdf))

    # 保存为文件
    torch.save(preprocessed_data_cylinder, "preprocessed_data_cylinder.pt")
    print("Preprocessed data saved to preprocessed_data_cylinder.pt")
    return preprocessed_data_cylinder


# In[ ]:


def preprocess_data_cone(encoder, data, num_samples=5000, n_samples=500):
    embeddings = encoder(data3)  # 从 encoder 获取嵌入
    preprocessed_data_cone = []

    for embedding in embeddings:
        uv_samples = sample_uv_extended(num_samples).float()
        inside_points, outside_points = query_cad_kernel(uv_samples)  # CAD kernel
        sdf_points, sdf_values = compute_sdf(inside_points, outside_points)  # SDF计算
        sampled_points, sampled_sdf = bias_sample_sdf(sdf_points, sdf_values, n_samples=n_samples)

        # 转换为 float 并保存
        sampled_points = sampled_points.float()
        sampled_sdf = sampled_sdf.float()

        preprocessed_data_cone.append((embedding, sampled_points, sampled_sdf))

    # 保存为文件
    torch.save(preprocessed_data_cone, "preprocessed_data_cone.pt")
    print("Preprocessed data saved to preprocessed_data_cone.pt")
    return preprocessed_data_cone


# In[ ]:


def preprocess_data_sphere(encoder, data, num_samples=5000, n_samples=500):
    embeddings = encoder(data4)  # 从 encoder 获取嵌入
    preprocessed_data_sphere = []

    for embedding in embeddings:
        uv_samples = sample_uv_extended(num_samples).float()
        inside_points, outside_points = query_cad_kernel(uv_samples)  # CAD kernel
        sdf_points, sdf_values = compute_sdf(inside_points, outside_points)  # SDF计算
        sampled_points, sampled_sdf = bias_sample_sdf(sdf_points, sdf_values, n_samples=n_samples)

        # 转换为 float 并保存
        sampled_points = sampled_points.float()
        sampled_sdf = sampled_sdf.float()

        preprocessed_data_sphere.append((embedding, sampled_points, sampled_sdf))

    # 保存为文件
    torch.save(preprocessed_data_sphere, "preprocessed_data_sphere.pt")
    print("Preprocessed data saved to preprocessed_data_sphere.pt")
    return preprocessed_data_sphere

