#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

