# **Overview**

This repository mainly focuses on proving the feasibility of the self-supervised hierarchical encoder-decoder network which means we mainly focus on the first and second part of the figure below:
- *Data preprocessing*
- *Encoder-decoder L2 loss minimization*
![image](https://github.com/user-attachments/assets/528f34d8-9f0e-4bac-86e6-a8b925c73020)
*Figure 1. Overview of ssl for CAD models*

*(The source paper:https://openaccess.thecvf.com/content/CVPR2023/html/Jones_Self-Supervised_Representation_Learning_for_CAD_CVPR_2023_paper.html)*

# **Data preprocessing**

For the input data of the encoder we need to extract BRep data and after that **transform the BRep data into the form which can suit the encoder structure** well to generate embeddings

## **1. BRep data extraction from Fusion 360**

The Brep_data extraction part is modified by BrepNet.

![image](https://github.com/user-attachments/assets/31ed605b-fd09-4953-ad74-292974f4b308)
*Figure 2. Pipeline of BRep extraction*

*(The source: https://github.com/AutodeskAILab/BRepNet/blob/master/docs/building_your_own_dataset.md)*

## **2. Alternative method: data generator**

For the feasibility test we don't need large dataset to test the algorithms of the encoder-decoder structure.

We move to another strategy to create a python-based model generator and we can **extract the BRep data within the generator** without additional operations.

The weakness is the complexity of the generated model is not high which may influence the performance of the pre-training result.


# **SSL validation experiment**
We get the simplified typical datasets from the data generator and use the BRep data as input for the encoder-decoder to **minimize the L2 loss**.

Here are the result based on self-prepared datasets:
![image](https://github.com/user-attachments/assets/ca29a498-d4e0-4492-8b57-ad4b48e380c0)

*Figure 2. Training results over 4 datasets*

# **Future work**
- Larger dataset preparation and data preprocessing necessary.
- Computation nodes(A100/H100 * 4 server) for the large-scale training.
- Few shot learning by pretraining + downstream GCN for modelling segmentation/machining segmentation/part classification



# **Quickstart instructions in Juypter notebook**

### **Setting up the environment**

We don not need GPU for this repository. Please make sure the correct environment configuration according to the file **environment.yml**.

### **Dataset preparation**

You can prepare your own BRep dataset for the encoder by the file **data_generater.py**(modification for the local data).

After you get the output embeddings from the encoder, run the file **sampled_data.py** and get the input of the decoder for the rasterization.

You can find the files in the Example_data folder.

### **Pre-training of the encoder-decoder structure**

Prepare the two flies named **encoder.py** and **decoder.py** (network folder) locally and run the code. 


Here is a simply example :

Finish the environment configuration and prepare all the files as the quickstart instructions.

### 1.data_generator

run the code below:

```python

encoder = CustomBRepEncoder(
    v_in_width=3, e_in_width=6, f_in_width=7,
    out_width=64, num_layers=2, use_attention=True
)
data = CuboidData(2,3,4)
print(encoder(data))
```
```python
encoder = CustomBRepEncoder(
    v_in_width=3, e_in_width=6, f_in_width=7,
    out_width=64, num_layers=2, use_attention=True
)
data2 = CylinderData(3, 6)
print(encoder(data2))

```
```python
encoder = CustomBRepEncoder(
    v_in_width=3, e_in_width=6, f_in_width=7,
    out_width=64, num_layers=2, use_attention=True
)
data3 = ConeData(2, 5)
print(encoder(data3))
```
```python
encoder = CustomBRepEncoder(
    v_in_width=3, e_in_width=6, f_in_width=7,
    out_width=64, num_layers=2, use_attention=True
)
data4 = SphereData(4)
print(encoder(data4))
```
### 2.sampled_data and L2 loss

```python

preprocessed_data = preprocess_data(encoder, data)

decoder = ConditionalDecoder(latent_size=64, hidden_dims=[1024, 1024, 1024, 1024])

optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

preprocessed_data = torch.load("preprocessed_data.pt")

n_epochs = 100

losses = []

for epoch in range(n_epochs):
    total_loss = 0.0
    for embedding, sampled_points, sampled_sdf in preprocessed_data:
        
        # Step 1: Decoder forward pass
        decoder_input = torch.cat([embedding.unsqueeze(0).repeat(sampled_points.shape[0], 1), sampled_points], dim=-1)
        predicted = decoder(sampled_points, embedding)

        # Step 2: Compute ground-truth and loss
        target_xyz = compute_xyz_from_uv(sampled_points).float()  # Compute ground-truth (x, y, z)
        loss = compute_loss(predicted, target_xyz, sampled_sdf)
        total_loss += loss.item()

        # Step 3: Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # retain_graph=False (默认行为)
        optimizer.step()

    average_loss = total_loss / len(preprocessed_data)
    losses.append(average_loss)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}")
```

```python

preprocessed_data_cylinder = preprocess_data_cylinder(encoder, data2)

decoder = ConditionalDecoder(latent_size=64, hidden_dims=[1024, 1024, 1024, 1024])

optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

preprocessed_data_cylinder = torch.load("preprocessed_data_cylinder.pt")

n_epochs = 100

losses = []

for epoch in range(n_epochs):
    total_loss = 0.0
    for embedding, sampled_points, sampled_sdf in preprocessed_data_cylinder:
        
        # Step 1: Decoder forward pass
        decoder_input = torch.cat([embedding.unsqueeze(0).repeat(sampled_points.shape[0], 1), sampled_points], dim=-1)
        predicted = decoder(sampled_points, embedding)

        # Step 2: Compute ground-truth and loss
        target_xyz = compute_xyz_from_uv(sampled_points).float()  # Compute ground-truth (x, y, z)
        loss = compute_loss(predicted, target_xyz, sampled_sdf)
        total_loss += loss.item()

        # Step 3: Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # retain_graph=False (默认行为)
        optimizer.step()

    average_loss = total_loss / len(preprocessed_data_cylinder)
    losses.append(average_loss)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}")
```

```python
preprocessed_data_cone = preprocess_data_cone(encoder, data3)

decoder = ConditionalDecoder(latent_size=64, hidden_dims=[1024, 1024, 1024, 1024])

optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

preprocessed_data_cone = torch.load("preprocessed_data_cone.pt")
print("Loaded preprocessed data.")

# 训练代码
n_epochs = 100

losses = []

for epoch in range(n_epochs):
    total_loss = 0.0
    for embedding, sampled_points, sampled_sdf in preprocessed_data_cone:
        
        # Step 1: Decoder forward pass
        decoder_input = torch.cat([embedding.unsqueeze(0).repeat(sampled_points.shape[0], 1), sampled_points], dim=-1)
        predicted = decoder(sampled_points, embedding)

        # Step 2: Compute ground-truth and loss
        target_xyz = compute_xyz_from_uv(sampled_points).float()  # Compute ground-truth (x, y, z)
        loss = compute_loss(predicted, target_xyz, sampled_sdf)
        total_loss += loss.item()

        # Step 3: Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # retain_graph=False (默认行为)
        optimizer.step()

    average_loss = total_loss / len(preprocessed_data_cone)
    losses.append(average_loss)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}")
```

```python
preprocessed_data_sphere = preprocess_data_sphere(encoder, data4)

decoder = ConditionalDecoder(latent_size=64, hidden_dims=[1024, 1024, 1024, 1024])

optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

preprocessed_data_sphere = torch.load("preprocessed_data_sphere.pt")
print("Loaded preprocessed data.")

# 训练代码
n_epochs = 100

losses = []

for epoch in range(n_epochs):
    total_loss = 0.0
    for embedding, sampled_points, sampled_sdf in preprocessed_data_sphere:
        
        # Step 1: Decoder forward pass
        decoder_input = torch.cat([embedding.unsqueeze(0).repeat(sampled_points.shape[0], 1), sampled_points], dim=-1)
        predicted = decoder(sampled_points, embedding)

        # Step 2: Compute ground-truth and loss
        target_xyz = compute_xyz_from_uv(sampled_points).float()  # Compute ground-truth (x, y, z)
        loss = compute_loss(predicted, target_xyz, sampled_sdf)
        total_loss += loss.item()

        # Step 3: Backward and optimize
        optimizer.zero_grad()
        loss.backward()  # retain_graph=False (默认行为)
        optimizer.step()

    average_loss = total_loss / len(preprocessed_data_sphere)
    losses.append(average_loss)
    print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {average_loss}")
```


