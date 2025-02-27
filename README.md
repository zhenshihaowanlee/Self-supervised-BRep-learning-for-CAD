# **Overview**

This repository mainly focuses on proving the feasibility of the self-supervised hierarchical encoder-decoder network which means we mainly focus on the first and second part of the figure below:
- *Data preprocessing*
- *Encoder-decoder L2 loss minimization*
![image](https://github.com/user-attachments/assets/528f34d8-9f0e-4bac-86e6-a8b925c73020)
*Figure 1. Overview of ssl for CAD models*

*(The source paper:https://openaccess.thecvf.com/content/CVPR2023/html/Jones_Self-Supervised_Representation_Learning_for_CAD_CVPR_2023_paper.html)*

# **Data preprocessing**

For the input data of the encoder we need to extract BRep data and after that **transform the BRep data into the form which can suit the encoder structure** well to generate embeddings

## **1.BRep data extraction from Fusion 360**

The Brep_data extraction part is modified by BrepNet.

![image](https://github.com/user-attachments/assets/31ed605b-fd09-4953-ad74-292974f4b308)
*Figure 2. Pipeline of BRep extraction*

*(The source: https://github.com/AutodeskAILab/BRepNet/blob/master/docs/building_your_own_dataset.md)*


3.For the further application we still need to combine it with downstream task GCN(segmentation/classification)
