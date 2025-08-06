import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from dynamic_network_architectures.visual.vnet import VNet  # 请改为你保存模型的路径，如 from my_model.vnet import VNet

# -------------------------------
# 1. 加载Nifti图像（MSD Task01）
# -------------------------------
nii_path = r"E:\github\datasets\Task01_BrainTumour\imagesTr/BRATS_001.nii.gz"  # 替换为你的路径
nii_img = nib.load(nii_path)
image = nii_img.get_fdata()  # shape: [H, W, D, C]（MSD 是最后一维为通道）

# 通道维转到前面 (C, D, H, W)，例如 T1, T1ce, T2, FLAIR 四个模态
image = np.transpose(image, (3, 2, 0, 1))  # [C, D, H, W]

# 选取一个模态，例如 FLAIR 通道（通常为第4个，即 index=3）
flair = image[3]  # shape: [D, H, W]

# 添加 batch 和 channel 维度，变为 [1, 1, D, H, W]
input_tensor = torch.from_numpy(flair).unsqueeze(0).unsqueeze(0).float()

# 简单归一化到 [0, 1]
input_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min() + 1e-5)

# -------------------------------
# 2. 初始化模型 & 加载权重
# -------------------------------
model = VNet(input_channels=4, num_classes=3, spatial_dims=2)  # 改成你的输出通道数
checkpoint_path = r"E:\github\nnUNetv2\dynamic-network-architectures\dynamic_network_architectures\visual\checkpoint_best.pth"  # 替换为你的权重路径
model.load_state_dict(torch.load(checkpoint_path, map_location='cuda'))
model.eval()

# -------------------------------
# 3. 注册钩子获取中间层特征图
# -------------------------------
activation_maps = {}

def get_activation(name):
    def hook(model, input, output):
        activation_maps[name] = output.detach()
    return hook

# 例如注册 down_tr64 的第一层
model.down_tr64.ops[0].conv_block.register_forward_hook(get_activation("down_tr64"))

# -------------------------------
# 4. 推理 + 热力图可视化
# -------------------------------
with torch.no_grad():
    _ = model(input_tensor)

# 获取特征图 [B, C, D, H, W]
features = activation_maps["down_tr64"]

# 选择通道和切片（可调）
channel = 0
depth_slice = features.shape[2] // 2
feature_slice = features[0, channel, depth_slice, :, :].cpu().numpy()

# 归一化
feature_slice -= feature_slice.min()
feature_slice /= feature_slice.max() + 1e-5

# 可视化
plt.imshow(feature_slice, cmap='hot')
plt.title(f'Heatmap from down_tr64, channel {channel}, slice {depth_slice}')
plt.colorbar()
plt.show()
