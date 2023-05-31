import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import detr_resnet50
from torchvision.datasets import CocoDetection
from torchvision import transforms

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor(),
])

# 加载数据集
# 请替换为你自己的数据集
dataset = CocoDetection(root='path_to_your_dataset', transform=transform)

# 创建数据加载器
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 加载预训练的 DETR 模型
model = detr_resnet50(pretrained=True)

# 将模型设为训练模式
model.train()

# 选择优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)

# 训练
for epoch in range(10):  # 你可以选择需要的 epoch 数
    for images, targets in data_loader:
        # 将数据移动到 GPU
        images = list(image.to('cuda') for image in images)
        targets = [{k: v.to('cuda') for k, v in t.items()} for t in targets]

        # 计算损失
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # 反向传播和优化
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
