import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.image as mpimg
from PIL import Image

# 获取数据集
dataset = torchvision.datasets.MNIST(root='./data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=False)
 
# 数据加载，按照batch_size大小加载，并随机打乱
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=1,
                                          shuffle=False)
thedata=0
for i, (x, _) in enumerate(data_loader):
    thedata=x
    if i>=0 :
        break

thedata=thedata.numpy()

thedata=thedata.reshape(28,28)

im = Image.fromarray(np.uint8(thedata*255))
im.show()
