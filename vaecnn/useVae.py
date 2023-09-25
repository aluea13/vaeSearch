import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
from PIL import Image
# 超参数设置
image_size = 784   #图片大小
h_dim = 400
z_dim = 20
num_epochs = 15   #15个循环
batch_size = 1   #一批的数量
learning_rate = 0.001   #学习率
log_interval = 10
latent_code_num = 32
# 配置GPU或CPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
# VAE 模型
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
                
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
                    
            nn.Conv2d(128, 128, kernel_size=3 ,stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),                  
            )
             
        self.fc11 = nn.Linear(128 * 7 * 7, latent_code_num)
        self.fc12 = nn.Linear(128 * 7 * 7, latent_code_num)
        self.fc2 = nn.Linear(latent_code_num, 128 * 7 * 7)
            
        self.decoder = nn.Sequential(                
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
                    
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
            )


    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var /2)
        eps = torch.randn_like(std)
        return mu+eps * std


    def forward(self, x):
        out1, out2 = self.encoder(x), self.encoder(x)
        mu = self.fc11(out1.view(out1.size(0),-1))
        log_var = self.fc12(out2.view(out2.size(0),-1))
        z = self.reparameterize(mu, log_var)
        out3 = self.fc2(z).view(z.size(0), 128, 7, 7)
        x_reconst = self.decoder(out3)
        return x_reconst, mu, log_var


print(thedata.shape)
for i in range(0,28):
    for j in range(0,28):
        if i<=j :
            thedata[0][0][i][j]=0
print("-----")
thedata=thedata.clone().numpy()
thedata=thedata.reshape(28,28)
im = Image.fromarray(np.uint8(thedata*255))
im.show()
'''
model=torch.load('myVae.pt')
thedata=thedata.to(device)
out1, out2 = model.encoder(thedata), model.encoder(thedata)
mu = model.fc11(out1.view(out1.size(0),-1))
log_var = model.fc12(out2.view(out2.size(0),-1))
z = model.reparameterize(mu, log_var)
#z = torch.randn(1, 32)
nz=z.clone().detach().cpu().numpy()
print(nz.shape)
print(nz)

#out3 = model.fc2(z).view(z.size(0), 128, 7, 7)
#out = model.decoder(out3).view(-1, 1, 28, 28)
'''