import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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

model=torch.load('myVae.pt')
z = [[ 0.00360918 , 2.1439033 ,  3.0785916 ,  0.37259835 ,-1.4487599 ,  0.44860405
 , -0.93411684 , 0.4581674  ,-0.63027006 , 0.1900351 ,  0.39372241 , 1.7865871
  , 1.2728329 , -0.4823635  , 2.5061452 , -0.8993141,  -2.0712206 , -1.6678908
  , 2.4588966  , 1.2420831 ,  0.632474  , -3.400418  ,  1.4520209  , 1.1479979
 , -1.7005702 ,  2.1224966 , -0.48628387 ,-1.6945927 , -0.6031383 ,  0.37891412
 ,  2.407934  ,  0.57290787]]

z = np.array(z).reshape(1,32)
z = torch.from_numpy(z).to(torch.float32).to(device)

out3 = model.fc2(z).view(z.size(0), 128, 7, 7)
out = model.decoder(out3).view(-1, 1, 28, 28)
nout=out.clone().detach().cpu().numpy()
nout = nout.reshape(28,28)

im = Image.fromarray(np.uint8(nout*255))
im.show()
