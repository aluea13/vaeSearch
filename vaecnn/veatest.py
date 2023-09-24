from cmath import log
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
 
# 配置GPU或CPU设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
# 创建目录保存生成的图片
sample_dir = 'samples'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
 
# 超参数设置
image_size = 784   #图片大小
h_dim = 400
z_dim = 20
num_epochs = 15   #15个循环
batch_size = 64   #一批的数量
learning_rate = 0.001   #学习率
log_interval = 10
latent_code_num = 32
 
# 获取数据集
dataset = torchvision.datasets.MNIST(root='./data',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)
 
# 数据加载，按照batch_size大小加载，并随机打乱
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

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

model = VAE().to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for epoch in range (num_epochs):
    for i, (x, _) in enumerate(data_loader):
        x = x.to(device)
        x_reconst, mu, log_var = model(x)

        reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
        kl_div = -0.5 * torch.sum(1+ log_var -mu.pow(2) - log_var.exp())

        loss = reconst_loss + kl_div

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if(i+1) % 50 == 0:
            print("Epoch[{}/{}],Step[{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}"
            .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))

    with torch.no_grad():
        z = torch.randn(batch_size, latent_code_num).to(device)

        out3 = model.fc2(z).view(z.size(0), 128, 7, 7)
        out = model.decoder(out3).view(-1, 1, 28, 28)

        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        out, _, _ = model(x)

        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch + 1)))

torch.save(model,"myVae.pt")