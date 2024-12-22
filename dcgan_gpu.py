import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils  # vutils를 utils로 변경
import matplotlib.pyplot as plt
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset

from PIL import Image  # PIL 라이브러리 추가

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")  # PIL로 이미지 읽기
        if self.transform:
            image = self.transform(image)  # 이미지 변환
        return image


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # 입력은 Z, 컨볼루션으로 들어감 (c x 100 x 1)
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 상태 크기. c x 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 상태 크기. c x 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 상태 크기. c x 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 상태 크기. 64 x 32 x 32
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # 상태 크기. 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 입력은 3 x 64 x 64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. 64 x 32 x 32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. 128 x 16 x 16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. 256 x 8 x 8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # 상태 크기. 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def show_generated_images(images, num_images=64):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("생성된 이미지")
    images = utils.make_grid(images[:num_images], padding=2, normalize=True)
    images = images.detach().cpu().numpy()  # Tensor를 numpy 배열로 변환
    images = np.transpose(images, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.imshow(images)  # 변환된 numpy 배열 사용
    plt.show()

def save_generated_images(images, num_images, epoch):
    # output 디렉토리 생성
    output_dir = './output_300'
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    
    # 이미지를 그리드로 변환하고 numpy 배열로 변환
    grid_images = utils.make_grid(images[:num_images], padding=2, normalize=True)
    
    # Tensor를 numpy 배열로 변환하고 차원 변경
    grid_images = grid_images.detach().cpu().numpy()  # .detach()와 .cpu()를 추가
    grid_images = np.transpose(grid_images, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    # 이미지 저장
    fname = './output_300/epoch' + '_' + str(epoch) + '.jpg'
    plt.imsave(fname, grid_images)  # numpy 배열을 사용
    plt.close()

# 생성기와 판별기 생성
netG = Generator()
netD = Discriminator()

# 모든 가중치 초기화
netG.apply(weights_init)
netD.apply(weights_init)

# 데이터 준비
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# 사용자 정의 데이터셋 클래스 사용
dataset = CustomDataset(root_dir='./dataset/image_filter', transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 학습
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
netG.to(device)
netD.to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.00002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.00002, betas=(0.5, 0.999))

num_epochs = 300  # 데모 목적으로; 더 나은 결과를 위해 증가시킬 수 있음

fixed_noise = torch.randn(64, 100, 1, 1, device=device)  # 중간 시각화
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # 판별기 업데이트: log(D(x)) + log(1 - D(G(z))) 최대화
        netD.zero_grad()
        real_data = data.to(device)
        batch_size = real_data.size(0)
        real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
        fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

        # D에 진짜 배치를 통과시킴
        output = netD(real_data).view(-1)
        errD_real = criterion(output, real_label)
        errD_real.backward()

        # G로 가짜 이미지 배치 생성
        noise = torch.randn(batch_size, 100, 1, 1, device=device)
        fake_data = netG(noise)
        output = netD(fake_data.detach()).view(-1)

        # 가짜 배치에서 D의 손실 계산
        errD_fake = criterion(output, fake_label)
        errD_fake.backward()

        # 진짜와 가짜 배치의 기울기 더하기
        errD = errD_real + errD_fake
        optimizerD.step()

        # 생성기 업데이트: log(D(G(z))) 최대화
        netG.zero_grad()
        output = netD(fake_data).view(-1)
        errG = criterion(output, real_label)
        errG.backward()
        optimizerG.step()

        # 훈련 통계 출력
        if i % 1 == 0:  # 배치 인덱스에 대해 모든 배치 출력
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item()))

    # 에포크가 20으로 나누어 떨어질 때마다 이미지 저장
    if epoch == 0 or epoch % 20 == 0:
        fake_images = netG(fixed_noise)
        save_generated_images(fake_images, 64, epoch=epoch)
# 시각화 함수
fake_images = netG(fixed_noise)
show_generated_images(fake_images)