import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils  # vutils를 utils로 변경
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image  # PIL 라이브러리 추가

import torch
print(torch.__version__)
print(torch.cuda.is_available())

# 한글 글꼴 경로 설정 (시스템에 맞게 수정 필요)
font_path = "C:\\Windows\\Fonts\\malgun.ttf"  # Windows의 Malgun Gothic
font_prop = fm.FontProperties(fname=font_path, size=12)

plt.rc('font', family=font_prop.get_name())  # Matplotlib에 글꼴 설정

# 손실 기록을 위한 리스트
D_losses = []
G_losses = []

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"총 이미지 개수: {len(self.image_files)}개")

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

# 가중치 초기화 코드
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        

# 이미지 복원 함수 (denormalize)
def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    # 텐서를 GPU로 이동
    device = tensor.device  # 텐서가 위치한 장치(device)를 가져옵니다.
    
    # mean과 std를 tensor의 shape에 맞게 확장
    mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)  # (1, 3, 1, 1)
    std = torch.tensor(std, device=device).view(1, 3, 1, 1)
    
    # 이미지 복원
    tensor = tensor * std + mean
    return tensor


def show_generated_images(images, num_images=64):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("생성된 이미지")
    
    # 이미지 복원
    images = denormalize(images)
    
    images = utils.make_grid(images[:num_images], padding=2, normalize=True)
    images = images.detach().cpu().numpy()  # Tensor를 numpy 배열로 변환
    images = np.transpose(images, (1, 2, 0))  # (C, H, W) -> (H, W, C)
    plt.imshow(images)  # 변환된 numpy 배열 사용
    plt.show()


def save_generated_images(images, num_images, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    
    # 이미지 복원
    images = denormalize(images)
    
    # 이미지를 그리드로 변환하고 numpy 배열로 변환
    grid_images = utils.make_grid(images[:num_images], padding=2, normalize=True)
    
    # Tensor를 numpy 배열로 변환하고 차원 변경
    grid_images = grid_images.detach().cpu().numpy()  # .detach()와 .cpu()를 추가
    grid_images = np.transpose(grid_images, (1, 2, 0))  # (C, H, W) -> (H, W, C)

    # 이미지 저장
    fname = os.path.join(output_dir, f'epoch_{epoch}_images.jpg')
    plt.imsave(fname, grid_images)  # numpy 배열을 사용
    plt.close()
    
# 모델 저장 함수
def save_model(netG, netD, optimizerG, optimizerD, epoch, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': netG.state_dict(),
        'optimizer_state_dict': optimizerG.state_dict()
    }, os.path.join(output_dir, f'generator_epoch_{epoch}.pth'))
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': netD.state_dict(),
        'optimizer_state_dict': optimizerD.state_dict()
    }, os.path.join(output_dir, f'discriminator_epoch_{epoch}.pth'))
    
    print(f"모델 저장됨: Generator와 Discriminator의 가중치 및 옵티마이저 상태가 epoch {epoch}에 저장되었습니다.")

# 손실 그래프 저장 함수
def save_loss_plot(D_losses, G_losses, output_dir, epoch=None):
    plt.figure(figsize=(10, 5))
    plt.plot(D_losses, label='Discriminator Loss')
    plt.plot(G_losses, label='Generator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Discriminator and Generator Loss During Training')
    plt.legend()
    if epoch is not None:
        plt.savefig(os.path.join(output_dir, f'loss_plot_epoch_{epoch}.png'))
    else:
        plt.savefig(os.path.join(output_dir, 'loss_plot_final.png'))
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
dataset = CustomDataset(root_dir='./dataset/deleted_dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# 데이터 값 범위 확인
sample_image = next(iter(dataloader))
print(f"최초 이미지의 최소값: {sample_image.min()}")
print(f"최초 이미지의 최대값: {sample_image.max()}")

# 학습
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"현재 사용 중인 장치: {device}")
netG.to(device)
netD.to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.00005, betas=(0.5, 0.999))   # 기존 0.0002 학습률
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))   # 기존 0.0002 학습률

num_epochs = 400
fixed_noise = torch.randn(64, 100, 1, 1, device=device)  # 중간 시각화
output_dir = 'C:\Code\Python\AIPhoto\dataset\output\dcgan_output(11.26)'

for epoch in range(num_epochs):   
    for i, data in enumerate(dataloader, 0):
        # 판별기 업데이트: log(D(x)) + log(1 - D(G(z))) 최대화
        netD.zero_grad()
        real_data = data.to(device)
        batch_size = real_data.size(0)
        real_label = torch.full((batch_size,), 0.9, dtype=torch.float, device=device)
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
        
        D_losses.append(errD.item())  # 판별기 손실 기록
        G_losses.append(errG.item())  # 생성기 손실 기록

        # 훈련 통계 출력
        if i % 50 == 0:  # 배치 인덱스에 대해 모든 배치 출력
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader), errD.item(), errG.item()))

    # 에포크가 20으로 나누어 떨어질 때마다 이미지 저장
    if epoch == 0 or epoch % 20 == 0:
        fake_images = netG(fixed_noise)
        save_generated_images(fake_images, 64, epoch, output_dir)
    
    # 에포크 50 증가마다 모델 저장
    if epoch > 0 and epoch % 50 == 0:
        save_model(netG, netD, optimizerG, optimizerD, epoch, output_dir)
        # 손실 그래프를 에포크마다 저장
        save_loss_plot(D_losses, G_losses, output_dir, epoch)
        
# 최종 저장
fake_images = netG(fixed_noise)
save_generated_images(fake_images, 64, num_epochs, output_dir)

save_model(netG, netD, optimizerG, optimizerD, num_epochs, output_dir)
show_generated_images(fake_images)

# 최종 손실 그래프 저장
save_loss_plot(D_losses, G_losses, output_dir, epoch)