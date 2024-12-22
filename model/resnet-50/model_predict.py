import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import random
import numpy as np

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DCGAN Generator 모델 정의 =====
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

# ===== DCGAN Discriminator 모델 정의 =====
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

# DCGAN Generator 불러오기
latent_dim = 100
generator = Generator().to(device)
gen_checkpoint = torch.load(r"C:\Code\Python\AIPhoto\dataset\output\11.25\generator_epoch_300.pth")
generator_state_dict = gen_checkpoint['model_state_dict']
new_gen_state_dict = {}

for key in generator_state_dict.keys():
    new_key = key.replace('main.', 'gen.')
    new_gen_state_dict[new_key] = generator_state_dict[key]

generator.load_state_dict(new_gen_state_dict, strict=False)
generator.eval()

# DCGAN Discriminator 불러오기
discriminator = Discriminator().to(device)
dis_checkpoint = torch.load(r"C:\Code\Python\AIPhoto\dataset\output\11.25\discriminator_epoch_300.pth")
discriminator_state_dict = dis_checkpoint['model_state_dict']
new_dis_state_dict = {}

for key in discriminator_state_dict.keys():
    new_key = key.replace('main.', 'gen.')
    new_gen_state_dict[new_key] = discriminator_state_dict[key]

discriminator.load_state_dict(new_dis_state_dict, strict=False)
discriminator.eval()

# ===== ResNet-50 모델 정의 (사전 학습된 모델 사용) =====
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# ===== DCGAN으로 이미지 생성 =====
def generate_image(generator, latent_dim):
    latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
    generated_image = generator(latent_vector).detach().cpu()
    return generated_image

# ===== 지표를 사용해 DCGAN 품질 평가 =====
def evaluate_generated_images(generator, discriminator, latent_dim, num_samples=100):
    generator.eval()
    discriminator.eval()

    realness_scores = []
    for _ in range(num_samples):
        latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
        generated_img = generator(latent_vector)
        score = discriminator(generated_img).item()  # 진짜 데이터와의 유사도 (0~1)
        realness_scores.append(score)

    avg_score = sum(realness_scores) / len(realness_scores)
    return avg_score

# ===== Discriminator로 가짜 이미지 판별 =====
def predict_real_or_fake(discriminator, generated_image):
    input_tensor = generated_image.to(device)  # 생성된 이미지 (1, 3, 64, 64)
    
    with torch.no_grad():
        prediction = discriminator(input_tensor)
        fake_or_real = prediction.item()  # 0 ~ 1 (가짜: 0, 진짜: 1)
        return fake_or_real

# ===== ResNet-50으로 합격 예측 =====
def predict_success(resnet_model, generated_image=None, image_path=None):
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if image_path:
        image = Image.open(image_path).convert("RGB")
    elif generated_image is not None:  # Tensor가 None이 아닌지 확인
        image = transforms.ToPILImage()(generated_image.squeeze(0))
    else:
        raise ValueError("image_path 또는 generated_image 중 하나를 제공해야 합니다.")

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = resnet_model(input_tensor)
        success_prob = torch.sigmoid(prediction[0][1]).item()  
        return success_prob


# ===== 실행 =====
generated_img = generate_image(generator, latent_dim)

# 생성된 이미지에 대해 Discriminator로 가짜 판별
fake_score = predict_real_or_fake(discriminator, generated_img)
print(f"생성된 이미지가 가짜일 확률: {fake_score * 100:.2f}%")

# ResNet-50으로 합격 예측
success_rate_generated = predict_success(resnet_model, generated_image=generated_img)
print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# 생성 지표를 활용한 DCGAN 품질 평가
avg_realness = evaluate_generated_images(generator, discriminator, latent_dim)
print(f"DCGAN 생성 이미지의 평균 진짜 유사도 점수: {avg_realness * 100:.2f}%")

# 실제 이미지로 예측 (옵션)
real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강수지.png"
success_rate_real = predict_success(resnet_model, image_path=real_image_path)
print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")
