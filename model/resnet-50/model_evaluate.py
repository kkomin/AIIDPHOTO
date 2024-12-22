# import os
# import torch
# import torch.nn as nn
# import torchvision.transforms as transforms
# from PIL import Image
# from torchvision import models
# from torchvision.models import resnet50, ResNet50_Weights

# # 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ===== DCGAN Generator 모델 정의 =====
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # 입력은 Z, 컨볼루션으로 들어감 (c x 100 x 1)
#             nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(True),
#             # 상태 크기. c x 512 x 4 x 4
#             nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(True),
#             # 상태 크기. c x 256 x 8 x 8
#             nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(True),
#             # 상태 크기. c x 128 x 16 x 16
#             nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(True),
#             # 상태 크기. 64 x 32 x 32
#             nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # 상태 크기. 3 x 64 x 64
#         )

#     def forward(self, z):
#         return self.main(z)
    
# # Discriminator
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.main = nn.Sequential(
#             # 입력은 3 x 64 x 64
#             nn.Conv2d(3, 64, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 상태 크기. 64 x 32 x 32
#             nn.Conv2d(64, 128, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 상태 크기. 128 x 16 x 16
#             nn.Conv2d(128, 256, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 상태 크기. 256 x 8 x 8
#             nn.Conv2d(256, 512, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(0.2, inplace=True),
#             # 상태 크기. 512 x 4 x 4
#             nn.Conv2d(512, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )

#     def forward(self, input):
#         return self.main(input).view(-1, 1).squeeze(1)

# # DCGAN Generator 불러오기
# latent_dim = 100
# img_channels = 3
# feature_map_size = 64

# # DCGAN Generator 모델 정의 및 로딩
# generator = Generator()
# generator = generator.to(device)

# # 모델 파일에서 필요한 state_dict만 로드
# checkpoint = torch.load(r"C:\Code\Python\AIPhoto\dataset\output\11.25\generator_epoch_300.pth")

# # 기존의 checkpoint에서 키 이름 변경
# model_state_dict = checkpoint['model_state_dict']
# new_state_dict = {}

# for key in model_state_dict.keys():
#     # 'main.'을 'gen.'으로 변경
#     new_key = key.replace('main.', 'gen.')
#     new_state_dict[new_key] = model_state_dict[key]

# # generator에 로드
# generator.load_state_dict(new_state_dict, strict=False) # 모델 로드 시 strict=False로 불일치한 키를 무시
# generator.eval()

# # ===== ResNet-50 모델 정의 (사전 학습된 모델 사용) =====
# resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)  # ResNet-50 모델을 사전 학습된 상태로 불러오기

# # 마지막 FC layer의 출력 크기를 예측할 클래스 수로 변경 (여기서는 2로 설정)
# resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
# resnet_model = resnet_model.to(device)
# resnet_model.eval()

# # ===== 이미지 저장 경로 설정 =====
# output_dir = "./output/test"  # 저장 경로
# os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성

# # ===== DCGAN으로 이미지 생성 =====
# # def generate_image(generator, latent_dim, save_path=None):
# #     latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
# #     generated_image = generator(latent_vector).detach().cpu()

# #     # 64채널에서 첫 3채널만 선택
# #     generated_image = generated_image[:, :3, :, :]  # 64채널에서 첫 3개 채널만 사용

# #     # 이미지를 [-1, 1]에서 [0, 1]로 스케일링
# #     generated_image = (generated_image + 1) / 2

# #     # ToPILImage로 변환
# #     transform = transforms.ToPILImage()
# #     generated_image = transform(generated_image.squeeze(0))  # 배치 차원 제거

# #     if save_path:
# #         generated_image.save(save_path)
# #         print(f"이미지 저장 완료: {save_path}")
# #     return generated_image
# def generate_image(generator, latent_dim, save_path=None):
#     latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
#     generated_image = generator(latent_vector).detach().cpu()
#     # 이미지 후처리
#     transform = transforms.ToPILImage()
#     generated_image = transform(generated_image.squeeze(0))
    
#     if save_path:
#         generated_image.save(save_path)
#         print(f"이미지 저장 완료: {save_path}")
#     return generated_image


# # # ===== CNN으로 합격 예측 =====
# # def predict_success(cnn_model, image_path=None, generated_image=None):
# #     # 이미지 전처리
# #     preprocess = transforms.Compose([
# #         transforms.Resize((128, 128)),
# #         transforms.ToTensor(),
# #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
# #     ])

# #     if image_path:
# #         # 실제 이미지 사용
# #         image = Image.open(image_path).convert("RGB")
# #     elif generated_image:
# #         # 생성된 이미지 사용
# #         image = generated_image
# #     else:
# #         raise ValueError("image_path 또는 generated_image 중 하나를 제공해야 합니다.")

# #     input_tensor = preprocess(image).unsqueeze(0).to(device)

# #     # 예측
# #     with torch.no_grad():
# #         prediction = cnn_model(input_tensor)
# #         success_prob = prediction[0][1].item()  # 합격 확률
# #         return success_prob

# # ===== ResNet-50으로 합격 예측 =====
# def predict_success(resnet_model, image_path=None, generated_image=None):
#     # 이미지 전처리
#     preprocess = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet-50의 정규화 값 사용
#     ])

#     if image_path:
#         # 실제 이미지 사용
#         image = Image.open(image_path).convert("RGB")
#     elif generated_image:
#         # 생성된 이미지 사용
#         image = generated_image
#     else:
#         raise ValueError("image_path 또는 generated_image 중 하나를 제공해야 합니다.")

#     input_tensor = preprocess(image).unsqueeze(0).to(device)

#     # 예측
#     with torch.no_grad():
#         prediction = resnet_model(input_tensor)
#         success_prob = torch.sigmoid(prediction[0][1]).item()  # 로짓을 sigmoid를 통해 확률로 변환
#         return success_prob

# # ===== 실행 =====
# # (1) DCGAN으로 이미지 생성 및 저장
# generated_img_path = os.path.join(output_dir, "generated_image(11.25).jpg")  # 저장될 파일 경로
# generated_img = generate_image(generator, latent_dim, save_path=generated_img_path)

# # # (2) CNN으로 합격 예측
# # success_rate_generated = predict_success(cnn_model, generated_image=generated_img)
# # print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# # # (3) 실제 이미지로 예측 (옵션)
# # real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강민주.png"
# # success_rate_real = predict_success(cnn_model, image_path=real_image_path)
# # print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")

# # (2) ResNet-50으로 합격 예측
# success_rate_generated = predict_success(resnet_model, generated_image=generated_img)
# print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# # (3) 실제 이미지로 예측 (옵션)
# real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강민주.png"
# success_rate_real = predict_success(resnet_model, image_path=real_image_path)
# print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

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

    def forward(self, z):
        return self.main(z)
    
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
img_channels = 3
feature_map_size = 64

# DCGAN Generator 모델 정의 및 로딩
generator = Generator()
generator = generator.to(device)

# 모델 파일에서 필요한 state_dict만 로드
checkpoint = torch.load(r"C:\Code\Python\AIPhoto\dataset\output\11.25\generator_epoch_300.pth")

# 기존의 checkpoint에서 키 이름 변경
model_state_dict = checkpoint['model_state_dict']
new_state_dict = {}

for key in model_state_dict.keys():
    # 'main.'을 'gen.'으로 변경
    new_key = key.replace('main.', 'gen.')
    new_state_dict[new_key] = model_state_dict[key]

# generator에 로드
generator.load_state_dict(new_state_dict, strict=False) # 모델 로드 시 strict=False로 불일치한 키를 무시
generator.eval()

# ===== ResNet-50 모델 정의 (사전 학습된 모델 사용) =====
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)  # ResNet-50 모델을 사전 학습된 상태로 불러오기

# 마지막 FC layer의 출력 크기를 예측할 클래스 수로 변경 (여기서는 2로 설정)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# ===== 이미지 저장 경로 설정 =====
output_dir = "./output/test"  # 저장 경로
os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성

# ===== DCGAN으로 이미지 생성 =====
# def generate_image(generator, latent_dim, save_path=None):
#     latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
#     generated_image = generator(latent_vector).detach().cpu()

#     # 64채널에서 첫 3채널만 선택
#     generated_image = generated_image[:, :3, :, :]  # 64채널에서 첫 3개 채널만 사용

#     # 이미지를 [-1, 1]에서 [0, 1]로 스케일링
#     generated_image = (generated_image + 1) / 2

#     # ToPILImage로 변환
#     transform = transforms.ToPILImage()
#     generated_image = transform(generated_image.squeeze(0))  # 배치 차원 제거

#     if save_path:
#         generated_image.save(save_path)
#         print(f"이미지 저장 완료: {save_path}")
#     return generated_image
def generate_image(generator, latent_dim, save_path=None):
    latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
    generated_image = generator(latent_vector).detach().cpu()
    # 이미지 후처리
    transform = transforms.ToPILImage()
    generated_image = transform(generated_image.squeeze(0))
    
    if save_path:
        generated_image.save(save_path)
        print(f"이미지 저장 완료: {save_path}")
    return generated_image


# # ===== CNN으로 합격 예측 =====
# def predict_success(cnn_model, image_path=None, generated_image=None):
#     # 이미지 전처리
#     preprocess = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     if image_path:
#         # 실제 이미지 사용
#         image = Image.open(image_path).convert("RGB")
#     elif generated_image:
#         # 생성된 이미지 사용
#         image = generated_image
#     else:
#         raise ValueError("image_path 또는 generated_image 중 하나를 제공해야 합니다.")

#     input_tensor = preprocess(image).unsqueeze(0).to(device)

#     # 예측
#     with torch.no_grad():
#         prediction = cnn_model(input_tensor)
#         success_prob = prediction[0][1].item()  # 합격 확률
#         return success_prob

# ===== ResNet-50으로 합격 예측 =====
def predict_success(resnet_model, image_path=None, generated_image=None):
    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet-50의 정규화 값 사용
    ])

    if image_path:
        # 실제 이미지 사용
        image = Image.open(image_path).convert("RGB")
    elif generated_image:
        # 생성된 이미지 사용
        image = generated_image
    else:
        raise ValueError("image_path 또는 generated_image 중 하나를 제공해야 합니다.")

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        prediction = resnet_model(input_tensor)
        success_prob = torch.sigmoid(prediction[0][1]).item()  # 로짓을 sigmoid를 통해 확률로 변환
        return success_prob

# ===== 실행 =====
# (1) DCGAN으로 이미지 생성 및 저장
generated_img_path = os.path.join(output_dir, "generated_image(11.25).jpg")  # 저장될 파일 경로
generated_img = generate_image(generator, latent_dim, save_path=generated_img_path)

# # (2) CNN으로 합격 예측
# success_rate_generated = predict_success(cnn_model, generated_image=generated_img)
# print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# # (3) 실제 이미지로 예측 (옵션)
# real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강민주.png"
# success_rate_real = predict_success(cnn_model, image_path=real_image_path)
# print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")

# (2) ResNet-50으로 합격 예측
success_rate_generated = predict_success(resnet_model, generated_image=generated_img)
print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# (3) 실제 이미지로 예측 (옵션)
real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강민주.png"
success_rate_real = predict_success(resnet_model, image_path=real_image_path)
print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights

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

    def forward(self, z):
        return self.main(z)
    
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
img_channels = 3
feature_map_size = 64

# DCGAN Generator 모델 정의 및 로딩
generator = Generator()
generator = generator.to(device)

# 모델 파일에서 필요한 state_dict만 로드
checkpoint = torch.load(r"C:\Code\Python\AIPhoto\dataset\output\11.25\generator_epoch_300.pth")

# 기존의 checkpoint에서 키 이름 변경
model_state_dict = checkpoint['model_state_dict']
new_state_dict = {}

for key in model_state_dict.keys():
    # 'main.'을 'gen.'으로 변경
    new_key = key.replace('main.', 'gen.')
    new_state_dict[new_key] = model_state_dict[key]

# generator에 로드
generator.load_state_dict(new_state_dict, strict=False) # 모델 로드 시 strict=False로 불일치한 키를 무시
generator.eval()

# ===== ResNet-50 모델 정의 (사전 학습된 모델 사용) =====
resnet_model = resnet50(weights=ResNet50_Weights.DEFAULT)  # ResNet-50 모델을 사전 학습된 상태로 불러오기

# 마지막 FC layer의 출력 크기를 예측할 클래스 수로 변경 (여기서는 2로 설정)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# ===== 이미지 저장 경로 설정 =====
output_dir = "./output/test"  # 저장 경로
os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성

# ===== DCGAN으로 이미지 생성 =====
# def generate_image(generator, latent_dim, save_path=None):
#     latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
#     generated_image = generator(latent_vector).detach().cpu()

#     # 64채널에서 첫 3채널만 선택
#     generated_image = generated_image[:, :3, :, :]  # 64채널에서 첫 3개 채널만 사용

#     # 이미지를 [-1, 1]에서 [0, 1]로 스케일링
#     generated_image = (generated_image + 1) / 2

#     # ToPILImage로 변환
#     transform = transforms.ToPILImage()
#     generated_image = transform(generated_image.squeeze(0))  # 배치 차원 제거

#     if save_path:
#         generated_image.save(save_path)
#         print(f"이미지 저장 완료: {save_path}")
#     return generated_image
def generate_image(generator, latent_dim, save_path=None):
    latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
    generated_image = generator(latent_vector).detach().cpu()
    # 이미지 후처리
    transform = transforms.ToPILImage()
    generated_image = transform(generated_image.squeeze(0))
    
    if save_path:
        generated_image.save(save_path)
        print(f"이미지 저장 완료: {save_path}")
    return generated_image


# # ===== CNN으로 합격 예측 =====
# def predict_success(cnn_model, image_path=None, generated_image=None):
#     # 이미지 전처리
#     preprocess = transforms.Compose([
#         transforms.Resize((128, 128)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
#     ])

#     if image_path:
#         # 실제 이미지 사용
#         image = Image.open(image_path).convert("RGB")
#     elif generated_image:
#         # 생성된 이미지 사용
#         image = generated_image
#     else:
#         raise ValueError("image_path 또는 generated_image 중 하나를 제공해야 합니다.")

#     input_tensor = preprocess(image).unsqueeze(0).to(device)

#     # 예측
#     with torch.no_grad():
#         prediction = cnn_model(input_tensor)
#         success_prob = prediction[0][1].item()  # 합격 확률
#         return success_prob

# ===== ResNet-50으로 합격 예측 =====
def predict_success(resnet_model, image_path=None, generated_image=None):
    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ResNet-50의 정규화 값 사용
    ])

    if image_path:
        # 실제 이미지 사용
        image = Image.open(image_path).convert("RGB")
    elif generated_image:
        # 생성된 이미지 사용
        image = generated_image
    else:
        raise ValueError("image_path 또는 generated_image 중 하나를 제공해야 합니다.")

    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        prediction = resnet_model(input_tensor)
        success_prob = torch.sigmoid(prediction[0][1]).item()  # 로짓을 sigmoid를 통해 확률로 변환
        return success_prob

# ===== 실행 =====
# (1) DCGAN으로 이미지 생성 및 저장
generated_img_path = os.path.join(output_dir, "generated_image(11.25).jpg")  # 저장될 파일 경로
generated_img = generate_image(generator, latent_dim, save_path=generated_img_path)

# # (2) CNN으로 합격 예측
# success_rate_generated = predict_success(cnn_model, generated_image=generated_img)
# print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# # (3) 실제 이미지로 예측 (옵션)
# real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강민주.png"
# success_rate_real = predict_success(cnn_model, image_path=real_image_path)
# print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")

# (2) ResNet-50으로 합격 예측
success_rate_generated = predict_success(resnet_model, generated_image=generated_img)
print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# (3) 실제 이미지로 예측 (옵션)
real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강민주.png"
success_rate_real = predict_success(resnet_model, image_path=real_image_path)
print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")
