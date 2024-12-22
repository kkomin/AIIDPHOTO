import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

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

# 생성기와 판별기 초기화
generator = Generator()
discriminator = Discriminator()

# 체크포인트 로드
generator_checkpoint = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\11.26\generator_epoch_250.pth')
generator.load_state_dict(generator_checkpoint['model_state_dict'])

discriminator_checkpoint = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\11.26\discriminator_epoch_250.pth')
discriminator.load_state_dict(discriminator_checkpoint['model_state_dict'])

# 모델 평가 모드로 전환
generator.eval()
discriminator.eval()

# 이미지 전처리 함수 정의
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # DCGAN은 64x64 이미지를 처리
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # [-1, 1] 범위로 정규화
])

# 테스트 이미지 로드 및 "합격 확률" 계산
def evaluate_image(image_path):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('RGB')  # RGB로 변환
    image = transform(image)
    image = image.unsqueeze(0)  # 배치 차원 추가 (1, 3, 64, 64)

    # 판별기 통과
    with torch.no_grad():
        probability = discriminator(image).item()  # Sigmoid 결과를 확률로 변환

    print(f"Image: {image_path}, 합격 확률: {probability * 100:.2f}%")
    return probability

# 테스트 이미지 폴더에서 모든 이미지 평가
# test_dir = r'C:\Code\Python\AIPhoto\dataset\test'
# for file_name in os.listdir(test_dir):
#     file_path = os.path.join(test_dir, file_name)
#     if file_path.endswith(('.png', '.jpg', '.jpeg')):  # 이미지 파일만 처리
#         evaluate_image(file_path)

# 테스트할 이미지 경로를 직접 지정
image_path = r'C:\Code\Python\AIPhoto\dataset\test\강성규02.jpg'  # 평가할 이미지 경로

# 이미지 평가
evaluate_image(image_path)

# 디바이스 설정
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== DCGAN Generator 모델 정의 =====
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_map_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_size * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# DCGAN Generator 불러오기
latent_dim = 100
img_channels = 64
feature_map_size = 64

# DCGAN Generator 모델 정의 및 로딩
generator = Generator(latent_dim, img_channels, feature_map_size).to(device)

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
resnet_model = models.resnet50(pretrained=True)  # ResNet-50 모델을 사전 학습된 상태로 불러오기

# 마지막 FC layer의 출력 크기를 예측할 클래스 수로 변경 (여기서는 2로 설정)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 2)
resnet_model = resnet_model.to(device)
resnet_model.eval()

# ===== 이미지 저장 경로 설정 =====
output_dir = "./output/test"  # 저장 경로
os.makedirs(output_dir, exist_ok=True)  # 경로가 없으면 생성

# ===== DCGAN으로 이미지 생성 =====
def generate_image(generator, latent_dim, save_path=None):
    latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
    generated_image = generator(latent_vector).detach().cpu()

    # 64채널에서 첫 3채널만 선택
    generated_image = generated_image[:, :3, :, :]  # 64채널에서 첫 3개 채널만 사용

    # 이미지를 [-1, 1]에서 [0, 1]로 스케일링
    generated_image = (generated_image + 1) / 2

    # ToPILImage로 변환
    transform = transforms.ToPILImage()
    generated_image = transform(generated_image.squeeze(0))  # 배치 차원 제거

    if save_path:
        generated_image.save(save_path)
        print(f"이미지 저장 완료: {save_path}")
    return generated_image

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

# (2) ResNet-50으로 합격 예측
success_rate_generated = predict_success(resnet_model, generated_image=generated_img)
print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate_generated * 100:.2f}%")

# (3) 실제 이미지로 예측 (옵션)
real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\이태훈.jpg"
success_rate_real = predict_success(resnet_model, image_path=real_image_path)
print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")

# ===== DCGAN Generator 모델 정의 =====
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, feature_map_size):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(feature_map_size * 2, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

# DCGAN Generator 불러오기
latent_dim = 100
img_channels = 3
feature_map_size = 64
generator = Generator(latent_dim, img_channels, feature_map_size).to(device)
generator.load_state_dict(torch.load("dcgan_generator.pth"))  # DCGAN 학습된 모델 파일
generator.eval()

# ===== CNN 모델 정의 =====
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# CNN 모델 불러오기
num_classes = 2  # 합격/불합격
cnn_model = CNNModel(num_classes).to(device)
cnn_model.load_state_dict(torch.load("cnn_model.pth"))  # CNN 학습된 모델 파일
cnn_model.eval()

# ===== 이미지 생성 =====
# DCGAN으로 이미지 생성
def generate_image(generator, latent_dim):
    latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
    generated_image = generator(latent_vector).detach().cpu()
    # 이미지 후처리
    transform = transforms.ToPILImage()
    generated_image = transform(generated_image.squeeze(0))
    return generated_image

# ===== 합격 예측 =====
def predict_success(cnn_model, generated_image):
    # 이미지 전처리
    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    input_tensor = preprocess(generated_image).unsqueeze(0).to(device)

    # 예측
    with torch.no_grad():
        prediction = cnn_model(input_tensor)
        success_prob = prediction[0][1].item()  # 합격 확률
        return success_prob

# ===== 실행 =====
# (1) DCGAN으로 이미지 생성
generated_img = generate_image(generator, latent_dim)

# (2) 생성된 이미지로 합격 예측
success_rate = predict_success(cnn_model, generated_img)
print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate * 100:.2f}%")

# (3) 실제 이미지로 예측 (원하는 실제 이미지를 경로에 맞게 입력)
# real_image_path = "path_to_test_image.jpg"
# real_image = Image.open(real_image_path).convert("RGB")
# success_rate_real = predict_success(cnn_model, real_image)
# print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import transforms
# from PIL import Image

# # 디바이스 설정
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # ===== DCGAN Generator 모델 정의 =====
# class Generator(nn.Module):
#     def __init__(self, latent_dim, img_channels, feature_map_size):
#         super(Generator, self).__init__()
#         self.gen = nn.Sequential(
#             nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(feature_map_size * 8),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(feature_map_size * 4),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(feature_map_size * 2),
#             nn.ReLU(inplace=True),
#             nn.ConvTranspose2d(feature_map_size * 2, img_channels, 4, 2, 1, bias=False),
#             nn.Tanh()
#         )

#     def forward(self, x):
#         return self.gen(x)

# # DCGAN Generator 불러오기
# latent_dim = 100
# img_channels = 3
# feature_map_size = 64
# generator = Generator(latent_dim, img_channels, feature_map_size).to(device)
# generator.load_state_dict(torch.load("dcgan_generator.pth"))  # DCGAN 학습된 모델 파일
# generator.eval()

# # ===== CNN 모델 정의 =====
# class CNNModel(nn.Module):
#     def __init__(self, num_classes):
#         super(CNNModel, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.fc_layers = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(64 * 32 * 32, 128),
#             nn.ReLU(),
#             nn.Linear(128, num_classes),
#             nn.Softmax(dim=1)
#         )

#     def forward(self, x):
#         x = self.conv_layers(x)
#         x = self.fc_layers(x)
#         return x

# # CNN 모델 불러오기
# num_classes = 2  # 합격/불합격
# cnn_model = CNNModel(num_classes).to(device)
# cnn_model.load_state_dict(torch.load("cnn_model.pth"))  # CNN 학습된 모델 파일
# cnn_model.eval()

# # ===== 이미지 생성 및 예측 =====
# # DCGAN으로 이미지 생성
# def generate_image(generator, latent_dim):
#     latent_vector = torch.randn(1, latent_dim, 1, 1).to(device)
#     generated_image = generator(latent_vector).detach().cpu()
#     # 이미지 후처리
#     transform = transforms.ToPILImage()
#     generated_image = transform(generated_image.squeeze(0))
#     return generated_image

# # CNN으로 합격 예측
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

# # ===== 실행 =====
# # (1) DCGAN으로 이미지 생성
# generated_img = generate_image(generator, latent_dim)

# # (2) CNN으로 합격 예측
# success_rate = predict_success(cnn_model, generated_image=generated_img)
# print(f"DCGAN 생성 이미지의 합격 예측률: {success_rate * 100:.2f}%")

# # (3) 실제 이미지로 예측
# # real_image_path = "path_to_test_image.jpg"
# # success_rate_real = predict_success(cnn_model, image_path=real_image_path)
# # print(f"실제 이미지의 합격 예측률: {success_rate_real * 100:.2f}%")
