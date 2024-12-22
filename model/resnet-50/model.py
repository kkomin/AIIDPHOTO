import torch
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T

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

# 모델 준비 (ResNet-50 모델 로드)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50(pretrained=True).to(device)
resnet50.eval()

# DCGAN 모델 불러오기 (학습된 DCGAN 모델 로드)
checkpoint_G = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\11.26\generator_epoch_250.pth')
generator = Generator().to(device)
generator.load_state_dict(checkpoint_G['model_state_dict'])
generator.eval()  # 평가 모드로 설정

checkpoint_D = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\11.26\discriminator_epoch_250.pth')
discriminator = Discriminator().to(device)
discriminator.load_state_dict(checkpoint_D['model_state_dict'])
discriminator.eval()  # 평가 모드로 설정

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 실제 이미지 예측 함수 (ResNet-50)
def predict_with_resnet50(model, image):
    image = transform(image).unsqueeze(0).to(device)  # 이미지 전처리 후 배치 차원 추가
    with torch.no_grad():
        output = model(image)
    prob_real = torch.nn.functional.softmax(output, dim=1)[:, 1].item()  # 합격 확률
    return prob_real

# DCGAN으로 생성된 이미지 예측
def predict_announcer(generator, discriminator, model):
    # DCGAN을 통해 이미지를 생성
    noise = torch.randn(1, 100, 1, 1).to(device)  # 노이즈 벡터
    generated_image = generator(noise)  # DCGAN을 통해 이미지 생성
    generated_image = generated_image.squeeze().cpu().detach()  # 배치 차원 제거 후 CPU로 이동

    # 생성된 이미지를 PIL 형식으로 변환 후 ResNet-50 예측
    generated_image_pil = T.ToPILImage()(generated_image.clamp(0, 1))  # 생성된 텐서를 PIL 이미지로 변환
    prob_real_resnet = predict_with_resnet50(model, generated_image_pil)

    # 생성된 이미지에 대해 Discriminator로 진위 판별
    with torch.no_grad():
        discriminator_output = discriminator(generated_image.unsqueeze(0).to(device))  # 배치 차원 추가 후 Discriminator에 입력
        prob_real_discriminator = torch.sigmoid(discriminator_output).item()  # 진짜일 확률 (0~1)

    # 생성된 이미지를 보기 위한 출력
    plt.imshow(generated_image.permute(1, 2, 0).clamp(0, 1))  # 생성된 이미지를 출력
    plt.show()

    print(f"생성된 이미지 예측 - ResNet-50 합격 확률: {prob_real_resnet*100:.2f}%")
    print(f"생성된 이미지 예측 - Discriminator 진짜 확률: {prob_real_discriminator*100:.2f}%")

    return prob_real_resnet, prob_real_discriminator

# 실제 이미지를 예측하는 함수 (ResNet-50)
def predict_real_image(image_path, model):
    image = Image.open(image_path).convert("RGB")  # 실제 이미지 로드
    prob_real = predict_with_resnet50(model, image)  # 실제 이미지 예측
    print(f"실제 이미지 예측 - 합격 확률: {prob_real*100:.2f}%")
    return prob_real

# 예시: DCGAN으로 생성된 이미지를 예측 (ResNet-50, Discriminator)
generated_image_prob_resnet, generated_image_prob_discriminator = predict_announcer(generator, discriminator, resnet50)

# 예시: 실제 이미지를 예측
real_image_path = r"C:\Code\Python\AIPhoto\dataset\testdata\images\강민주.png"  # 실제 이미지 경로로 변경
real_image_prob = predict_real_image(real_image_path, resnet50)
