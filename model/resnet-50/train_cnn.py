import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 모델을 불러오기 전에 먼저 Generator와 Discriminator 정의
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 랜덤 시드 고정
torch.manual_seed(42)  # 원하는 시드 값 (예: 42)
latent_vector = torch.randn(1, 100, 1, 1).to(device)

# DCGAN Generator 모델 정의
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# DCGAN Discriminator 모델 정의
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# Generator와 Discriminator 객체 생성
checkpoint_G = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\generator_epoch_300.pth')
generator = Generator().to(device)
generator.load_state_dict(checkpoint_G['model_state_dict'])
generator.eval()  # 평가 모드로 설정

checkpoint_D = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\discriminator_epoch_300.pth')
discriminator = Discriminator().to(device)
discriminator.load_state_dict(checkpoint_D['model_state_dict'])
discriminator.eval()  # 평가 모드로 설정

# 이미지 전처리 설정 (CNN 모델에 맞는 입력 형태로 변환)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# CNN 모델 준비 (예시: ResNet-18)
cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = cnn_model.fc.in_features
cnn_model.fc = nn.Linear(num_ftrs, 1)  # 1개의 출력 노드로 합격 확률 예측 (0~1 사이의 값)

# 출력에 Sigmoid 활성화 함수 추가
cnn_model = nn.Sequential(
    cnn_model,
    nn.Sigmoid()  # 합격 확률을 0과 1 사이로 출력
).to(device)

# Discriminator 평가 함수
def evaluate_discriminator(discriminator, real_images, generated_images, device):
    discriminator.eval()

    # 실제 이미지 예측
    real_images = real_images.to(device)
    real_labels = torch.ones(real_images.size(0)).to(device)  # 실제: 1
    real_outputs = discriminator(real_images)
    real_loss = nn.BCELoss()(real_outputs, real_labels)

    # 생성 이미지 예측
    generated_images = generated_images.to(device)
    fake_labels = torch.zeros(generated_images.size(0)).to(device)  # 가짜: 0
    fake_outputs = discriminator(generated_images)
    fake_loss = nn.BCELoss()(fake_outputs, fake_labels)

    # 총 Discriminator 손실
    total_loss = real_loss + fake_loss
    return total_loss.item()

# 이미지 예측 및 평가 함수
def predict_image_with_evaluation(image_path, cnn_model, generator, discriminator, device):
    # 이미지 전처리
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Generator로 생성된 이미지
    latent_vector = torch.randn(1, 100, 1, 1).to(device)
    generated_image = generator(latent_vector).to(device)

    # CNN 모델 예측
    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(generated_image)
        predicted_probability = output.item()

    # Discriminator 평가
    discriminator_loss = evaluate_discriminator(discriminator, image, generated_image, device)

    return predicted_probability, discriminator_loss

# 예시 실행
image_path = r"C:\Code\Python\AIPhoto\dataset\test_data\positive\강다솜01.jpg"
probability, discriminator_loss = predict_image_with_evaluation(image_path, cnn_model, generator, discriminator, device)

# 결과 출력
print(f"합격 확률: {probability * 100:.2f}%")
print(f"Discriminator Loss: {discriminator_loss:.4f}")