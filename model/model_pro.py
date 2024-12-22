import os
import glob
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 랜덤 시드 고정
torch.manual_seed(42)

# Generator 모델 정의
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

# Discriminator 모델 정의
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

# 모델 불러오기
generator = Generator().to(device)
checkpoint_G = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\11.27\generator_epoch_300.pth')
generator.load_state_dict(checkpoint_G['model_state_dict'])
generator.eval()

discriminator = Discriminator().to(device)
checkpoint_D = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\11.27\discriminator_epoch_300.pth')
discriminator.load_state_dict(checkpoint_D['model_state_dict'])
discriminator.eval()

# CNN 모델 정의
cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = cnn_model.fc.in_features
cnn_model.fc = nn.Linear(num_ftrs, 1)  # 출력 노드 1개 (합격 확률)
cnn_model = nn.Sequential(cnn_model, nn.Sigmoid())
cnn_model = cnn_model.to(device)
cnn_model.eval()

# 이미지 전처리 설정
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# 평가 함수
def predict_image_with_evaluation(image_path, cnn_model, generator, discriminator, device):
    # 이미지 전처리
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Generator 생성 이미지
    latent_vector = torch.randn(1, 100, 1, 1).to(device)
    generated_image = generator(latent_vector).to(device)

    # Discriminator 평가
    real_labels = torch.ones(1, device=device)
    fake_labels = torch.zeros(1, device=device)

    real_outputs = discriminator(image)
    fake_outputs = discriminator(generated_image)

    loss_fn = nn.BCELoss()
    real_loss = loss_fn(real_outputs, real_labels)
    fake_loss = loss_fn(fake_outputs, fake_labels)
    discriminator_loss = (real_loss + fake_loss).item()

    # CNN 모델 예측
    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(image)
        predicted_probability = output.item()

    return predicted_probability, discriminator_loss

# 폴더 내 모든 이미지 경로를 가져오는 함수
def get_image_paths_from_folder(folder_path):
    image_paths = glob.glob(os.path.join(folder_path, "*.jpg")) + glob.glob(os.path.join(folder_path, "*.png"))
    return image_paths

# 새로운 데이터로 모델 평가
def evaluate_model_with_new_data_from_folder(folder_path, cnn_model, generator, discriminator, device):
    all_predictions = []
    all_labels = []
    all_probabilities = []

    # 폴더 내 이미지 경로 가져오기
    new_data_path = get_image_paths_from_folder(folder_path)

    for image_path in new_data_path:
        # 실제 라벨 추출 (예: 파일명에서 positive 여부로 라벨 결정)
        actual_label = 1 if 'positive' in image_path else 0

        # 예측 결과
        predicted_probability, _ = predict_image_with_evaluation(image_path, cnn_model, generator, discriminator, device)
        predicted_label = 1 if predicted_probability > 0.5 else 0

        all_predictions.append(predicted_label)
        all_labels.append(actual_label)
        all_probabilities.append(predicted_probability)

    # 정확도 계산
    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'일반화 DCGAN 성능 평가 정확도: {accuracy * 100:.2f}%')

    # 평균 합격 확률 계산
    average_probability = sum(all_probabilities) / len(all_probabilities)
    print(f'DCGAN 적합성 예측률 평균: {average_probability * 100:.2f}%')

# 특정 이미지 평가
def test_single_image(image_path, cnn_model, generator, discriminator, device):
    predicted_probability, _ = predict_image_with_evaluation(image_path, cnn_model, generator, discriminator, device)
    file_name_with_extension = os.path.basename(test_image_path)  # '도승민.png'
    file_name = os.path.splitext(file_name_with_extension)[0]
    print(f"Resnet50 {file_name} 이미지 적합성 예측률 : {predicted_probability * 100:.2f}%")

# 실행 코드
folder_path = r'C:\Code\Python\AIPhoto\dataset\test01'  # 폴더 경로
test_image_path = r'C:\Code\Python\AIPhoto\dataset\testdata\images\이태훈.jpg'  # 테스트할 특정 이미지 경로

# 폴더 내 이미지 평가
evaluate_model_with_new_data_from_folder(folder_path, cnn_model, generator, discriminator, device)

# 단일 이미지 테스트
test_single_image(test_image_path, cnn_model, generator, discriminator, device)