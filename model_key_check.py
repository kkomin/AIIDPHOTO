import torch
import torch.nn as nn

# 체크포인트 파일 로드
checkpoint = torch.load(r'C:\Code\Python\AIPhoto\dataset\test_model\generator_epoch_300.pth')
print(checkpoint.keys())

# Generator 클래스 정의
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

# 모델을 GPU 또는 CPU로 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator().to(device)

# 체크포인트에서 model_state_dict만 로드
try:
    generator.load_state_dict(checkpoint['model_state_dict'])
    print("model_state_dict가 정상적으로 로드되었습니다.")
except Exception as e:
    print(f"로딩 오류 발생: {e}")
