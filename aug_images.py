import random
from torchvision import transforms
from PIL import Image
import os

# 이미지 증강 정의
def random_color_jitter():
    brightness = random.uniform(0.5, 1.5)
    contrast = random.uniform(0.5, 1.5)
    saturation = random.uniform(0.5, 1.5)
    hue = random.uniform(-0.1, 0.1)
    return transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=(hue, hue))


# 이미지 증강 정의
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 좌우 반전
    # transforms.RandomRotation(15),    # -15도에서 15도 사이로 회전
    transforms.RandomRotation(10, fill=(255, 255, 255)),      # -10도에서 10도 사이로 회전
    transforms.Lambda(lambda img: random_color_jitter()(img)),  # 랜덤 컬러 조정
    # transforms.RandomResizedCrop(64, scale=(0.8, 1.2)),  # 크기 조정
    # transforms.ColorJitter(brightness=0.2, contrast=0.3) # 밝기와 대비 조정
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 2.0)),  # 가우시안 블러
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), fill=(255, 255, 255), scale=None, shear=None),  # 이미지의 미세한 변형
    transforms.RandomPerspective(distortion_scale=0.1, p=0.5, fill=(255, 255, 255)),  # 원근 왜곡
    transforms.Lambda(lambda img: transforms.functional.adjust_gamma(img, gamma=0.9, gain=1)),  # 감마 보정
])

# 새로운 폴더 생성 및 경로 지정
new_output_dir = "aug_filter"
if not os.path.exists(new_output_dir):
    os.makedirs(new_output_dir)

# 이미지 증강 및 새로운 폴더에 저장
def augment_images(input_dir):
    filenames = [f for f in os.listdir(input_dir) if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg")]
    total_files = len(filenames)
    print(f"총 {total_files} 개의 파일을 읽었습니다.")
    
    for index, filename in enumerate(filenames):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path).convert("RGB")

        print(f"현재 처리 중인 파일: {filename} (진행률: {index + 1}/{total_files})")
    
        for i in range(40):  # 각 이미지에 대해 30개의 증강된 버전 생성
            augmented_img = transform(img)
            
            # 저장할 파일명을 'aug_2.png', 'aug_3.png' 등으로 설정
            augmented_img.save(os.path.join(new_output_dir, f"aug_{index * 40 + i + 2}.png"))

# 이미지가 저장된 경로 지정
input_dir = r"C:\Code\Python\AIPhoto\train_image"
augment_images(input_dir)