from PIL import Image
import os

# 분할 대상 이미지 경로
input_image_path = r"C:\Users\ASUS\Downloads\model_save\11.27\gan_train_images(11.27)\epoch_200_images.jpg"  # 8x8로 구성된 이미지 경로
output_folder = r"C:\Code\Python\AIPhoto\dataset\output_images_200better"  # 분할된 이미지를 저장할 폴더 경로

# 출력 폴더 생성
os.makedirs(output_folder, exist_ok=True)

# 이미지 열기
image = Image.open(input_image_path)
width, height = image.size

# 분할 설정: 8x8 그리드로 가정
grid_size = 8
tile_width = width // grid_size
tile_height = height // grid_size

# 이미지 저장 크기 설정 (업스케일 크기)
upscale_width = 512  # 저장할 이미지의 가로 크기
upscale_height = 512  # 저장할 이미지의 세로 크기

# 이미지 분할 및 고화질 저장
counter = 1
for row in range(grid_size):
    for col in range(grid_size):
        # 분할 영역 정의
        left = col * tile_width
        upper = row * tile_height
        right = (col + 1) * tile_width
        lower = (row + 1) * tile_height

        # 이미지를 자르고 업스케일링
        cropped_image = image.crop((left, upper, right, lower))
        resized_image = cropped_image.resize((upscale_width, upscale_height), Image.LANCZOS)  # 고품질 리샘플링

        # 최상의 화질로 저장
        output_path = os.path.join(output_folder, f"tile_{counter:02d}.png")
        resized_image.save(output_path, format="PNG", optimize=True)  # PNG로 저장 (무손실 압축)
        print(f"Saved: {output_path}")
        counter += 1

print(f"이미지 분할 및 저장 완료! 총 {counter - 1}개 이미지가 저장되었습니다.")