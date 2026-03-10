import torch
from torchvision import datasets, transforms
from PIL import Image
import os

# 1. 데이터 다운로드 및 불러오기
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 2. 저장할 메인 폴더 생성
base_dir = 'mnist_all_images'
os.makedirs(base_dir, exist_ok=True)

# 3. 0부터 9까지 하위 폴더 미리 만들기
for i in range(10):
    os.makedirs(os.path.join(base_dir, str(i)), exist_ok=True)

print("이미지 추출을 시작합니다. 잠시만 기다려 주세요...")

# 4. 6만 장 반복문 돌면서 저장
for i in range(len(train_dataset)):
    # 데이터 꺼내기
    img_tensor, label = train_dataset[i]
    
    # 텐서를 이미지 형식으로 변환 (0~1 -> 0~255)
    img_array = (img_tensor.squeeze().numpy() * 255).astype('uint8')
    img = Image.fromarray(img_array)
    
    # 저장 경로 설정 (예: mnist_all_images/5/img_123.png)
    save_path = os.path.join(base_dir, str(label), f'img_{i}.png')
    img.save(save_path)

    # 진행 상황 출력 (10000장마다)
    if (i + 1) % 10000 == 0:
        print(f"{i + 1}번째 이미지 저장 중...")

print(f"✅ 모든 작업이 완료되었습니다! '{base_dir}' 폴더를 확인해 보세요.")