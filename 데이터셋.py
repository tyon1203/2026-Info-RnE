import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 데이터를 숫자로 변환하는 설정
transform = transforms.Compose([transforms.ToTensor()])

# MNIST 다운로드 코드 (root 경로에 data 폴더가 생깁니다)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

print("다운로드 완료!")
print(f"데이터 개수: {len(train_dataset)}")

# 실제로 잘 받았는지 하나만 그려보기
image, label = train_dataset[0]
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.show()

from PIL import Image
import os

# 'output' 폴더 생성
os.makedirs('mnist_images', exist_ok=True)

# 처음 10개만 이미지 파일로 저장해보기
for i in range(10):
    img = Image.fromarray(train_images[i])
    img.save(f'mnist_images/image_{i}_label_{train_labels[i]}.png')