import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import struct
import os

# --- [설정: 정확도 조절 및 환경 설정] ---
SUBSET_RATIO = 0.05   # 학습 데이터 사용량 (정확도를 낮추기 위해 조정)
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. IDX 바이너리 파일 읽기 함수
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# 2. 데이터 로드 및 전처리
raw_path = './data/raw/'
try:
    train_images = read_idx(os.path.join(raw_path, 'train-images-idx3-ubyte'))
    train_labels = read_idx(os.path.join(raw_path, 'train-labels-idx1-ubyte'))
    test_images = read_idx(os.path.join(raw_path, 't10k-images-idx3-ubyte'))
    test_labels = read_idx(os.path.join(raw_path, 't10k-labels-idx1-ubyte'))
except FileNotFoundError:
    print("에러: data/raw/ 폴더에 MNIST 바이너리 파일이 없습니다.")
    exit()

# Tensor 변환 및 정규화
train_X = torch.tensor(train_images, dtype=torch.float32).unsqueeze(1) / 255.0
train_y = torch.tensor(train_labels, dtype=torch.long)
test_X = torch.tensor(test_images, dtype=torch.float32).unsqueeze(1) / 255.0
test_y = torch.tensor(test_labels, dtype=torch.long)

full_train_ds = TensorDataset(train_X, train_y)
test_ds = TensorDataset(test_X, test_y)

# 일부 데이터만 추출 (정확도 70%대 유도)
train_indices = list(range(int(len(full_train_ds) * SUBSET_RATIO)))
train_subset = Subset(full_train_ds, train_indices)

# shuffle=False (데이터 순서 고정 연구용)
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# 3. LeNet-5 모델 정의
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(),
            nn.AvgPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(),
            nn.AvgPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 16*5*5)
        return self.classifier(x)

model = LeNet5().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 4. 학습 루프
print(f"학습 시작 (데이터 사용량: {SUBSET_RATIO*100}%)...")
model.train()
for epoch in range(EPOCHS):
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{EPOCHS}] 완료")

# 5. 테스트 및 맞은 것/틀린 것 분류 추출
print("\n테스트 및 데이터 분류 중...")
model.eval()

wrong_images, wrong_labels, predicted_labels = [], [], []
correct_images, correct_labels = [], []
total_correct = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        preds = outputs.argmax(dim=1)
        
        # 틀린 데이터 마스킹
        wrong_mask = (preds != labels)
        if wrong_mask.any():
            wrong_images.append(images[wrong_mask].cpu())
            wrong_labels.append(labels[wrong_mask].cpu())
            predicted_labels.append(preds[wrong_mask].cpu())
            
        # 맞은 데이터 마스킹
        correct_mask = (preds == labels)
        if correct_mask.any():
            correct_images.append(images[correct_mask].cpu())
            correct_labels.append(labels[correct_mask].cpu())
            
        total_correct += correct_mask.sum().item()

# 텐서 합치기
all_wrong_imgs = torch.cat(wrong_images)
all_wrong_labs = torch.cat(wrong_labels)
all_preds = torch.cat(predicted_labels)

all_correct_imgs = torch.cat(correct_images)
all_correct_labs = torch.cat(correct_labels)

final_acc = (total_correct / len(test_ds)) * 100

# 6. 결과 저장
results_path = './results'
os.makedirs(results_path, exist_ok=True)

# 틀린 데이터 저장 (.pt)
torch.save({
    'images': all_wrong_imgs,
    'labels': all_wrong_labs,
    'predictions': all_preds
}, os.path.join(results_path, 'wrong_data.pt'))

# 맞은 데이터 저장 (.pt)
torch.save({
    'images': all_correct_imgs,
    'labels': all_correct_labs
}, os.path.join(results_path, 'correct_data.pt'))

print("-" * 35)
print(f"최종 테스트 정확도: {final_acc:.2f}%")
print(f"맞은 데이터 개수: {len(all_correct_imgs)}개 -> correct_data.pt 저장됨")
print(f"틀린 데이터 개수: {len(all_wrong_imgs)}개 -> wrong_data.pt 저장됨")
print("-" * 35)

if len(all_wrong_labs) > 0:
    print(f"샘플(틀린 것) - 실제정답: {all_wrong_labs[0]}, 모델예측: {all_preds[0]}")