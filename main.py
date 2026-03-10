import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 1. 모델 정의 (LeNet-5)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2), # 28x28 -> 28x28
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),               # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5, stride=1),          # 14x14 -> 10x10
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),               # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits

# 2. 하이퍼파라미터 설정
batch_size = 64
learning_rate = 0.001
epochs = 3  # 정확도가 너무 높지 않게 하려면 에포크를 낮게 설정하세요.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. 데이터 로드 (여기서 shuffle=True/False가 데이터 순서 연구의 핵심입니다)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# 연구 포인트: shuffle을 False로 하고 직접 샘플러를 만들면 순서 제어가 가능합니다.
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 4. 모델, 손실함수, 최적화 도구 초기화
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 학습 (Train)
print("학습 시작...")
model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# 6. 테스트 (Test)
print("\n테스트 시작...")
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'최종 테스트 정확도: {100 * correct / total:.2f}%')