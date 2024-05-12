import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTFeatureExtractor
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import optuna
from optuna import trial as optuna_trial
from optuna.samplers import TPESampler

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

seed_everything(42)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        # 데이터셋 초기화 및 전처리 작업 수행
        self.data = data
        self.transform = transform

    def __len__(self):
        # 데이터셋의 샘플 수 반환
        return len(self.data)

    def __getitem__(self, idx):
        # 인덱스(idx)에 해당하는 샘플 반환
        image = Image.open(self.data['img_path'][idx])
        label = self.data['label'][idx]

        if self.transform:
            image = self.transform(image)
        
        return image, label.item()
    
# 데이터셋과 데이터로더 설정
transform = transforms.Compose([
    # 여기에 필요한 전처리 작업을 추가합니다.
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # 다른 전처리 작업들을 추가할 수 있습니다.
])

from torchvision.transforms import RandomRotation, RandomHorizontalFlip

# 데이터 증강을 위한 변환 정의
augmentation_transform = transforms.Compose([
    RandomRotation(degrees=30),  # 랜덤 회전 (±30도)
    RandomHorizontalFlip(p=0.5),  # 랜덤 수평 뒤집기
])

# CSV 파일을 읽어옴
df = pd.read_csv("train.csv")

# img_path와 upscale_img_path를 기준으로 병합하여 새로운 DataFrame 생성
df_1 = df[['img_path', 'label']]
df_2 = df[['upscale_img_path', 'label']]

# 컬럼명을 일치시킴
df_1.columns = ['img_path', 'label']
df_2.columns = ['img_path', 'label']

# 데이터프레임 병합
merged_df = pd.concat([df_1, df_2])

# 결과 출력
merged_df.info()

df = pd.read_csv('train.csv')

# 데이터를 train, validation, test로 나누기
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=42)
train_df, valid_df = train_test_split(train_df, test_size=0.1, random_state=42)

train_df.reset_index(drop=True, inplace=True)
valid_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

# 훈련 데이터, 검증 데이터, 테스트 데이터의 레이블을 합칩니다.
all_labels = pd.concat([train_df['label'], valid_df['label'], test_df['label']])

# 중복을 제거하여 unique한 레이블을 얻습니다.
unique_labels = all_labels.drop_duplicates()

# unique_labels에 대해 인덱스를 부여하여 딕셔너리를 만듭니다.
label_to_index = {}
idx = 0

for label in unique_labels:
    label_to_index[label] = idx
    idx += 1

# 각 데이터셋의 레이블을 딕셔너리를 사용하여 매핑합니다.
train_df['label'] = train_df['label'].map(label_to_index)
valid_df['label'] = valid_df['label'].map(label_to_index)
test_df['label'] = test_df['label'].map(label_to_index)

train_dataset = CustomDataset(train_df, transform=transforms.Compose([transform, augmentation_transform]))
valid_dataset = CustomDataset(valid_df, transform=transform)
test_dataset = CustomDataset(test_df, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

# Define model training function
def train_model(trial: optuna_trial.Trial, learning_rate: float) -> float:
    # ViT 모델 및 전이 학습 준비
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 손실 함수 및 최적화 기법 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 모델 학습
    epochs = 20
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # 검증 데이터셋을 이용한 성능 평가
        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            correct = 0
            total = 0
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                valid_loss += criterion(outputs.logits, labels).item()
                _, predicted = torch.max(outputs.logits, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_valid_loss = valid_loss / len(valid_loader.dataset)

            accuracy = 100 * correct / total

            # Report intermediate results to Optuna
            trial.report(epoch_valid_loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

    return accuracy

# Define objective function for Optuna
def objective(trial: optuna_trial.Trial) -> float:
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)

    # Train model and return validation loss
    return train_model(trial, learning_rate)

if __name__ == "__main__":
    # Set up Optuna
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),  # Use TPE sampler
    )

    # Run optimization
    study.optimize(objective, n_trials=10)

    # Get the best trial
    best_trial = study.best_trial

    # Print best hyperparameters and results
    print("Best trial:")
    print("  Value: ", best_trial.value)
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))