import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from PIL import Image
import os

# Definizione della classe del dataset
class ComponentDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data = self.data[self.data['split'] == os.path.basename(img_dir)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.iloc[idx]['label_num'], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label

# Definizione del modello struttura + pesi
class CustomEfficientNet(nn.Module):
    def __init__(self, model_path):
        super(CustomEfficientNet, self).__init__()
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.efficientnet.load_state_dict(state_dict)
        self.attention = nn.Sequential(
            nn.Conv2d(1280, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        features = self.efficientnet.forward_features(x)
        attention = self.attention(features)
        weighted_features = features * attention
        pooled_features = torch.sum(weighted_features, dim=[2, 3])
        return self.classifier(pooled_features)

# Definizione delle trasformazioni
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Percorsi
csv_file = r'C:\Users\Christian\dataset_componenti\etichette_dataset_improved.csv'
train_dir = r'C:\Users\Christian\dataset_componenti\train'
val_dir = r'C:\Users\Christian\dataset_componenti\val'
test_dir = r'C:\Users\Christian\dataset_componenti\test'
model_path = r"D:\MoDelli\vision_model\pytorch_model.bin"
save_dir = r"D:\MoDelli\newmodel_vision"

# Creazione dei dataset e dataloader
train_dataset = ComponentDataset(csv_file, train_dir, transform=train_transform)
val_dataset = ComponentDataset(csv_file, val_dir, transform=val_transform)
test_dataset = ComponentDataset(csv_file, test_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Creazione del modello
model = CustomEfficientNet(model_path)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definizione di loss function, optimizer e scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Training loop
num_epochs = 25
best_val_loss = float('inf')
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # Validazione
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(save_dir, 'b0_best_model.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Modello salvato in: {save_path}")

# Test finale
best_model_path = os.path.join(save_dir, 'b0_best_model.pth')
model.load_state_dict(torch.load(best_model_path))
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_acc = 100 * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')
print(f'Il modello finale è salvato in: {best_model_path}')