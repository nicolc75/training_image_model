# Importazione delle librerie necessarie
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
from PIL import Image
import os
import timm

# Definizione della classe per il dataset personalizzato
class ComponentDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        # Carica il file CSV con le informazioni sulle immagini
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        # Filtra i dati in base alla directory (train, val, test)
        self.data = self.data[self.data['split'] == os.path.basename(img_dir)]

    def __len__(self):
        # Restituisce il numero totale di immagini nel dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Carica un'immagine e la sua etichetta
        img_name = os.path.join(self.img_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.data.iloc[idx]['label_num'], dtype=torch.long)

        # Applica le trasformazioni se specificate
        if self.transform:
            image = self.transform(image)

        return image, label

# Definizione del modello ViT personalizzato
class CustomViT(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomViT, self).__init__()
        # Crea un modello ViT base
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False)
        
        # Carica il dizionario di stato del modello pre-addestrato
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Rimuovi il prefisso 'vit.' dalle chiavi se presente
        new_state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items()}
        
        # Carica il dizionario di stato nel modello
        self.vit.load_state_dict(new_state_dict, strict=False)
        
        # Sostituisci l'ultimo layer di classificazione per adattarlo al nostro task
        self.vit.head = nn.Linear(self.vit.head.in_features, num_classes)

    def forward(self, x):
        # Definisce il passaggio in avanti del modello
        return self.vit(x)

# Definizione delle trasformazioni da applicare alle immagini
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Ridimensiona l'immagine
    transforms.ToTensor(),  # Converte l'immagine in un tensore
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizza i valori dei pixel
])

# Definizione dei percorsi per i file e le directory
csv_file = r'C:\Users\Christian\dataset_componenti\etichette_dataset_improved.csv'
train_dir = r'C:\Users\Christian\dataset_componenti\train'
val_dir = r'C:\Users\Christian\dataset_componenti\val'
test_dir = r'C:\Users\Christian\dataset_componenti\test'
save_dir = r"D:\MoDelli\newmodel_vision"
model_path = r"D:\MoDelli\vision_model\vit-base\pytorch_model.bin"

# Creazione dei dataset per training, validazione e test
train_dataset = ComponentDataset(csv_file, train_dir, transform=transform)
val_dataset = ComponentDataset(csv_file, val_dir, transform=transform)
test_dataset = ComponentDataset(csv_file, test_dir, transform=transform)

# Creazione dei data loader per l'alimentazione dei dati durante l'addestramento
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Inizializzazione del modello
model = CustomViT(model_path)
# Selezione del dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Definizione della funzione di loss, dell'ottimizzatore e dello scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

# Ciclo di addestramento
num_epochs = 30
best_val_loss = float('inf')
os.makedirs(save_dir, exist_ok=True)

for epoch in range(num_epochs):
    model.train()  # Imposta il modello in modalità di addestramento
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    # Ciclo di addestramento su batch
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()  # Azzera i gradienti
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calcolo della loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Aggiornamento dei pesi

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Calcolo delle metriche di addestramento
    train_loss = train_loss / len(train_loader)
    train_acc = 100 * train_correct / train_total

    # Validazione
    model.eval()  # Imposta il modello in modalità di valutazione
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():  # Disabilita il calcolo dei gradienti
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calcolo delle metriche di validazione
    val_loss = val_loss / len(val_loader)
    val_acc = 100 * val_correct / val_total

    # Stampa dei risultati dell'epoca
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Aggiornamento dello scheduler
    scheduler.step(val_loss)

    # Salvataggio del modello se la loss di validazione è migliorata
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_path = os.path.join(save_dir, 'vit_best_model.pth')
        torch.save(model.state_dict(), save_path)
        print(f"Modello salvato in: {save_path}")

# Test finale
best_model_path = os.path.join(save_dir, 'vit_best_model.pth')
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

# Calcolo e stampa dell'accuratezza del test
test_acc = 100 * test_correct / test_total
print(f'Test Accuracy: {test_acc:.2f}%')
print(f'Il modello finale è salvato in: {best_model_path}')