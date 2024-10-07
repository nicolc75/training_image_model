# Importazione delle librerie necessarie
import torch  # Framework principale per il deep learning
import torch.nn as nn  # Moduli di rete neurale
import timm  # Libreria per modelli di computer vision pre-addestrati
import torchvision.transforms as transforms  # Trasformazioni per le immagini
from PIL import Image  # Libreria per la manipolazione delle immagini
import tkinter as tk  # Libreria per l'interfaccia grafica
from tkinter import filedialog  # Modulo per la selezione dei file
import os  # Operazioni sul sistema operativo
from datetime import datetime  # Gestione delle date e orari
import csv  # Gestione dei file CSV

# Definizione del modello personalizzato EfficientNet
class CustomEfficientNet(nn.Module):
    def __init__(self, model_path):
        super(CustomEfficientNet, self).__init__()
        # Creazione del modello EfficientNet-B0 di base
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Caricamento dei pesi del modello pre-addestrato
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Filtro e adattamento del dizionario di stato per EfficientNet
        efficientnet_state_dict = {k: v for k, v in state_dict.items() if k.startswith('efficientnet.')}
        efficientnet_state_dict = {k.replace('efficientnet.', ''): v for k, v in efficientnet_state_dict.items()}
        
        # Caricamento dei pesi filtrati nel modello
        self.efficientnet.load_state_dict(efficientnet_state_dict, strict=False)
        
        # Definizione del layer di attenzione
        self.attention = nn.Sequential(
            nn.Conv2d(1280, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Definizione del classificatore
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
        # Caricamento dei pesi del layer di attenzione, se presenti
        if 'attention.0.weight' in state_dict:
            self.attention[0].weight.data = state_dict['attention.0.weight']
            self.attention[0].bias.data = state_dict['attention.0.bias']
        
        # Caricamento dei pesi del classificatore, se presenti
        if 'classifier.0.weight' in state_dict:
            self.classifier[0].weight.data = state_dict['classifier.0.weight']
            self.classifier[0].bias.data = state_dict['classifier.0.bias']
            self.classifier[3].weight.data = state_dict['classifier.3.weight']
            self.classifier[3].bias.data = state_dict['classifier.3.bias']

    def forward(self, x):
        # Estrazione delle feature dall'EfficientNet
        features = self.efficientnet.forward_features(x)
        # Applicazione del meccanismo di attenzione
        attention = self.attention(features)
        weighted_features = features * attention
        # Pooling globale delle feature pesate
        pooled_features = torch.sum(weighted_features, dim=[2, 3])
        # Classificazione finale
        return self.classifier(pooled_features)

# Funzione per caricare il modello
def load_model(model_path):
    model = CustomEfficientNet(model_path)
    model.eval()  # Imposta il modello in modalità di valutazione
    return model

# Funzione per preprocessare l'immagine
def preprocess_image(image_path):
    # Definizione delle trasformazioni da applicare all'immagine
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ridimensionamento
        transforms.ToTensor(),  # Conversione in tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione
    ])
    image = Image.open(image_path).convert('RGB')  # Apertura e conversione dell'immagine in RGB
    return transform(image).unsqueeze(0)  # Aggiunta di una dimensione per il batch

# Funzione per classificare un'immagine
def classify_image(model, image_tensor):
    with torch.no_grad():  # Disattiva il calcolo del gradiente per l'inferenza
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Calcolo delle probabilità
        confidence, predicted = torch.max(probabilities, 1)  # Ottenimento della classe predetta e della confidenza
    return predicted.item(), confidence.item()

# Funzione per selezionare le immagini tramite GUI
def select_images():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di Tkinter
    file_paths = filedialog.askopenfilenames(filetypes=[("JPEG files", "*.jpg")])  # Apre il dialogo di selezione file
    return file_paths

# Funzione principale
def main():
    model_path = r"D:\MoDelli\newmodel_vision\b0_best_model.pth"  # Percorso del modello addestrato
    model = load_model(model_path)  # Caricamento del modello
    image_paths = select_images()  # Selezione delle immagini da classificare
    if not image_paths:
        print("Nessuna immagine selezionata.")
        return

    results = []
    correct_count = 0
    broken_count = 0

    # Ciclo di classificazione per ogni immagine selezionata
    for image_path in image_paths:
        image_tensor = preprocess_image(image_path)
        prediction, confidence = classify_image(model, image_tensor)
        
        class_name = "corretto" if prediction == 0 else "rotto"
        if prediction == 0:
            correct_count += 1
        else:
            broken_count += 1

        # Preparazione e stampa del risultato per ogni immagine
        result = f"Immagine: {os.path.basename(image_path)}, Classificazione: {class_name}, Confidenza: {confidence:.2f}"
        print(result)
        results.append([os.path.basename(image_path), class_name, f"{confidence:.2f}"])

    # Calcolo delle percentuali finali
    total_images = len(image_paths)
    correct_percentage = (correct_count / total_images) * 100
    broken_percentage = (broken_count / total_images) * 100

    # Stampa dei risultati finali
    print(f"\nRisultati finali:")
    print(f"Componenti corretti: {correct_percentage:.2f}%")
    print(f"Componenti rotti: {broken_percentage:.2f}%")

    # Salvataggio dei risultati in un file CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"classificazione_risultati_{timestamp}.csv"
    
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Nome Immagine", "Classificazione", "Confidenza"])
        writer.writerows(results)
    
    print(f"\nRisultati salvati in: {output_file}")

# Punto di ingresso dello script
if __name__ == "__main__":
    main()