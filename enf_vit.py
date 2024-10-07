# Importazione delle librerie necessarie
import torch  # Libreria principale per il deep learning
import torch.nn as nn  # Moduli di rete neurale
import timm  # Libreria per modelli di computer vision pre-addestrati
import torchvision.transforms as transforms  # Trasformazioni per le immagini
from PIL import Image  # Libreria per la manipolazione delle immagini
import tkinter as tk  # Libreria per l'interfaccia grafica
from tkinter import filedialog  # Modulo per la selezione dei file
import os  # Operazioni sul sistema operativo
from datetime import datetime  # Gestione delle date e orari
import csv  # Gestione dei file CSV

# Definizione del modello personalizzato ViT (Vision Transformer)
class CustomViT(nn.Module):
    def __init__(self, model_path, num_classes=2):
        super(CustomViT, self).__init__()
        # Creazione del modello ViT base con il numero di classi specificato
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
        
        # Caricamento dei pesi del modello pre-addestrato
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Rimozione del prefisso 'vit.' dalle chiavi del dizionario di stato se presente
        new_state_dict = {k.replace('vit.', ''): v for k, v in state_dict.items()}
        
        # Caricamento del dizionario di stato nel modello
        self.vit.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        # Metodo forward per la propagazione in avanti dell'input attraverso il modello
        return self.vit(x)

# Funzione per caricare il modello
def load_model(model_path):
    model = CustomViT(model_path)
    model.eval()  # Imposta il modello in modalità di valutazione
    return model

# Funzione per preprocessare l'immagine
def preprocess_image(image_path):
    # Definizione delle trasformazioni da applicare all'immagine
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Ridimensionamento a 224x224 pixel
        transforms.ToTensor(),  # Conversione dell'immagine in un tensore
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalizzazione dei valori dei pixel
    ])
    image = Image.open(image_path).convert('RGB')  # Apertura e conversione dell'immagine in formato RGB
    return transform(image).unsqueeze(0)  # Applicazione delle trasformazioni e aggiunta di una dimensione per il batch

# Funzione per classificare un'immagine
def classify_image(model, image_tensor):
    with torch.no_grad():  # Disattiva il calcolo del gradiente per l'inferenza
        outputs = model(image_tensor)  # Passa l'immagine attraverso il modello
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Calcolo delle probabilità con softmax
        confidence, predicted = torch.max(probabilities, 1)  # Ottiene la classe predetta e la confidenza
    return predicted.item(), confidence.item()  # Restituisce la predizione e la confidenza come valori scalari

# Funzione per selezionare le immagini tramite GUI
def select_images():
    root = tk.Tk()
    root.withdraw()  # Nasconde la finestra principale di Tkinter
    file_paths = filedialog.askopenfilenames(filetypes=[("JPEG files", "*.jpg")])  # Apre il dialogo di selezione file
    return file_paths

# Funzione principale
def main():
    model_path = r"D:\MoDelli\newmodel_vision\vit_best_model.pth"  # Percorso del modello addestrato
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
        
        # Interpretazione della predizione
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