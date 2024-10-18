import sys
import os
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QProgressBar, QListWidget, 
                             QMessageBox, QLineEdit, QComboBox, QListWidgetItem, QToolTip)
from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import csv  # Aggiunto per la scrittura in CSV

# Definizione della classe CustomEfficientNet, che estende nn.Module di PyTorch
class CustomEfficientNet(nn.Module):
    def __init__(self, model_path):
        super(CustomEfficientNet, self).__init__()
        # Caricamento del modello EfficientNet B0
        self.efficientnet = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Caricamento dello state_dict del modello salvato
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Filtraggio dei parametri per EfficientNet
        efficientnet_state_dict = {k: v for k, v in state_dict.items() if k.startswith('efficientnet.')}
        efficientnet_state_dict = {k.replace('efficientnet.', ''): v for k, v in efficientnet_state_dict.items()}
        
        # Caricamento dei pesi per il modello EfficientNet
        self.efficientnet.load_state_dict(efficientnet_state_dict, strict=False)
        
        # Definizione del livello di attenzione
        self.attention = nn.Sequential(
            nn.Conv2d(1280, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # Definizione del classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )
        
        # Caricamento dei pesi del livello di attenzione, se presenti
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
        # Passaggio avanti nel modello per estrarre le feature
        features = self.efficientnet.forward_features(x)
        # Applicazione del livello di attenzione
        attention = self.attention(features)
        # Moltiplicazione delle feature per il livello di attenzione
        weighted_features = features * attention
        # Somma delle feature per ottenere un vettore di rappresentazione
        pooled_features = torch.sum(weighted_features, dim=[2, 3])
        # Passaggio nel classificatore finale
        return self.classifier(pooled_features)   

# Funzione per caricare il modello
def load_model(model_path):
    try:
        model = CustomEfficientNet(model_path)
        model.eval()  # Imposta il modello in modalità valutazione
        return model
    except Exception as e:
        raise RuntimeError(f"Errore nel caricamento del modello: {str(e)}")

# Funzione per preprocessare un'immagine
def preprocess_image(image_path):
    # Definizione delle trasformazioni per l'immagine
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Caricamento e trasformazione dell'immagine
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Funzione per classificare un'immagine
def classify_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)  # Ottieni l'output del modello
        probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Calcola le probabilità con softmax
        confidence, predicted = torch.max(probabilities, 1)  # Trova la classe con la massima probabilità
    return predicted.item(), confidence.item()

# Thread per la classificazione delle immagini in modo asincrono
class ImageClassifierThread(QThread):
    # Definizione dei segnali per aggiornare la GUI
    update_progress = pyqtSignal(int)
    update_result = pyqtSignal(str, str, float)
    classification_complete = pyqtSignal(list)

    def __init__(self, model, image_paths, paused):
        super().__init__()
        self.model = model
        self.image_paths = image_paths
        self.paused = paused
        self.stopped = False

    def run(self):
        results = []
        for i, image_path in enumerate(self.image_paths):
            if self.stopped:
                break
            while self.paused:
                self.sleep(1)
            try:
                # Preprocessing e classificazione dell'immagine
                image_tensor = preprocess_image(image_path)
                predicted, confidence = classify_image(self.model, image_tensor)
                result = "Normale" if predicted == 0 else "Anomalo"
                results.append((os.path.basename(image_path), result, confidence))
                self.update_result.emit(os.path.basename(image_path), result, confidence)
            except Exception as e:
                print(f"Errore nella classificazione dell'immagine {image_path}: {str(e)}")
                results.append((os.path.basename(image_path), "Errore", 0.0))
                self.update_result.emit(os.path.basename(image_path), "Errore", 0.0)
            finally:
                # Aggiornamento della progressione
                self.update_progress.emit(int((i + 1) / len(self.image_paths) * 100))
        self.classification_complete.emit(results)

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.stopped = True

# Classe principale della GUI per la classificazione delle immagini
class ImageClassifierGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Classificatore di Immagini")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QHBoxLayout(self.central_widget)
        self.setup_ui()
        self.image_paths = []
        self.results = []
        self.paused = False
        try:
            # Caricamento del modello all'avvio della GUI
            self.model = load_model(r"C:\MoDelli\newmodel_vision\b0_best_model.pth")
        except Exception as e:
            QMessageBox.critical(self, "Errore", f"Impossibile caricare il modello: {str(e)}")
            sys.exit(1)

    # Configurazione dell'interfaccia utente
    def setup_ui(self):
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Pulsante per selezionare le immagini
        self.select_button = QPushButton("Seleziona Immagini")
        self.select_button.clicked.connect(self.select_images)
        left_layout.addWidget(self.select_button)
        
        # Pulsante per avviare la classificazione
        self.classify_button = QPushButton("Classifica")
        self.classify_button.clicked.connect(self.start_classification)
        left_layout.addWidget(self.classify_button)
        
        # Barra di progresso per mostrare l'avanzamento della classificazione
        self.progress_bar = QProgressBar()
        left_layout.addWidget(self.progress_bar)
        
        # Lista per mostrare i risultati della classificazione
        self.result_list = QListWidget()
        self.result_list.setMouseTracking(True)
        self.result_list.itemEntered.connect(self.show_image_preview)
        left_layout.addWidget(self.result_list)
        
        # Barra di ricerca per filtrare i risultati
        self.search_bar = QLineEdit()
        self.search_bar.setPlaceholderText("Cerca...")
        self.search_bar.textChanged.connect(self.filter_results)
        left_layout.addWidget(self.search_bar)

        # Pulsante per mettere in pausa la classificazione
        self.pause_button = QPushButton("Pausa")
        self.pause_button.clicked.connect(self.pause_resume_classification)
        left_layout.addWidget(self.pause_button)
        
        # Pulsante per salvare i risultati in un file CSV
        self.save_button = QPushButton("Esporta Report")
        self.save_button.clicked.connect(self.save_report)
        left_layout.addWidget(self.save_button)

        # Pulsante per uscire dall'applicazione
        self.exit_button = QPushButton("Esci")
        self.exit_button.clicked.connect(self.close)
        left_layout.addWidget(self.exit_button)

        # Pulsante per mostrare un grafico dei risultati
        self.show_chart_button = QPushButton("Mostra Grafico")
        self.show_chart_button.clicked.connect(self.update_histogram)
        left_layout.addWidget(self.show_chart_button)

        # Pannello di destra per mostrare il grafico
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Aggiunta di un canvas per il grafico
        self.canvas = FigureCanvas(plt.Figure(figsize=(6, 8)))
        right_layout.addWidget(self.canvas)
        
        # Etichetta per mostrare i risultati della classificazione
        self.result_label = QLabel("Risultato della Classificazione")
        self.result_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(self.result_label)
        
        # Aggiunta dei pannelli alla finestra principale
        self.layout.addWidget(left_panel, 1)
        self.layout.addWidget(right_panel, 4)

    # Funzione per selezionare le immagini da classificare
    def select_images(self):
        file_dialog = QFileDialog()
        self.image_paths, _ = file_dialog.getOpenFileNames(self, "Seleziona Immagini", "", "Immagini (*.png *.jpg *.jpeg)")
        self.result_list.clear()
        
        # Pulisce il grafico precedente
        ax = self.canvas.figure.gca()
        ax.clear()
        self.canvas.draw()
        
        # Aggiunta delle immagini selezionate alla lista
        for path in self.image_paths:
            item = QListWidgetItem(os.path.basename(path))
            item.setIcon(QIcon(QPixmap(path).scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            item.setData(Qt.UserRole, path)  # Salva il percorso dell'immagine
            self.result_list.addItem(item)
        self.results = []

    # Funzione per avviare la classificazione delle immagini
    def start_classification(self):
        if not self.image_paths:
            QMessageBox.warning(self, "Attenzione", "Nessuna immagine selezionata.")
            return
        self.classify_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.results = []
        # Crea un thread per la classificazione
        self.classification_thread = ImageClassifierThread(self.model, self.image_paths, self.paused)
        self.classification_thread.update_progress.connect(self.update_progress)
        self.classification_thread.update_result.connect(self.update_result)
        self.classification_thread.classification_complete.connect(self.classification_finished)
        self.classification_thread.start()

    # Funzione per aggiornare la barra di progresso
    def update_progress(self, value):
        self.progress_bar.setValue(value)

    # Funzione per aggiornare i risultati nella lista
    def update_result(self, image_name, result, confidence):
        items = self.result_list.findItems(image_name, Qt.MatchExactly)
        if items:
            item = items[0]
            item.setText(f"{image_name}: {result} ({confidence:.2f})")
            if result == "Anomalo":
                item.setForeground(Qt.red)
            else:
                item.setForeground(Qt.darkGreen)  # Verde scuro per "Normale"
    
    # Funzione per mostrare un'anteprima dell'immagine al passaggio del mouse
    def show_image_preview(self, item):
        image_path = item.data(Qt.UserRole)
        pixmap = QPixmap(image_path).scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        QToolTip.showText(self.mapToGlobal(self.result_list.mapFromGlobal(self.cursor().pos())), "", self)
        QToolTip.showText(self.cursor().pos(), f"<img src='{image_path}' width='200' height='200'>", self.result_list)

    # Funzione per mettere in pausa o riprendere la classificazione
    def pause_resume_classification(self):
        if not self.paused:
            self.classification_thread.pause()
            self.pause_button.setText("Riprendi")
        else:
            self.classification_thread.resume()
            self.pause_button.setText("Pausa")
        self.paused = not self.paused

    # Funzione per salvare il report dei risultati in un file CSV
    def save_report(self):
        csv_path = QFileDialog.getSaveFileName(self, "Salva Report in CSV", "", "CSV files (*.csv)")[0]
        if csv_path:
            try:
                with open(csv_path, mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Nome Immagine", "Risultato", "Confidenza"])
                    for result in self.results:
                        writer.writerow([result[0], result[1], f"{result[2]:.2f}"])
                QMessageBox.information(self, "Completato", "Report CSV salvato con successo.")
            except Exception as e:
                QMessageBox.critical(self, "Errore", f"Errore durante il salvataggio del CSV: {str(e)}")

    # Funzione per filtrare i risultati della classificazione in base al testo di ricerca
    def filter_results(self, text):
        for index in range(self.result_list.count()):
            item = self.result_list.item(index)
            item.setHidden(text.lower() not in item.text().lower())

    # Funzione chiamata quando la classificazione è completata
    def classification_finished(self, results):
        self.results = results
        self.classify_button.setEnabled(True)

    # Funzione per aggiornare l'istogramma dei risultati
    def update_histogram(self):
        ax = self.canvas.figure.add_subplot(111)
        ax.clear()
        normal_count = sum(1 for _, result, _ in self.results if result == "Normale")
        anomaly_count = sum(1 for _, result, _ in self.results if result == "Anomalo")
        categories = ['Normale', 'Anomalo']
        counts = [normal_count, anomaly_count]
        ax.bar(categories, counts, color=['#006400', '#FF0000'])  # Verde scuro e rosso
        
        # Aggiunta delle percentuali sopra le barre
        total = normal_count + anomaly_count
        if total > 0:
            percentages = [count / total * 100 for count in counts]
            for i, (count, percentage) in enumerate(zip(counts, percentages)):
                ax.text(i, count + 0.5, f'{percentage:.1f}%', ha='center')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title("Distribuzione dei Risultati (con percentuali)")
        self.canvas.draw()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageClassifierGUI()
    window.show()
    sys.exit(app.exec_())






