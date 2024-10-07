Questo progetto contiene script Python per l'addestramento e l'inferenza di modelli di deep learning per la classificazione di immagini, utilizzando sia EfficientNet che Vision Transformer (ViT).

train_b0.py

Scopo: Addestra un modello EfficientNet-B0 personalizzato per la classificazione binaria di componenti (corretti/rotti).
Caratteristiche:

Utilizza un dataset personalizzato caricato da un file CSV.
Implementa data augmentation per il training set.
Salva il miglior modello basato sulla loss di validazione.




train_vit_base.py

Scopo: Addestra un modello Vision Transformer (ViT) per lo stesso compito di classificazione.
Caratteristiche:

Utilizza la stessa struttura del dataset del file train_b0.py.
Carica un modello ViT pre-addestrato e lo adatta al task specifico.
Implementa fine-tuning con ottimizzatore AdamW e learning rate scheduling.




enf_b0.py

Scopo: Script di inferenza per il modello EfficientNet-B0 addestrato.
Caratteristiche:

Permette la selezione di immagini attraverso un'interfaccia grafica.
Classifica le immagini selezionate e fornisce risultati con confidenza.
Salva i risultati in un file CSV con timestamp.




enf_vit.py

Scopo: Script di inferenza per il modello ViT addestrato.
Caratteristiche:

Funzionalità simili a enf_b0.py, ma utilizza il modello ViT.
Adatta il preprocessing delle immagini specifico per ViT.





Utilizzo:

Utilizzare train_b0.py o train_vit_base.py per addestrare il modello desiderato.
Una volta addestrato il modello, usare enf_b0.py o enf_vit.py per effettuare inferenze su nuove immagini.

Requisiti:

Python 3.x
PyTorch
torchvision
timm (per i modelli pre-addestrati)
PIL (per la manipolazione delle immagini)
pandas (per la gestione dei dati)

Nota: Assicurarsi di avere i percorsi corretti per i dataset e i modelli pre-addestrati prima di eseguire gli script.
Questo progetto dimostra l'implementazione e l'utilizzo di due architetture moderne di deep learning per un task di classificazione di immagini, offrendo flessibilità nella scelta del modello e facilità d'uso per l'inferenza
