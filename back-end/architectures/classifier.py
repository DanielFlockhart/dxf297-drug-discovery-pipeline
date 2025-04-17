import os, sys, json, random, warnings
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system.constants import *
from system.utils import *
warnings.simplefilter("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StableClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3) -> None:
        super(StableClassifier, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


class StableMoleculeClassifier:
    def __init__(self, input_dim: int, num_classes: int, num_effects: int, learning_rate: float) -> None:
        self.class_model = StableClassifier(input_dim, num_classes).to(device)
        self.effect_model = StableClassifier(input_dim, num_effects).to(device)
        self.class_criterion = nn.BCEWithLogitsLoss()
        self.effect_criterion = nn.BCEWithLogitsLoss()
        self.class_optimizer = optim.Adam(self.class_model.parameters(), lr=learning_rate * 0.2, weight_decay=1e-4)
        self.effect_optimizer = optim.Adam(self.effect_model.parameters(), lr=learning_rate * 0.2, weight_decay=1e-4)

    def _train_step(self, model: nn.Module, optimizer: optim.Optimizer, 
                    criterion: nn.Module, inputs: torch.Tensor, 
                    targets: torch.Tensor) -> float:
        model.train()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        return loss.item()
   
    def train_model(self, dataloader: DataLoader, save_frequency: int = 10, 
                    epochs: int = 200, restore_epoch: int = None, 
                    checkpoint_path: str = None) -> None:

        loss_graph_path = "loss_graph_classifier_stable.png"
        if restore_epoch:
            print(f"Restoring checkpoint from epoch {restore_epoch}")
            self.load_checkpoint(checkpoint_path)
        self.class_model.train()
        self.effect_model.train()
        
        class_loss_history = []
        effect_loss_history = []

        for epoch in range(epochs + 1):
            total_class_loss = 0.0
            total_effect_loss = 0.0

            for vectors, class_labels, effect_labels in dataloader:
                vectors = vectors.to(device)
                class_labels = class_labels.to(device)
                effect_labels = effect_labels.to(device)
                
                total_class_loss += self._train_step(self.class_model, self.class_optimizer,
                                                     self.class_criterion, vectors, class_labels)
                total_effect_loss += self._train_step(self.effect_model, self.effect_optimizer,
                                                      self.effect_criterion, vectors, effect_labels)
            
            class_loss_history.append(total_class_loss)
            effect_loss_history.append(total_effect_loss)
            avg_class_loss = total_class_loss / len(dataloader)
            avg_effect_loss = total_effect_loss / len(dataloader)
            print(f"Epoch {epoch}: avg_class_loss = {avg_class_loss:.4f}, avg_effect_loss = {avg_effect_loss:.4f}")
            
            if epoch % save_frequency == 0 and checkpoint_path is not None:
                self.save_checkpoint(epoch, checkpoint_path)

        if checkpoint_path is not None:
            self.save_checkpoint(epochs, checkpoint_path)

        plt.figure(figsize=(10, 5))
        epochs_range = range(epochs + 1)
        plt.plot(epochs_range, class_loss_history, label='Class Loss', linewidth=2)
        plt.plot(epochs_range, effect_loss_history, label='Effect Loss', linewidth=2)
        plt.title('Stable Training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_graph_path)
        plt.close()

    def save_checkpoint(self, epoch: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'class_model_state_dict': self.class_model.state_dict(),
            'effect_model_state_dict': self.effect_model.state_dict(),
            'class_optimizer_state_dict': self.class_optimizer.state_dict(),
            'effect_optimizer_state_dict': self.effect_optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, checkpoint_path: str) -> int:
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.class_model.load_state_dict(checkpoint['class_model_state_dict'])
            self.effect_model.load_state_dict(checkpoint['effect_model_state_dict'])
            self.class_optimizer.load_state_dict(checkpoint['class_optimizer_state_dict'])
            self.effect_optimizer.load_state_dict(checkpoint['effect_optimizer_state_dict'])
            return checkpoint['epoch']
        return None

    def load_model(self, model_path: str):

        if os.path.exists(model_path):
            model = torch.load(model_path)
            return model
        return None

    def predict_class_probabilities(self, vector: torch.Tensor, label_list: list, model: nn.Module) -> dict:

        model.eval()
        with torch.no_grad():
            logits = model(vector.to(device))
            probs = torch.sigmoid(logits)
            label_probabilities = {label_list[i]: round(prob.item() * 100, 2)
                                   for i, prob in enumerate(probs[0])}
            sorted_probabilities = dict(sorted(label_probabilities.items(),
                                               key=lambda item: item[1],
                                               reverse=True))
        return sorted_probabilities

    def predict(self, vector: torch.Tensor, class_list: list, effect_list: list) -> tuple:
        class_probs = self.predict_class_probabilities(vector, class_list, self.class_model)
        effect_probs = self.predict_class_probabilities(vector, effect_list, self.effect_model)
        return class_probs, effect_probs


class MoleculeVectorDataset(Dataset):
    def __init__(self, dataset: str) -> None:
        self.data = pd.read_csv(dataset)
        self.vectors = [json.loads(vector_str) for vector_str in self.data['VECTOR']]
        self.class_labels = [json.loads(label_str) for label_str in self.data['CLASSES']]
        self.effect_labels = [json.loads(label_str) for label_str in self.data['EFFECTS']]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        vector = torch.tensor(self.vectors[idx], dtype=torch.float32)
        class_label = torch.tensor(self.class_labels[idx], dtype=torch.float32)
        effect_label = torch.tensor(self.effect_labels[idx], dtype=torch.float32)
        return vector, class_label, effect_label


def train_classifier() -> None:
    dataset = MoleculeVectorDataset(DATASET)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    classifier = StableMoleculeClassifier(MOLECULE_LENGTH, CLASSES_COUNT, EFFECTS_COUNT, LEARNING_RATE)
    classifier.train_model(dataloader, epochs=CLASSIFIER_EPOCHS, save_frequency=SAVE_FREQUENCY,
                             restore_epoch=RESTORE_CLASSIFIER_EPOCH, checkpoint_path=CLASSIFIER_PATH)


def test_classifier():
    print("\n=== Testing the Stable MoleculeClassifier with a Random Sample ===\n")
    DATASET = f"../../datasets/{MODEL_NAME}/dataset.csv"
    CLASSIFIER_PATH = f"../models/{MODEL_NAME}/classifier/classifier_model.pth"
    dataset = MoleculeVectorDataset(DATASET)
    sample_index = random.randint(0, len(dataset) - 1)
    vector, true_class, true_effect = dataset[sample_index]
    molecule_name = dataset.data['MOLECULE'][sample_index]
    print(f"Sample Molecule: {molecule_name}\n")
    
    classifier = StableMoleculeClassifier(MOLECULE_LENGTH, CLASSES_COUNT, EFFECTS_COUNT, LEARNING_RATE)
    if os.path.isfile(CLASSIFIER_PATH):
        epoch = classifier.load_checkpoint(CLASSIFIER_PATH)
        print(f"Loaded classifier checkpoint from epoch: {epoch}\n")
    
    classifier.class_model.eval()
    classifier.effect_model.eval()
    input_vector = vector.unsqueeze(0)
    predicted_class_probs, predicted_effect_probs = classifier.predict(input_vector, classes, effects)
    
    print("=== Predicted Class Probabilities (≥1%) ===")
    for label, prob in predicted_class_probs.items():
        if prob >= 1.0:
            print(f"  • {label}: {prob:.2f}%")
    print()
    
    print("=== Predicted Effect Probabilities (≥1%) ===")
    for label, prob in predicted_effect_probs.items():
        if prob >= 1.0:
            print(f"  • {label}: {prob:.2f}%")
    print()


def test_classifier_with_user_smiles():
    CLASSIFIER_PATH = f"../models/{MODEL_NAME}/classifier/classifier_model.pth"
    user_smiles = input("Enter the SMILES of the molecule: ")
    standardised = standardise_complex_smiles(user_smiles)
    print(f"Standardised SMILES: {standardised}")
    vector = smile_to_vector_ChemBERTa(standardised)
    vector = torch.tensor(vector, dtype=torch.float32).unsqueeze(0)
    classifier = StableMoleculeClassifier(MOLECULE_LENGTH, CLASSES_COUNT, EFFECTS_COUNT, LEARNING_RATE)
    
    if os.path.isfile(CLASSIFIER_PATH):
        epoch = classifier.load_checkpoint(CLASSIFIER_PATH)
        print(f"Loaded classifier checkpoint from epoch: {epoch}\n")
    
    classifier.class_model.eval()
    classifier.effect_model.eval()
    
    predicted_class_probs, predicted_effect_probs = classifier.predict(vector, classes, effects)
    print("\n=== Predicted Class Probabilities (≥1%) ===")
    for label, prob in predicted_class_probs.items():
        if prob >= 1.0:
            print(f"  • {label}: {prob:.2f}%")
    print()
    
    print("=== Predicted Effect Probabilities (≥1%) ===")
    for label, prob in predicted_effect_probs.items():
        if prob >= 1.0:
            print(f"  • {label}: {prob:.2f}%")
    print()


if __name__ == "__main__":
    while True:
        test_classifier_with_user_smiles()