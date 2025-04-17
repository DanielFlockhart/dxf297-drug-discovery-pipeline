import os,sys
import json
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
if __name__ == "__main__":
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system.constants import *
from system.utils import *
import random
import warnings
import matplotlib.pyplot as plt
warnings.simplefilter("ignore", category=FutureWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MolecularAutoencoder(nn.Module):
    def __init__(self, input_size: int, latent_dim: int) -> None:
        super(MolecularAutoencoder, self).__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size)
        )

    def forward(self, x: torch.Tensor) -> tuple:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return latent, reconstructed

    def train_model(self, dataset, epochs: int, optimizer, criterion, start_epoch: int = 0,
                path: str = None, restore_epochs: int = None, batch_size: int = 32,
                ) -> None:
        loss_graph_path = "loss_graph_auto_encoder.png"  
        if restore_epochs:
            self.load_autoencoder(optimizer, path)

        self.to(device)
        tensor_data = torch.tensor(dataset, dtype=torch.float32)
        train_dataset = TensorDataset(tensor_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        loss_history = []

        for epoch in custom_tqdm(range(start_epoch, epochs + 1), desc="Training Epochs", unit="epoch"):
            total_loss = 0.0
            for batch in train_loader:
                input_vector = batch[0].to(device)
                _, reconstructed = self(input_vector)
                loss = criterion(reconstructed, input_vector)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * input_vector.size(0)

            avg_loss = total_loss / len(tensor_data)
            loss_history.append(avg_loss)
            
            tqdm.write(f"Epoch {epoch} completed, Avg Loss: {avg_loss:.4f}")
            
            if (epoch % 100) == 0:
                print("Saving latent vectors")
                latent_vectors = self.create_latent_vectors(dataset)
                dataset, smiles_list, molecule_list = preprocess_dataset(DATASET)
                self.save_latent_vectors(latent_vectors, smiles_list, molecule_list, dataset,
                                        output_file=LATENT_PATH, effects_count=EFFECTS_COUNT, classes_count=CLASSES_COUNT)
            
            if path is not None and epoch % 5 == 0:
                self.save_autoencoder(optimizer, epoch, path)
        
        if path is not None:
            self.save_autoencoder(optimizer, epochs, path)
        
        if loss_graph_path is not None:
            plt.figure(figsize=(10, 5))
            plt.plot(range(start_epoch, epochs + 1), loss_history, linewidth=2)
            plt.title("Auto-encoder Training Loss over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.grid(True)
            plt.savefig("loss_graph_auto_encoder.png")
            plt.close()


    def save_autoencoder(self, optimizer, epoch: int, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)

    def load_autoencoder(self, optimizer, checkpoint_path: str) -> int:
       
        print("Loading autoencoder checkpoint from " + checkpoint_path)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return checkpoint['epoch']
        return None

    def create_latent_vectors(self, dataset, batch_size: int = 32) -> torch.Tensor:

        self.encoder.eval()
        tensor_data = torch.tensor(dataset, dtype=torch.float32)
        latent_vectors = []
        data_loader = DataLoader(TensorDataset(tensor_data), batch_size=batch_size)
        with torch.no_grad():
            for batch in data_loader:
                input_vector = batch[0].to(device)
                latent = self.encoder(input_vector)
                latent_vectors.append(latent)
        return torch.cat(latent_vectors, dim=0)

    def save_latent_vectors(self, latent_vectors: torch.Tensor, smiles_list: list, molecule_list: list,
                              dataset, output_file: str = "latent_vectors.csv", effects_count: int = 0,
                              classes_count: int = 0) -> None:
        data_effects = []
        data_classes = []
        data_vectors = []
        for sample in dataset:
            effects_sample = sample[:effects_count]
            classes_sample = sample[effects_count:effects_count + classes_count]
            vector = sample[effects_count + classes_count:]
            data_effects.append(json.dumps([float(x) for x in effects_sample]))
            data_classes.append(json.dumps([float(x) for x in classes_sample]))
            data_vectors.append(json.dumps([float(x) for x in vector]))

        data = {
            "MOLECULE": molecule_list,
            "SMILES": smiles_list,
            "VECTOR": data_vectors,
            "LATENT_VECTOR": [json.dumps([float(value) for value in latent]) for latent in latent_vectors],
            "EFFECTS": data_effects,
            "CLASSES": data_classes,
        }
        df = pd.DataFrame(data)
        print("Saving latent vectors to " + output_file)
        df.to_csv(output_file, index=False)


def train_autoencoder():
    # DATASET = "../datasets/zinc/dataset.csv"
    # USE_FULL_DATASET = False
    # fraction = 0.1
    model = MolecularAutoencoder(input_size=AUTOENCODER_INPUT_SIZE, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    dataset, smiles_list, molecule_list = preprocess_dataset(DATASET)
    
    latent_vectors = model.create_latent_vectors(dataset)
    model.save_latent_vectors(latent_vectors, smiles_list, molecule_list, dataset,
                              output_file=LATENT_PATH, effects_count=EFFECTS_COUNT, classes_count=CLASSES_COUNT)
    
    model.train_model(dataset, AUTOENCODER_EPOCHS, optimizer, criterion, path=AUTOENCODER_PATH,
                      restore_epochs=RESTORE_AUTOENCODER_EPOCH, batch_size=32)

def test_autoencoder():
    print("\n=== Testing MolecularAutoencoder for Reconstruction Performance ===")

    model = MolecularAutoencoder(input_size=AUTOENCODER_INPUT_SIZE, latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if os.path.exists(AUTOENCODER_PATH):
        loaded_epoch = model.load_autoencoder(optimizer, AUTOENCODER_PATH)
        if loaded_epoch is not None:
            print(f"Loaded autoencoder checkpoint from epoch {loaded_epoch}")
        else:
            print("No valid checkpoint found; proceeding with untrained model.")
    else:
        print("No checkpoint found; proceeding with untrained model.")

    model.eval() 

    dataset, smiles_list, molecule_list = preprocess_dataset(DATASET)
    test_data = torch.tensor(dataset, dtype=torch.float32)
    test_loader = DataLoader(TensorDataset(test_data), batch_size=32, shuffle=False)

    criterion = nn.MSELoss(reduction='sum')
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            input_vector = batch[0].to(device)
            _, reconstructed = model(input_vector)
            loss = criterion(reconstructed, input_vector)
            total_loss += loss.item()
            total_samples += input_vector.size(0)

    avg_loss = total_loss / total_samples
    print(f"Average MSE Reconstruction Loss: {avg_loss:.4f}")

    sample_idx = random.randint(0, len(dataset) - 1)
    sample_input = test_data[sample_idx].unsqueeze(0).to(device)
    with torch.no_grad():
        _, sample_recon = model(sample_input)

    sample_recon = sample_recon.squeeze(0).cpu().numpy()
    original_data = test_data[sample_idx].cpu().numpy()

    print("\n=== Example Reconstruction ===")
    print(f"Random sample index: {sample_idx}")
    print(f"Original (first 5 dims):     {original_data[:5]}")
    print(f"Reconstructed (first 5 dims): {sample_recon[:5]}\n")
