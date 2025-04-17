import os
import csv
import math
import random
import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from rdkit.Chem import QED
import sys
if __name__ == "__main__":
   sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system.constants import *
from system.utils import *


from architectures.classifier import *
from architectures.autoencoder import *
from architectures.decoder import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
torch.set_num_threads(os.cpu_count())

class ConditionalTransformer(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        effects_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super(ConditionalTransformer, self).__init__()
        self.latent_dim = latent_dim
        self.effects_dim = effects_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.latent_embedding = nn.Linear(latent_dim, hidden_dim)
        self.effects_embedding = nn.Linear(effects_dim, hidden_dim)
        self.positional_encoding = self.create_positional_encoding(max_len=250, d_model=hidden_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )


    def create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, latent_vector: torch.Tensor, effects_vector: torch.Tensor) -> torch.Tensor:
        latent_emb = self.latent_embedding(latent_vector).unsqueeze(1)
        effects_emb = self.effects_embedding(effects_vector).unsqueeze(1)
        input_seq = torch.cat([latent_emb, effects_emb], dim=1)
        seq_len = input_seq.size(1)
        input_seq = input_seq + self.positional_encoding[:, :seq_len, :].to(input_seq.device)
        transformer_output = self.encoder(input_seq.permute(1, 0, 2))
        output_seq = transformer_output.permute(1, 0, 2)[:, -1, :]
        return self.output_layer(output_seq)

    def restore_transformer(self, model_path: str) -> None:
        self.load_state_dict(torch.load(model_path))
        self.eval()

    def pre_train(
        self,
        model: "ConditionalTransformer",
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module,
        latent_vectors: torch.Tensor,
        molecular_vectors: torch.Tensor,
        molecule_effects: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 32,
        path: Optional[Dict[str, str]] = None,
        restore_epochs: Optional[int] = None,
    ) -> None:
        model = model.to(device)
        latent_vectors = latent_vectors.to(device)
        molecular_vectors = molecular_vectors.to(device)
        molecule_effects = molecule_effects.to(device)
        
        dataset = torch.utils.data.TensorDataset(latent_vectors, molecule_effects, molecular_vectors)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        if restore_epochs and path is not None:
            print(f"Restoring from epoch {restore_epochs}")
            self.load_checkpoint(path)
        
        loss_history = []
        
        for epoch in custom_tqdm(range(epochs), desc="Training Epochs", unit="epoch"):
            total_loss = 0.0
            batch_count = 0

            for latent_batch, effect_batch, target_batch in dataloader:
                optimizer.zero_grad()    
                output_molecules = self(latent_batch, effect_batch)
                loss = loss_fn(output_molecules, target_batch)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            loss_history.append(avg_loss)
            
            if path is not None and (epoch + 1) % SAVE_FREQUENCY == 0 and epoch > 0:
                print("Saving model...")
                self.save_model(model, path, epoch + 1, optimizer)
            tqdm.write(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

        # Hardcoded path for the pre-training loss graph
        graph_file_path = "loss_graph_pre_trained.png"
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, epochs + 1), loss_history, linewidth=2)  # Continuous line without markers
        plt.title('Pre-training Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Average Loss')
        plt.grid(True)
        plt.savefig(graph_file_path)
        plt.close()

    def save_model(self, model: "ConditionalTransformer", path: Dict[str, str],epoch, optimizer) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[int]:
        if checkpoint_path is not None and os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            self.load_state_dict(checkpoint["model_state_dict"])
            return checkpoint.get("epoch")
        return None



class GeneticAlgorithm:
    def __init__(self, population_size: int, num_scientists: int, mutation_rate: float = 0.001) -> None:
        self.population_size = population_size
        self.num_scientists = num_scientists
        self.mutation_rate = mutation_rate

    def mutate_state_dict(self,
                      state_dict: Dict[str, torch.Tensor],
                      mutation_rate: float,
                      noise_std: float = 0.02) -> Dict[str, torch.Tensor]:
        """Flip each weight with probability = mutation_rate and add N(0, noise_std²) noise."""
        mutated_state_dict = {}
        with torch.no_grad():
            for k, v in state_dict.items():
                if "weight" in k and isinstance(v, torch.Tensor):
                    mask = torch.rand_like(v) < mutation_rate          # True where we mutate
                    noise = torch.zeros_like(v).normal_(0.0, noise_std)
                    mutated = v.clone()
                    mutated[mask] += noise[mask]
                    mutated_state_dict[k] = mutated
                else:
                    mutated_state_dict[k] = v
        return mutated_state_dict


    def crossover(
        self, parent1_state: Dict[str, torch.Tensor], parent2_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        offspring_state_dict = {}
        with torch.no_grad():
            for key in parent1_state.keys():
                if "weight" in key and isinstance(parent1_state[key], torch.Tensor):
                    p1 = parent1_state[key]
                    p2 = parent2_state[key]
                    mask = torch.randint(0, 2, p1.shape, device=p1.device, dtype=torch.bool)
                    offspring_weights = torch.where(mask, p1, p2)
                    offspring_state_dict[key] = offspring_weights
                else:
                    offspring_state_dict[key] = parent1_state[key]
        return offspring_state_dict
    def update_population(self, fitness_scores: List[float], scientists: List[ConditionalTransformer]) -> List[ConditionalTransformer]:
        sorted_indices = np.argsort(fitness_scores)[::-1]
        num_total = self.num_scientists

        num_elite = max(1, int(num_total * 0.25))
        num_mutate = max(1, int(num_total * 0.25))
        num_crossover = max(1, int(num_total * 0.25))
        num_mutate_crossover = num_total - (num_elite + num_mutate + num_crossover)

        new_population: List[ConditionalTransformer] = []

        for idx in sorted_indices[:num_elite]:
            elite = self._create_scientist_copy(scientists[idx]).to(device)
            elite.load_state_dict(scientists[idx].state_dict())
            new_population.append(elite)

        for idx in sorted_indices[num_elite:num_elite + num_mutate]:
            mutant = self._create_scientist_copy(scientists[idx]).to(device)
            mutated_state = self.mutate_state_dict(scientists[idx].state_dict(), self.mutation_rate)
            mutant.load_state_dict(mutated_state)
            new_population.append(mutant)

        parent_pool = [scientists[idx] for idx in sorted_indices[:max(1, num_total // 2)]]
        for _ in range(num_crossover):
            parent1, parent2 = np.random.choice(parent_pool, size=2, replace=False)
            offspring_state = self.crossover(parent1.state_dict(), parent2.state_dict())
            offspring = self._create_scientist_copy(parent1).to(device)
            offspring.load_state_dict(offspring_state)
            new_population.append(offspring)

        for _ in range(num_mutate_crossover):
            parent1, parent2 = np.random.choice(parent_pool, size=2, replace=False)
            offspring_state = self.crossover(parent1.state_dict(), parent2.state_dict())
            offspring_state = self.mutate_state_dict(offspring_state, self.mutation_rate)
            offspring = self._create_scientist_copy(parent1).to(device)
            offspring.load_state_dict(offspring_state)
            new_population.append(offspring)


        return new_population


    def evaluate_scientists(self, molecules_per_scientist: List[torch.Tensor], fitness_scores: List[float]) -> List[float]:
        scientist_scores = []
        molecule_idx = 0
        for molecules in molecules_per_scientist:
            num_molecules = molecules.size(0)
            scores = fitness_scores[molecule_idx: molecule_idx + num_molecules]
            scientist_avg_score = sum(scores) / len(scores)
            scientist_scores.append(scientist_avg_score)
            molecule_idx += num_molecules
        return scientist_scores

    def rank_molecules(
        self,
        decoder: nn.Module,
        classifier: Any,
        molecules: torch.Tensor,
        desired_smiles: List[str],
        desired_effects: torch.Tensor,
        desired_classes: torch.Tensor,
        num_workers: int = 8,
    ) -> List[float]:
        smiles_results: List[Optional[str]] = [None] * len(molecules)
        pbar = tqdm(total=len(molecules), desc="Generating SMILES")
        count = 0
        with tqdm(total=len(molecules)) as pbar:
            # with max cpu works 50% was 6:39 mins, with 2 is was 5:28 mins, with 1 it was 3:09 mins
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Submit all tasks.
                futures = {
                    executor.submit(decoder.beam_search_decode, molecule, beam_width=3): i
                    for i, molecule in enumerate(molecules)
                }

                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    try:
                        smiles_results[idx] = future.result()
                    except Exception:
                        smiles_results[idx] = ""
                    count += 1
                    if count % (len(molecules) // 10) == 0:
                        pbar.update(len(molecules) // 10)
        valid_indices = [i for i, s in enumerate(smiles_results)]

        total_molecules = len(molecules)
        valid_count = len(valid_indices)
        percentage_valid = (valid_count / total_molecules) * 100 if total_molecules > 0 else 0
        avg_length = sum(len(smiles_results[i]) for i in valid_indices) / valid_count if valid_count > 0 else 0
        print(f"Valid molecules: {valid_count}/{total_molecules} ({percentage_valid:.2f}%), average SMILES length: {avg_length:.2f}")

        fitness_scores = [0.0] * total_molecules

        import difflib

        def string_similarity(str1, str2):
            matcher = difflib.SequenceMatcher(None, str1, str2)
            return matcher.ratio()

        if smiles_results:
            valid_mols = torch.stack([molecules[i] for i in valid_indices], dim=0).to(device)

            with torch.no_grad():
                predicted_effects_batch = classifier.effect_model(valid_mols)
                predicted_classes_batch = classifier.class_model(valid_mols)

            for j, idx in enumerate(valid_indices):
                try:
                    weighted_effect_similarity = (
                        (desired_effects[idx] * predicted_effects_batch[j]).sum() /
                        (desired_effects[idx].sum() + 1e-8)
                    ).item()
                    weighted_class_similarity = (
                        (desired_classes[idx] * predicted_classes_batch[j]).sum() /
                        (desired_classes[idx].sum() + 1e-8)
                    ).item()

                    canonical_generated = standardise_complex_smiles(smiles_results[idx])
                    canonical_desired = standardise_complex_smiles(desired_smiles[idx])
                    if canonical_generated is not None and canonical_desired is not None:
                        smile_similarity = string_similarity(canonical_generated, canonical_desired)
                    else:
                        smile_similarity = string_similarity(smiles_results[idx], desired_smiles[idx])

                    length_bonus = 0.0
                    if canonical_generated is not None and canonical_desired is not None:
                        if len(canonical_generated) >= len(canonical_desired):
                            length_bonus = LENGTH_BONUS
                    else:
                        if len(smiles_results[idx]) >= len(desired_smiles[idx]):
                            length_bonus = LENGTH_BONUS

                    penalty = 0.0
                    if not isValidSMILES(smiles_results[idx]):
                        penalty = INVALID_SMILES_PENALTY

                    fitness_scores[idx] = (
                        EFFECT_SIMILARITY_SIGNIFICANCE * (weighted_effect_similarity / EFFECTS_COUNT) +
                        CLASS_SIMILARITY_SIGNIFICANCE * (weighted_class_similarity / CLASSES_COUNT) +
                        SMILES_SIGNIFICANCE * smile_similarity +
                        length_bonus
                    ) - penalty

                except Exception as e:
                    print(f"Error processing molecule at index {idx} with SMILES: {smiles_results[idx]} - {e}")
                    fitness_scores[idx] = 0.0

        return fitness_scores


    def restore_scientists_from_checkpoint(
        self,
        checkpoint_path: str,
        num_scientists: int,
        latent_dim: int,
        effects_dim: int,
        output_dim: int,
    ) -> List[ConditionalTransformer]:

        scientists: List[ConditionalTransformer] = []
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if not isinstance(checkpoint, list):
            raise ValueError("Checkpoint format invalid—expected a list of state_dicts.")
        for state_dict in checkpoint:
            scientist = ConditionalTransformer(latent_dim, effects_dim, output_dim).to(device)
            scientist.load_state_dict(state_dict)
            scientist.eval()
            scientists.append(scientist)
        
        if len(scientists) > num_scientists:
            scientists = random.sample(scientists, num_scientists)
        while len(scientists) < num_scientists:
            scientists.append(random.choice(scientists))
        
        return scientists

    def initialize_scientists_from_pretrained(
        self,
        base_model_path: str,
        num_scientists: int,
        latent_dim: int,
        effects_dim: int,
        output_dim: int,
    ) -> Optional[List[ConditionalTransformer]]:
        try:
            base_model = ConditionalTransformer(latent_dim, effects_dim, output_dim).to(device)
            checkpoint = torch.load(base_model_path, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            base_model.load_state_dict(state_dict)
        except Exception as e:
            print("Error loading base model:", e)
            return None
        scientists = [ConditionalTransformer(latent_dim, effects_dim, output_dim).to(device) for _ in range(num_scientists)]
        for scientist in scientists:
            scientist.load_state_dict(base_model.state_dict())
        return scientists


    def train_model(
        self,
        scientists: List[ConditionalTransformer],
        autoencoder: nn.Module,
        decoder: nn.Module,
        classifier: Any,
        dataset: List[Any],
        smiles_list: List[str],
        rl_generations: int,
        num_generated_molecules: int,
        effects_dim: int,
        checkpoint_offset: Any,
        EFFECTS_COUNT: int,
        CLASSES_COUNT: int,
        path: str,
    ) -> None:

        dataset_tensor = torch.tensor(dataset, dtype=torch.float32).to(device)
        dataset_size = dataset_tensor.size(0)

        best_fitness_history: List[float] = []
        avg_fitness_history: List[float] = []

        for generation in tqdm(range(rl_generations), desc="RL Generations", unit="generation"):
            random_indices = torch.randperm(dataset_size, device=device)
            indices_per_scientist: List[torch.Tensor] = []
            desired_effects_per_scientist: List[torch.Tensor] = []
            for i in range(self.num_scientists):
                start_idx = i * num_generated_molecules
                end_idx = start_idx + num_generated_molecules
                selected_indices = random_indices[start_idx:end_idx]
                if selected_indices.size(0) < num_generated_molecules:
                    extra_needed = num_generated_molecules - selected_indices.size(0)
                    selected_indices = torch.cat([selected_indices, random_indices[:extra_needed]])
                indices_per_scientist.append(selected_indices)
                selected_effects = [dataset[idx.item()][:EFFECTS_COUNT] for idx in selected_indices]
                effects_tensor = torch.tensor(selected_effects, dtype=torch.float32).to(device)
                desired_effects_per_scientist.append(effects_tensor)

            molecules_per_scientist: List[torch.Tensor] = []
            class_labels_per_scientist: List[torch.Tensor] = []
            all_desired_smiles: List[str] = []

            print(f"Generation {generation + 1}")
            for i, scientist in enumerate(scientists):
                effects_vectors = desired_effects_per_scientist[i]
                selected_indices = indices_per_scientist[i]
                selected_class_labels = [dataset[idx.item()][EFFECTS_COUNT:EFFECTS_COUNT + CLASSES_COUNT]
                                        for idx in selected_indices]
                class_labels_tensor = torch.tensor(selected_class_labels, dtype=torch.float32).to(device)
                class_labels_per_scientist.append(class_labels_tensor)
                smiles_batch = [smiles_list[idx.item()] for idx in selected_indices]
                all_desired_smiles.extend(smiles_batch)
                input_vectors = dataset_tensor[selected_indices]
                latent_vectors = autoencoder.encoder(input_vectors)
                with torch.no_grad():
                    molecules = scientist(latent_vectors, effects_vectors)
                molecules_per_scientist.append(molecules)

            produced_molecules = torch.cat(molecules_per_scientist, dim=0)
            desired_effects_all = torch.cat(desired_effects_per_scientist, dim=0)
            desired_classes_all = torch.cat(class_labels_per_scientist, dim=0)

            assert produced_molecules.size(0) == desired_classes_all.size(0), "Mismatch between molecules and classes"
            assert len(all_desired_smiles) == produced_molecules.size(0), "Mismatch between molecules and SMILES"

            print("Ranking molecules...")
            fitness_scores = self.rank_molecules(decoder, classifier, produced_molecules,
                                                all_desired_smiles, desired_effects_all, desired_classes_all)
            print("Evaluating scientists...")
            scientist_scores = self.evaluate_scientists(molecules_per_scientist, fitness_scores)
            best_fitness = max(scientist_scores)
            avg_fitness = sum(scientist_scores) / len(scientist_scores)
            print(f"Generation {generation + 1}, Best Fitness: {best_fitness:.4f}, Avg Fitness: {avg_fitness:.4f}")

            best_fitness_history.append(best_fitness)
            avg_fitness_history.append(avg_fitness)

            scientists = self.update_population(scientist_scores, scientists)
            self.save_scientists(scientists, generation, path)
            print("Sample Generations:")
            for test in range(5):
                idx = random.randint(0, dataset_tensor.size(0) - 1)
                original_smile = smiles_list[idx]
                
                input_vector = dataset_tensor[idx]
                latent_vector = autoencoder.encoder(input_vector.unsqueeze(0))
                
                random_effects = torch.randint(0, 2, (1, effects_dim)).float().to(device)
                
                selected_scientist = random.choice(scientists)
                with torch.no_grad():
                    molecule_vector = selected_scientist(latent_vector, random_effects)
                
                generated_smiles = decoder.beam_search_decode(molecule_vector.squeeze(0), beam_width=3)
                
                print(f"Test {test+1}")
                print(f"OS: {original_smile}")
            
                print(f"GS: {generated_smiles}")

            # Save the the avg fitness and best fitness graph to a graph file
            plt.figure(figsize=(10, 5))
            plt.plot(best_fitness_history, label="Best Fitness", linewidth=2)
            plt.plot(avg_fitness_history, label="Average Fitness", linewidth=2)
            plt.title("Evolution of Fitness Over Generations")
            plt.xlabel("Generation")
            plt.ylabel("Fitness Score")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"loss_graph_gen_{generation + 1}.png")
            plt.close()
            # add best fitness and avg fitness to a csv file
            with open("fitness_history.csv", "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                if generation == 0:
                    writer.writerow(["Generation", "Best Fitness", "Avg Fitness"])
                writer.writerow([generation + 1, best_fitness, avg_fitness])

                

    def save_scientists(self, scientists: List[ConditionalTransformer], generation: int, path: str) -> None:
        print("Saving Scientists to", path)
        os.makedirs(path, exist_ok=True)
        checkpoint_path = os.path.join(path, f"gen_{generation + 1}.pth")
        if os.path.isdir(checkpoint_path):
            raise ValueError(f"Error: '{checkpoint_path}' is a directory, not a file.")
        torch.save([s.state_dict() for s in scientists], checkpoint_path)

    def _create_scientist_copy(self, scientist: ConditionalTransformer) -> ConditionalTransformer:

        bruh = ConditionalTransformer(
            latent_dim=scientist.latent_dim,
            effects_dim=scientist.effects_dim,
            output_dim=scientist.output_dim,
        )
        bruh.eval()
        return bruh

def extract_best_scientist(scientists_checkpoint, num_scientists, latent_dim, effects_dim, output_dim, num_generated_molecules,
                           save_path, effects_count, classes_count):
    dataset, smiles_list, _ = preprocess_dataset(DATASET)
    
    ga = GeneticAlgorithm(population_size=num_generated_molecules, num_scientists=num_scientists, mutation_rate=0.001)
    
    autoencoder = MolecularAutoencoder(input_size=AUTOENCODER_INPUT_SIZE, latent_dim=latent_dim).to(device)
    decoder = SMILESGenerator(
        vector_size=output_dim,
        hidden_size=SMILES_GENERATOR_HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        max_smiles_length=MAX_SMILES_LENGTH
    ).to(device)
    classifier = StableMoleculeClassifier(output_dim, classes_count, effects_count, LEARNING_RATE)
    
    scientists = ga.restore_scientists_from_checkpoint(scientists_checkpoint, num_scientists, latent_dim, effects_dim, output_dim)
    if not scientists:
        print("No scientists loaded!")
        return None
    
    # Load all models
    autoencoder.load_autoencoder(None, AUTOENCODER_PATH)
    decoder_checkpoint = torch.load(DECODER_PATH, map_location=device)
    if isinstance(decoder_checkpoint, dict) and "model_state_dict" in decoder_checkpoint:
        decoder.load_state_dict(decoder_checkpoint["model_state_dict"])
    else:
        decoder.load_state_dict(decoder_checkpoint)

    classifier.load_model(CLASSIFIER_PATH)


    autoencoder.eval()
    decoder.eval()
    classifier.class_model.eval()
    classifier.effect_model.eval()
    dataset_tensor = torch.tensor(dataset, dtype=torch.float32).to(device)
    total_samples = num_generated_molecules * num_scientists
    permuted_indices = torch.randint(0, dataset_tensor.size(0), (total_samples,), device=device)


    scientist_fitnesses = []
    
    for i, scientist in enumerate(scientists):
        start_idx = i * num_generated_molecules
        end_idx = start_idx + num_generated_molecules
        indices = permuted_indices[start_idx:end_idx]
        input_vectors = dataset_tensor[indices]
        latent_vectors = autoencoder.encoder(input_vectors)
        
        desired_effects = torch.bernoulli(torch.full((num_generated_molecules, effects_dim), 0.1)).float().to(device)
        
        class_labels = []
        for idx in indices:
            labels = dataset[idx.item()][effects_count: effects_count + classes_count]
            class_labels.append(labels)
        desired_classes = torch.tensor(class_labels, dtype=torch.float32).to(device)
        
        desired_smiles_list = [smiles_list[idx.item()] for idx in indices]
        
        with torch.no_grad():
            molecules = scientist(latent_vectors, desired_effects)
        
        fitness_scores = ga.rank_molecules(decoder, classifier, molecules, desired_smiles_list, desired_effects, desired_classes)
        
        if len(fitness_scores) > 0:
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
        else:
            avg_fitness = 0.0
        scientist_fitnesses.append(avg_fitness)
        print(f"Scientist {i} average fitness: {avg_fitness:.4f}")
    
    best_idx = np.argmax(scientist_fitnesses)
    best_scientist = scientists[best_idx]
    best_fitness = scientist_fitnesses[best_idx]
    print(f"Best scientist index: {best_idx} with average fitness: {best_fitness:.4f}")
    
    torch.save(best_scientist.state_dict(), save_path)
    print(f"Best scientist model saved to {save_path}")
    
    return best_scientist

# ---------------- Supervised Transformer ----------------
def train_pre_transformer() -> None:

    df = pd.read_csv(LATENT_PATH)
    latent_vectors = torch.tensor(list(df["LATENT_VECTOR"].apply(eval)), dtype=torch.float32)
    molecular_vectors = torch.tensor(list(df["VECTOR"].apply(eval)), dtype=torch.float32)
    molecule_effects = torch.tensor(list(df["EFFECTS"].apply(eval)), dtype=torch.float32)
    model = ConditionalTransformer(LATENT_DIM, EFFECTS_COUNT, MOLECULE_LENGTH)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    model.pre_train(
        model,
        optimizer,
        loss_fn,
        latent_vectors,
        molecular_vectors,
        molecule_effects,
        restore_epochs=RESTORE_PRETRAINED_TRANSFORMER_EPOCH,
        epochs=PRETRAINED_TRANSFORMER_EPOCHS,
        batch_size=BATCH_SIZE,
        path=PRETRAINED_TRANSFORMER_PATH,
    )


# ---------------- Reinforcement Transformer ----------------
def train_rl_transformer():

    dataset, smiles_list, molecule_list = preprocess_dataset(DATASET)
    ga = GeneticAlgorithm(population_size=NUM_GENERATED_MOLECULES, num_scientists=NUM_SCIENTISTS)

    if RESTORE_RL_TRANSFORMER_EPOCH:
        print("Restoring From Previously Trained Scientists ->", RESTORE_SCIENTISTS_PATH)
        scientists = ga.restore_scientists_from_checkpoint(
            RESTORE_SCIENTISTS_PATH, NUM_SCIENTISTS, LATENT_DIM, EFFECTS_COUNT, MOLECULE_LENGTH
        )
    else:
        print("Initializing Scientists from Pretrained Transformer")
        scientists = ga.initialize_scientists_from_pretrained(
            PRETRAINED_TRANSFORMER_PATH, NUM_SCIENTISTS, LATENT_DIM, EFFECTS_COUNT, MOLECULE_LENGTH
        )

    print("Training Scientists")

    autoencoder = MolecularAutoencoder(input_size=AUTOENCODER_INPUT_SIZE, latent_dim=LATENT_DIM).to(device)
    autoencoder_optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    autoencoder.load_autoencoder(autoencoder_optimizer, AUTOENCODER_PATH)
    autoencoder.eval()

    decoder = SMILESGenerator(
        vector_size=MOLECULE_LENGTH,
        hidden_size=SMILES_GENERATOR_HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        max_smiles_length=MAX_SMILES_LENGTH,
    ).to(device)
    checkpoint = torch.load(DECODER_PATH, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("Restoring Decoder from Checkpoint")
        decoder.load_state_dict(checkpoint["model_state_dict"])
    else:
        decoder = checkpoint
    decoder.eval()

    classifier_checkpoint = torch.load(CLASSIFIER_PATH, map_location=device)
    classifier = StableMoleculeClassifier(MOLECULE_LENGTH, CLASSES_COUNT, EFFECTS_COUNT, LEARNING_RATE)
    classifier.class_model.load_state_dict(classifier_checkpoint["class_model_state_dict"])
    classifier.effect_model.load_state_dict(classifier_checkpoint["effect_model_state_dict"])
   
    classifier.class_model.eval()
    classifier.effect_model.eval()
    classifier.class_model.to(device)
    classifier.effect_model.to(device)
    for scientist in scientists:
        scientist.eval()

    ga.train_model(
        scientists,
        autoencoder, 
        decoder, 
        classifier,
        dataset,
        smiles_list,
        RL_TRANSFORMER_EPOCHS,
        NUM_GENERATED_MOLECULES,
        EFFECTS_COUNT,
        SCIENTISTS_PATH,
        EFFECTS_COUNT,
        CLASSES_COUNT,
        SCIENTISTS_PATH
    )
