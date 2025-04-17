

from system.constants import *
import warnings
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch 
from transformers import AutoModel, AutoTokenizer

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import numpy as np
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False")


model = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
def smile_to_vector_ChemBERTa(smile):
    inputs = tokenizer(smile, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu()
    return vector.tolist()

NUM_EFFECTS = 41

effects = {
    0: "Stimulant",
    1: "Depressant",
    2: "Hallucinogenic",
    3: "Empathogenic/Entactogenic",
    4: "Dissociative",
    5: "Psychedelic",
    6: "Sedative",
    7: "Anxiolytic",
    8: "Analgesic/Pain Relief",
    9: "Antidepressant",
    10: "Nootropic/Cognitive Enhancement",
    11: "Anti-inflammatory",
    12: "Anti-anxiety",
    13: "Anti-psychotic",
    14: "Muscle Relaxant",
    15: "Euphoric",
    16: "Aphrodisiac",
    17: "Neuroprotective",
    18: "Sleep Aid/Hypnotic",
    19: "Mood Stabilizer",
    20: "Performance Enhancement",
    21: "Anti-convulsant",
    22: "Anti-addictive",
    23: "Immune Modulating",
    24: "Antiviral",
    25: "Antibacterial",
    26: "Anti-cancer",
    27: "Hormonal Regulation",
    28: "Bronchodilator",
    29: "Anti-diabetic",
    30: "Anti-hypertensive",
    31: "Vasodilator",
    32: "Psychoactive",
    33: "Anti-nausea",
    34: "Diuretic",
    35: "Anti-fungal",
    36: "Cannabinoid-like",
    37: "Serotonergic",
    38: "Dopaminergic",
    39: "Opioid-like",
    40: "Adrenergic"
}

def one_hot(selected_indices, total=NUM_EFFECTS):
    return [1 if i in selected_indices else 0 for i in range(total)]

combo1  = one_hot([0])    # [Stimulant]
combo2  = one_hot([6])    # [Sedative]
combo3  = one_hot([10])   # [Nootropic/Cognitive Enhancement]
combo4  = one_hot([9])    # [Antidepressant]
combo5  = one_hot([26])   # [Anti-cancer]
combo6  = one_hot([39])   # [Opioid-like]
combo7  = one_hot([0, 10])       # [Stimulant, Nootropic/Cognitive Enhancement]
combo8  = one_hot([1, 6])        # [Depressant, Sedative]
combo9  = one_hot([2, 5])        # [Hallucinogenic, Psychedelic]
combo10 = one_hot([8, 11])       # [Analgesic/Pain Relief, Anti-inflammatory]
combo11 = one_hot([7, 12])       # [Anxiolytic, Anti-anxiety]
combo12 = one_hot([17, 10])      # [Neuroprotective, Nootropic/Cognitive Enhancement]
combo13 = one_hot([36, 23])      # [Cannabinoid-like, Immune Modulating]
combo14 = one_hot([40, 20])      # [Adrenergic, Performance Enhancement]
combo15 = one_hot([0, 10, 20])       # [Stimulant, Nootropic/Cognitive Enhancement, Performance Enhancement]
combo16 = one_hot([1, 6, 18])         # [Depressant, Sedative, Sleep Aid/Hypnotic]
combo17 = one_hot([2, 3, 5])          # [Hallucinogenic, Empathogenic/Entactogenic, Psychedelic]
combo18 = one_hot([8, 11, 14])        # [Analgesic/Pain Relief, Anti-inflammatory, Muscle Relaxant]
combo19 = one_hot([9, 12, 19])        # [Antidepressant, Anti-anxiety, Mood Stabilizer]
combo20 = one_hot([39, 8, 6])         # [Opioid-like, Analgesic/Pain Relief, Sedative]
combo21 = one_hot([17, 10, 7])        # [Neuroprotective, Nootropic/Cognitive Enhancement, Anxiolytic]
combo22 = one_hot([36, 23, 11])       # [Cannabinoid-like, Immune Modulating, Anti-inflammatory]
combo23 = one_hot([0, 10, 20, 15])    # [Stimulant, Nootropic/Cognitive Enhancement, Performance Enhancement, Euphoric]
combo24 = one_hot([1, 6, 18, 12])     # [Depressant, Sedative, Sleep Aid/Hypnotic, Anti-anxiety]
combo25 = one_hot([2, 3, 5, 32])      # [Hallucinogenic, Empathogenic/Entactogenic, Psychedelic, Psychoactive]
combo26 = one_hot([8, 11, 14, 7])     # [Analgesic/Pain Relief, Anti-inflammatory, Muscle Relaxant, Anxiolytic]
combo27 = one_hot([9, 12, 19, 17])    # [Antidepressant, Anti-anxiety, Mood Stabilizer, Neuroprotective]
combo28 = one_hot([36, 23, 11, 26])   # [Cannabinoid-like, Immune Modulating, Anti-inflammatory, Anti-cancer]
combo29 = one_hot([0, 10, 20, 15, 40])      # [Stimulant, Nootropic/Cognitive Enhancement, Performance Enhancement, Euphoric, Adrenergic]
combo30 = one_hot([1, 6, 18, 12, 14])         # [Depressant, Sedative, Sleep Aid/Hypnotic, Anti-anxiety, Muscle Relaxant]
combo31 = one_hot([0, 10, 20, 15, 40, 37])     # [Stimulant, Nootropic/Cognitive Enhancement, Performance Enhancement, Euphoric, Adrenergic, Serotonergic]
combo32 = one_hot([2, 3, 5, 32, 7, 9])         # [Hallucinogenic, Empathogenic/Entactogenic, Psychedelic, Psychoactive, Anxiolytic, Antidepressant]
combo33 = one_hot([4])    # [Dissociative]
combo34 = one_hot([13])   # [Anti-psychotic]
combo35 = one_hot([16])   # [Aphrodisiac]
combo36 = one_hot([21])   # [Anti-convulsant]
combo37 = one_hot([22])   # [Anti-addictive]
combo38 = one_hot([24])   # [Antiviral]
combo39 = one_hot([25])   # [Antibacterial]
combo40 = one_hot([27])   # [Hormonal Regulation]
combo41 = one_hot([28])   # [Bronchodilator]
combo42 = one_hot([29])   # [Anti-diabetic]
combo43 = one_hot([30])   # [Anti-hypertensive]
combo44 = one_hot([31])   # [Vasodilator]
combo45 = one_hot([33])   # [Anti-nausea]
combo46 = one_hot([34])   # [Diuretic]
combo47 = one_hot([35])   # [Anti-fungal]
combo48 = one_hot([38])   # [Dopaminergic]

target_effect_combinations = [
    combo1,  combo2,  combo3,  combo4,  combo5,  combo6,
    combo7,  combo8,  combo9,  combo10, combo11, combo12, combo13, combo14,
    combo15, combo16, combo17, combo18, combo19, combo20, combo21, combo22,
    combo23, combo24, combo25, combo26, combo27, combo28,
    combo29, combo30, combo31, combo32,
    combo33, combo34, combo35, combo36, combo37, combo38, combo39, combo40,
    combo41, combo42, combo43, combo44, combo45, combo46, combo47, combo48
]

def print_effect_combination(combo_vector):
    active = [effects[i] for i, flag in enumerate(combo_vector) if flag]
    print(active)



import torch
from system.constants import *
import os
import torch
import torch.nn as nn
from architectures.autoencoder import *
from architectures.classifier import *
from architectures.decoder import *
from architectures.transformer import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

autoencoder = MolecularAutoencoder(input_size=AUTOENCODER_INPUT_SIZE, latent_dim=LATENT_DIM).to(device)
decoder = SMILESGenerator(
        vector_size=MOLECULE_LENGTH,
        hidden_size=SMILES_GENERATOR_HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        max_smiles_length=MAX_SMILES_LENGTH
    ).to(device)
transformer = ConditionalTransformer(LATENT_DIM, EFFECTS_COUNT, MOLECULE_LENGTH)
classifier = StableClassifier(MOLECULE_LENGTH, CLASSES_COUNT, EFFECTS_COUNT, LEARNING_RATE)



def load_all_models():
    global autoencoder, decoder, transformer, classifier
    load_model(autoencoder, AUTOENCODER_PATH)
    load_model(decoder, DECODER_PATH)
    checkpoint = torch.load(BEST_SCIENTIST_PATH, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    transformer.load_state_dict(state_dict)
    transformer.eval()
    print("Loaded model from: ", BEST_SCIENTIST_PATH)

    classifier_checkpoint = torch.load(
        CLASSIFIER_PATH,
        map_location=torch.device("cpu"),
        weights_only=False
    )
    print("Loaded model from: ", CLASSIFIER_PATH)

    classifier.class_model.load_state_dict(classifier_checkpoint['class_model_state_dict'])
    classifier.effect_model.load_state_dict(classifier_checkpoint['effect_model_state_dict'])
    classifier.class_model.eval()
    classifier.effect_model.eval()

def load_model(model, checkpoint_path):
    if os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model from: {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    return model


load_all_models()

class Pipeline:
    def __init__(self):

        self.autoencoder = autoencoder
        self.transformer = transformer
        self.decoder = decoder
        self.classifier = classifier 

    def generate(self, starting_smile, target_effect_vector, noise_scale=0.01, beam_width=10, temperature=0.8,noise_on=False):

        starting_vector_list = smile_to_vector_ChemBERTa( starting_smile)
        starting_vector = torch.tensor(starting_vector_list, dtype=torch.float32)

        effects_vector = self.classifier.effect_model(starting_vector.unsqueeze(0)).squeeze(0)
        classes_vector = self.classifier.class_model(starting_vector.unsqueeze(0)).squeeze(0)
        
        combined_input = torch.cat((effects_vector, classes_vector, starting_vector), dim=0).unsqueeze(0)
        
        latent_vector = self.autoencoder.encoder(combined_input)
        
        noisy_latent = latent_vector + (torch.randn_like(latent_vector) * noise_scale * (1 if noise_on else 0))
        
        target_effect_tensor = torch.tensor(target_effect_vector, dtype=torch.float32).unsqueeze(0)
        conditioned_latent = self.transformer(noisy_latent, target_effect_tensor)
        
        generated_smile = self.decoder.beam_search_decode(
            conditioned_latent[0],
            beam_width=beam_width,
            temperature=temperature
        )
        
        return generated_smile
    def generate_all(self, starting_molecules, target_effect_combos, molecules_per_pair=20):
        results = []
        total_pairs = len(starting_molecules) * len(target_effect_combos)
        total_attempts = len(starting_molecules) * len(target_effect_combos) * molecules_per_pair

        print("Total pairs", total_pairs)
        print("Starting molecules", len(starting_molecules))
        print("Target effect combos", len(target_effect_combos))
        print("Total molecules to generate", total_attempts)
        with tqdm(total=total_attempts, desc="Generating molecules", unit="molecule") as pbar:
            for sm in starting_molecules:
                for effect in target_effect_combos:
                    generated_list = []
                    for _ in range(molecules_per_pair):
                        try:
                            generated_smile = self.generate(sm, effect)
                            if isValidSMILES(generated_smile):
                                generated_list.append(generated_smile)
                        except Exception as gen_err:
                            print(f"Generation error for {sm} with effect {effect}: {gen_err}")
                        pbar.update(1)
                    results.append({
                        "starting_smile": sm,
                        "target_effect": effect,
                        "generated_molecules": generated_list
                    })
        return {"results": results}


def generate_smiles_list(num_samples=100):
    import pandas as pd
    from system.constants import DATASET 

    try:
        df = pd.read_csv(DATASET)
        
        valid_df = df[~df["SMILES"].str.contains(r"\.", regex=True)]
        
        if len(valid_df) < num_samples:
            print(f"Warning: Only {len(valid_df)} valid SMILES without a dot found; returning all.")
            sample_df = valid_df
        else:
            sample_df = valid_df.sample(n=num_samples, random_state=42)
        
        return sample_df["SMILES"].tolist()
    except Exception as e:
        print(f"Error generating SMILES list: {e}")
        return []
        

class Analyzer:
    def __init__(self, classifier):

        self.classifier = classifier

    def analyze_results(self, results_json):
        updated_results = []
        for group in tqdm(results_json.get("results", []), desc="Processing groups", unit="group"):
            starting_smile = group.get("starting_smile")
            target_effect = group.get("target_effect")
            updated_molecules = []
            
            for smile in tqdm(group.get("generated_molecules", []), desc="Processing molecules", unit="molecule", leave=False):
                try:
                    vec_list = smile_to_vector_ChemBERTa(smile)
                    vec_tensor = torch.tensor(vec_list, dtype=torch.float32).unsqueeze(0).to(device)
                    
                    pred_classes = self.classifier.predict_class_probabilities(
                        vec_tensor, list(classes.values()), self.classifier.class_model
                    )
                    pred_effects = self.classifier.predict_class_probabilities(
                        vec_tensor, list(effects.values()), self.classifier.effect_model
                    )
                except Exception as e:
                    print(f"Error processing SMILES {smile}: {e}")
                    pred_classes = {}
                    pred_effects = {}
                
                updated_molecules.append({
                    "smiles": smile,
                    "predicted_classes": pred_classes,
                    "predicted_effects": pred_effects
                })
            
            updated_results.append({
                "starting_smile": starting_smile,
                "target_effect": target_effect,
                "generated_molecules": updated_molecules
            })
        
        return {"results": updated_results}

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
class Visualiser:
    def __init__(self, results):

        self.results = results

    def plot_predicted_class_occurrences(
        self,
        threshold=2.0,
        exclude_top=0,
        save_fig=False,
        filename="predicted_class_occurrences.png",
        max_label_length=15
    ):

        class_counts = {}
        total_molecules = 0

        for group in self.results.get("results", []):
            generated_molecules = group.get("generated_molecules", [])
            for mol in generated_molecules:
                total_molecules += 1
                pred_classes = mol.get("predicted_classes", {})
                for class_name, prob in pred_classes.items():
                    if prob >= threshold:
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1

        if total_molecules == 0:
            print("No generated molecules found in the results.")
            return

        if not class_counts:
            print(f"No classes found with predicted probability >= {threshold}%.")
            return

        percentages_dict = {
            cn: (count / total_molecules) * 100.0
            for cn, count in class_counts.items()
        }

        sorted_items = sorted(percentages_dict.items(), key=lambda x: x[1], reverse=True)

        filtered_items = [(cn, pct) for cn, pct in sorted_items if pct != 0.0]
        filtered_items = filtered_items[exclude_top:]

        if not filtered_items:
            print(f"No classes remain after excluding top {exclude_top} and removing 0% items.")
            return

        class_names = []
        percentages = []
        for cn, pct in filtered_items:
            truncated_cn = cn[:max_label_length - 3] + '...' if len(cn) > max_label_length else cn
            class_names.append(truncated_cn)
            percentages.append(pct)

        plt.figure(figsize=(8, max(6, len(class_names) * 0.35)))
        plt.barh(class_names, percentages, color="orange", alpha=0.7)
        plt.xlabel(f"Percentage of All Generated Molecules (%)\n(predicted >= {threshold}%)")
        plt.title(f"Predicted Class Occurrences (Excluding Top {exclude_top})")

        plt.gca().invert_yaxis()
        plt.grid(True, axis="x", alpha=0.3)

        if save_fig:
            plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.show()

    def plot_predicted_effect_occurrences(
        self,
        threshold=2.0,
        exclude_top=0,
        save_fig=False,
        filename="predicted_effect_occurrences.png",
        max_label_length=15
    ):
        effect_counts = {}
        total_molecules = 0

        for group in self.results.get("results", []):
            generated_molecules = group.get("generated_molecules", [])
            for mol in generated_molecules:
                total_molecules += 1
                pred_effects = mol.get("predicted_effects", {})
                for effect_name, pct in pred_effects.items():
                    if pct >= threshold:
                        effect_counts[effect_name] = effect_counts.get(effect_name, 0) + 1

        if total_molecules == 0:
            print("No generated molecules found in the results.")
            return

        if not effect_counts:
            print(f"No effects found with predicted value >= {threshold}%.")
            return

        percentages_dict = {
            en: (count / total_molecules) * 100.0
            for en, count in effect_counts.items()
        }

        sorted_items = sorted(percentages_dict.items(), key=lambda x: x[1], reverse=True)

        filtered_items = [(en, pct) for en, pct in sorted_items if pct != 0.0]
        filtered_items = filtered_items[exclude_top:]
        
        if not filtered_items:
            print(f"No effects left after excluding top {exclude_top} and removing 0% items.")
            return

        effect_names = []
        percentages = []
        for en, pct in filtered_items:
            truncated = en[:max_label_length - 3] + '...' if len(en) > max_label_length else en
            effect_names.append(truncated)
            percentages.append(pct)

        plt.figure(figsize=(8, max(6, len(effect_names) * 0.35)))
        plt.barh(effect_names, percentages, color="cyan", alpha=0.7)
        plt.xlabel(f"Percentage of All Generated Molecules (%)\n(predicted >= {threshold}%)")
        plt.title(f"Predicted Effect Occurrences (Excluding Top {exclude_top})")

        plt.gca().invert_yaxis()
        plt.grid(True, axis="x", alpha=0.3)

        if save_fig:
            plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.show()



    def plot_accuracy_histogram_for_group(self, group, num_bins=20, save_fig=False, filename=None,
                                            filter_low=False, min_threshold=1.0):
        target_effect_vector = group.get("target_effect", [])
        active_indices = [i for i, flag in enumerate(target_effect_vector) if flag == 1]
        if not active_indices:
            print("No active target effects found for this group.")
            return

        active_effect_names = [effects[i] for i in active_indices if i in effects]

        accuracy_list = []
        for mol in group.get("generated_molecules", []):
            pred_effects = mol.get("predicted_effects", {})
            values = []
            for idx in active_indices:
                effect_name = effects.get(idx)
                if effect_name in pred_effects:
                    values.append(max(0, pred_effects[effect_name]))
            avg_accuracy = sum(values) / len(values) if values else 0.0
            accuracy_list.append(avg_accuracy)

        if not accuracy_list:
            print("No accuracy values computed for this group.")
            return

        if filter_low:
            accuracy_list = [acc for acc in accuracy_list if acc >= min_threshold]
            if not accuracy_list:
                print("No accuracy values above the minimum threshold for this group.")
                return

        max_accuracy = 100
        bins = np.linspace(0, max_accuracy, num_bins + 1)
        counts, _ = np.histogram(accuracy_list, bins=bins)
        percentages = (counts / len(accuracy_list)) * 100

        plt.figure(figsize=(8, 5))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.bar(bin_centers, percentages, width=(bins[1]-bins[0]), color="blue", alpha=0.7)
        plt.xlabel("Average Predicted Accuracy (%)")
        plt.ylabel("Percentage of Molecules (%)")
        plt.title("Accuracy Distribution for Active Effects:\n(" + ", ".join(active_effect_names) + ")")
        plt.grid(True)
        plt.xlim(0, max_accuracy)
        if save_fig and filename:
            plt.savefig(filename)
        plt.show()

    def plot_overall_accuracy_histogram(self, num_bins=20, save_fig=False, filename="overall_accuracy_histogram.png",
                                          filter_low=False, min_threshold=1.0):
        overall_accuracies = []
        for group in self.results.get("results", []):
            target_effect_vector = group.get("target_effect", [])
            active_indices = [i for i, flag in enumerate(target_effect_vector) if flag == 1]
            if not active_indices:
                continue
            for mol in group.get("generated_molecules", []):
                pred_effects = mol.get("predicted_effects", {})
                values = []
                for idx in active_indices:
                    effect_name = effects.get(idx)
                    if effect_name in pred_effects:
                        values.append(max(0, pred_effects[effect_name]))
                avg_accuracy = sum(values) / len(values) if values else 0.0
                overall_accuracies.append(avg_accuracy)

        if not overall_accuracies:
            print("No overall accuracy values computed.")
            return

        if filter_low:
            overall_accuracies = [acc for acc in overall_accuracies if acc >= min_threshold]
            if not overall_accuracies:
                print("No overall accuracy values above the minimum threshold.")
                return

        max_accuracy = 100
        bins = np.linspace(0, max_accuracy, num_bins + 1)
        counts, _ = np.histogram(overall_accuracies, bins=bins)
        percentages = (counts / len(overall_accuracies)) * 100

        plt.figure(figsize=(12, 8))
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.bar(bin_centers, percentages, width=(bins[1]-bins[0]), color="green", alpha=0.7)
        plt.xlabel("Average Predicted Accuracy (%)")
        plt.ylabel("Percentage of All Molecules (%)")
        plt.title("Overall Accuracy Distribution Across All Target Effects")
        plt.grid(True)
        plt.xlim(0, max_accuracy)
        if save_fig:
            plt.savefig(filename)
        plt.xticks(np.arange(0, max_accuracy + 1, step=10)) 

        plt.show()

    def plot_histograms_by_target_effect(self, num_bins=20, save_fig=False, filename_pattern="accuracy_histogram_target_{target}.png",
                                         filter_low=False, min_threshold=1.0):
        grouped = {}
        for group in self.results.get("results", []):
            target_key = tuple(group.get("target_effect", []))
            if target_key not in grouped:
                grouped[target_key] = {
                    "target_effect": group.get("target_effect", []),
                    "generated_molecules": []
                }
            grouped[target_key]["generated_molecules"].extend(group.get("generated_molecules", []))
        
        for target_key, group_data in grouped.items():
            active_indices = [i for i, flag in enumerate(group_data["target_effect"]) if flag == 1]
            active_names = [effects[i] for i in active_indices if i in effects]
            accuracy_list = []
            for mol in group_data["generated_molecules"]:
                pred_effects = mol.get("predicted_effects", {})
                values = []
                for idx in active_indices:
                    effect_name = effects.get(idx)
                    if effect_name in pred_effects:
                        values.append(max(0, pred_effects[effect_name]))
                avg_accuracy = sum(values) / len(values) if values else 0.0
                accuracy_list.append(avg_accuracy)
            if not accuracy_list:
                print(f"No accuracy values for target effects: {', '.join(active_names)}")
                continue

            if filter_low:
                accuracy_list = [acc for acc in accuracy_list if acc >= min_threshold]
                if not accuracy_list:
                    print(f"No accuracy values above threshold for target effects: {', '.join(active_names)}")
                    continue

            max_accuracy = 100
            bins = np.linspace(0, max_accuracy, num_bins + 1)
            counts, _ = np.histogram(accuracy_list, bins=bins)
            percentages = (counts / len(accuracy_list)) * 100

            plt.figure(figsize=(8, 5))
            bin_centers = (bins[:-1] + bins[1:]) / 2
            plt.bar(bin_centers, percentages, width=(bins[1]-bins[0]), color="blue", alpha=0.7)
            plt.xlabel("Average Predicted Accuracy (%)")
            plt.ylabel("Percentage of Molecules (%)")
            plt.title("Combined Accuracy Distribution for Target Effects:\n(" + ", ".join(active_names) + ")")
            plt.grid(True)
            plt.xlim(0, max_accuracy)
            if save_fig:
                target_label = "_".join(str(x) for x in group_data["target_effect"])
                fname = filename_pattern.format(target=target_label)
                plt.savefig(fname)
            plt.show()

    def plot_all_histograms(self, save_fig=False, filter_low=False, min_threshold=1.0):
        # self.plot_histograms_by_target_effect(num_bins=120, save_fig=save_fig, filter_low=filter_low, min_threshold=min_threshold)
        self.plot_overall_accuracy_histogram(num_bins=120, save_fig=save_fig, filter_low=False, min_threshold=min_threshold)
    def plot_novelty_distribution_with_subsampling(
        self,
        radius=2,
        nbits=2048,
        method="pairwise",
        save_fig=False,
        filename="novelty_distribution.png",
        max_samples=2000
    ):

        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        import random

        all_smiles = []
        for group in self.results.get("results", []):
            generated = group.get("generated_molecules", [])
            for moldata in generated:
                smi = moldata.get("smiles", "")
                if smi:
                    all_smiles.append(smi)

        all_smiles = list(set(all_smiles))
        n_total = len(all_smiles)

        if n_total == 0:
            print("No SMILES strings found in the results.")
            return
        elif n_total == 1:
            print("Only one valid molecule found. Can't plot novelty for a single molecule.")
            return

        if n_total > max_samples:
            print(f"Too many molecules ({n_total}). Randomly sampling {max_samples} of them.")
            all_smiles = random.sample(all_smiles, k=max_samples)
            n_sample = max_samples
        else:
            n_sample = n_total

        print(f"Using {n_sample} molecules for novelty computation.")

        from tqdm import tqdm
        mols = []
        fps = []
        print("Converting SMILES to RDKit Mol objects and computing fingerprints...")
        for smi in tqdm(all_smiles, desc="SMILES Conversion", unit="molecule"):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
                fps.append(fp)

        n = len(fps)
        if n < 2:
            print("Fewer than 2 valid RDKit molecules. Cannot compute pairwise similarities.")
            return

        if method == "pairwise":
            sims = []
            total_comps = (n * (n - 1)) // 2
            print(f"Computing pairwise Tanimoto similarities among {n} molecules...")
            with tqdm(total=total_comps, desc="Pairwise Similarities", unit="pairs") as pbar:
                for i in range(n):
                    for j in range(i + 1, n):
                        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        sims.append(sim)
                        pbar.update(1)

            if not sims:
                print("No pairwise similarities computed.")
                return

            plt.figure(figsize=(8, 6))
            plt.hist(sims, bins=50, color="green", alpha=0.7)
            plt.xlabel("Pairwise Tanimoto Similarity")
            plt.ylabel("Count")
            plt.title("Distribution of Pairwise Similarities (Subsampled)")
            plt.grid(True, alpha=0.3)

            if save_fig:
                plt.savefig(filename, dpi=200, bbox_inches="tight")
            plt.show()

        elif method == "average":
            avg_sims = []
            print(f"Computing average Tanimoto similarities among {n} molecules...")
            with tqdm(total=n, desc="Averaging Similarities", unit="mol") as pbar:
                for i in range(n):
                    sum_sim = 0.0
                    for j in range(n):
                        if i != j:
                            sum_sim += DataStructs.TanimotoSimilarity(fps[i], fps[j])
                    avg_sims.append(sum_sim / (n - 1))
                    pbar.update(1)

            plt.figure(figsize=(8, 6))
            plt.hist(avg_sims, bins=50, color="blue", alpha=0.7)
            plt.xlabel("Average Tanimoto Similarity to Others")
            plt.ylabel("Count of Molecules")
            plt.title("Distribution of Average Similarities (Subsampled)")
            plt.grid(True, alpha=0.3)

            if save_fig:
                plt.savefig(filename, dpi=200, bbox_inches="tight")
            plt.show()

        else:
            print(f"Unknown method '{method}'. Use 'pairwise' or 'average'.")
            return
    def plot_novelty_against_seed(
        self,
        seed_smiles,
        radius=2,
        nbits=2048,
        save_fig=False,
        filename="novelty_vs_seed_distribution.png"
    ):

        import random
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        import matplotlib.pyplot as plt
        from tqdm import tqdm

        seed_mols = []
        seed_fps = []
        for smi in seed_smiles:
            m = Chem.MolFromSmiles(smi)
            if m:
                seed_mols.append(m)
                seed_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits))
        n_seed = len(seed_fps)
        if n_seed == 0:
            print("No valid seed molecules. Cannot compute novelty vs. seed.")
            return

        gen_smiles = []
        for group in self.results.get("results", []):
            for moldata in group.get("generated_molecules", []):
                smi = moldata.get("smiles", "")
                if smi:
                    gen_smiles.append(smi)
        gen_smiles = list(set(gen_smiles))
        n_gen = len(gen_smiles)
        if n_gen == 0:
            print("No generated SMILES found.")
            return

        gen_fps = []
        valid_gen_smiles = []
        print("Computing fingerprints for generated molecules...")
        for smi in tqdm(gen_smiles, desc="FP for Gens", unit="molecule"):
            m = Chem.MolFromSmiles(smi)
            if m:
                gen_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits))
                valid_gen_smiles.append(smi)

        n_valid_gen = len(gen_fps)
        if n_valid_gen == 0:
            print("None of the generated SMILES were valid RDKit molecules.")
            return
        max_sims = []
        print("Calculating novelty relative to seed molecules (max similarity)...")
        for fp_gen in tqdm(gen_fps, desc="Novelty vs. seed", unit="molecule"):
            sims = DataStructs.BulkTanimotoSimilarity(fp_gen, seed_fps)
            max_sim = max(sims) if sims else 0.0
            max_sims.append(max_sim)

        plt.figure(figsize=(8,6))
        plt.hist(max_sims, bins=50, color="purple", alpha=0.7)
        plt.xlabel("Maximum Tanimoto Similarity to Seed Set")
        plt.ylabel("Count of Generated Molecules")
        plt.title("Novelty Relative to Seeds (Lower = More Novel)")
        plt.grid(True, alpha=0.3)

        if save_fig:
            plt.savefig(filename, dpi=200, bbox_inches="tight")
        plt.show()

        import numpy as np
        max_sims_np = np.array(max_sims)
        print(f"Number of valid generated molecules: {n_valid_gen}")
        print(f"Mean max similarity:   {max_sims_np.mean():.3f}")
        print(f"Median max similarity: {np.median(max_sims_np):.3f}")

        print(f"Range of max similarity: ({max_sims_np.min():.3f}, {max_sims_np.max():.3f})")

    def plot_groupwise_novelty_across_starting_smiles(
        self,
        radius=2,
        nbits=2048,
        method="average",
        save_fig=False,
        filename="groupwise_novelty_boxplot.png",
        limit_most_novel=20,
        limit_least_novel=24
    ):
        import matplotlib.pyplot as plt
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        from tqdm import tqdm

        group_map = {}
        for group_data in self.results.get("results", []):
            s_smile = group_data.get("starting_smile", "").strip()
            if not s_smile:
                continue
            if s_smile not in group_map:
                group_map[s_smile] = []
            for mol_item in group_data.get("generated_molecules", []):
                gm = mol_item.get("smiles", "").strip()
                if gm:
                    group_map[s_smile].append(gm)

        group_fps = {}
        for g_smile, generated_list in group_map.items():
            fps_list = []
            for smi in generated_list:
                m = Chem.MolFromSmiles(smi)
                if m:
                    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits)
                    fps_list.append(fp)
            group_fps[g_smile] = fps_list

        all_groups = list(group_fps.keys())
        print(f"Found {len(all_groups)} distinct groups (starting SMILES).")

        group_summaries = {}
        for g in tqdm(all_groups, desc="Groupwise Novelty", unit="group"):
            fps_g = group_fps[g]
            if not fps_g:
                print(f"Skipping group {g} (no valid molecules).")
                continue

            fps_others = []
            for h in all_groups:
                if h != g:
                    fps_others.extend(group_fps[h])

            if not fps_others:
                print(f"Skipping group {g} (no other molecules).")
                continue

            summary_list = []
            for fp_m in fps_g:
                sims = DataStructs.BulkTanimotoSimilarity(fp_m, fps_others)
                if not sims:
                    summary_val = 0.0
                else:
                    if method == "average":
                        summary_val = float(np.mean(sims))
                    elif method == "min":
                        summary_val = float(np.min(sims))
                    elif method == "max":
                        summary_val = float(np.max(sims))
                    else:
                        summary_val = float(np.mean(sims))
                summary_list.append(summary_val)

            group_summaries[g] = summary_list

        group_summaries = {g: arr for g, arr in group_summaries.items() if arr}
        if not group_summaries:
            print("No valid group summaries for novelty.")
            return

        group_means = {}
        for g, arr in group_summaries.items():
            group_means[g] = float(np.mean(arr))

        sorted_means = sorted(group_means.items(), key=lambda x: x[1])

        most_novel = sorted_means[:limit_most_novel]
        least_novel = sorted_means[-limit_least_novel:] if limit_least_novel > 0 else []

        def plot_and_print_stats(group_subset, subset_name="most_novel", base_filename=filename):
            """
            group_subset is a list of (group_name, mean_value) for each group, 
            sorted in ascending or descending order as appropriate.
            """
            if not group_subset:
                print(f"No groups in subset: {subset_name}")
                return

            selected_labels = []
            boxplot_data = []
            import numpy as np

            for (g, _) in group_subset:
                arr = group_summaries[g]
                if not arr:
                    continue
                short_label = g[:8] + "..." if len(g) > 8 else g
                selected_labels.append(short_label)
                boxplot_data.append(arr)

            if not boxplot_data:
                print(f"No data found for {subset_name} subset.")
                return

            plt.figure(figsize=(max(10, len(boxplot_data) * 0.5), 6))
            plt.boxplot(boxplot_data, vert=True, labels=selected_labels, patch_artist=True)
            plt.ylabel(f"Tanimoto to 'Other' Groups (method={method})")
            plt.xlabel(f"Groups ({subset_name})")
            plt.title(f"Groupwise Novelty Boxplot: {subset_name}")
            plt.xticks(rotation=90)
            plt.grid(True, alpha=0.3)
            if save_fig:
                out_fn = base_filename.replace(".png", f"_{subset_name}_box.png")
                plt.savefig(out_fn, dpi=200, bbox_inches="tight")
            plt.tight_layout()
            plt.show()

            bar_values = [group_means[g] for (g, _) in group_subset]
            plt.figure(figsize=(max(10, len(bar_values) * 0.5), 5))
            plt.bar(selected_labels, bar_values, color="darkblue", alpha=0.7)
            plt.ylabel(f"Mean Tanimoto (method={method})")
            plt.xlabel(f"Groups ({subset_name})")
            plt.title(f"Groupwise Novelty (Mean Similarity) - {subset_name}")
            plt.xticks(rotation=90)
            plt.grid(True, alpha=0.3)
            if save_fig:
                bar_fn = base_filename.replace(".png", f"_{subset_name}_bar.png")
                plt.savefig(bar_fn, dpi=200, bbox_inches="tight")
            plt.tight_layout()
            plt.show()

            all_values = []
            for (g, _) in group_subset:
                arr = group_summaries[g]
                if not arr:
                    continue
                arr_min = float(min(arr))
                arr_max = float(max(arr))
                arr_mean = float(np.mean(arr))
                Q1, Q3 = np.percentile(arr, [25, 75])
                iqr_val = Q3 - Q1
                arr_truncated = arr[:10]  # only show first 10

                print(f"\n[{subset_name}] Group '{g}' => Tanimoto array (truncated to 10): {arr_truncated}")
                print(f" Full length: {len(arr)} items")
                print(f" Range = {arr_min:.3f} to {arr_max:.3f}")
                print(f" Mean = {arr_mean:.3f}")
                print(f" IQR = {iqr_val:.3f}")

                all_values.extend(arr)

            if all_values:
                overall_mean = float(np.mean(all_values))
                overall_min = float(np.min(all_values))
                overall_max = float(np.max(all_values))
                Q1, Q3 = np.percentile(all_values, [25, 75])
                overall_iqr = Q3 - Q1
                print(f"\n[{subset_name}] => Overall range: {overall_min:.3f} to {overall_max:.3f}")
                print(f"             Overall mean: {overall_mean:.3f}")
                print(f"             Overall IQR:  {overall_iqr:.3f}")
            else:
                print(f"\nNo data in {subset_name} subset to compute overall stats.")

        print(f"\nNow plotting the {limit_most_novel} most novel groups (lowest mean similarity) ...")
        plot_and_print_stats(most_novel, subset_name="most_novel", base_filename=filename)

        if least_novel:
            print(f"\nNow plotting the {limit_least_novel} least novel groups (highest mean similarity) ...")
            least_novel = list(reversed(least_novel))  # so it goes from highest to lower
            plot_and_print_stats(least_novel, subset_name="least_novel", base_filename=filename)
        else:
            print("\nNo 'least novel' subset requested or found.")




import json

def extract_seed_smiles_from_updated_json(filepath="updated_generated_molecules.json"):
   
    with open(filepath, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    seed_smiles = set()

    for entry in results:
        sm = entry.get("starting_smile", "").strip()
        if sm:
            seed_smiles.add(sm)

    return list(seed_smiles)



def generate():
    pipeline = Pipeline()

    starting_molecules = generate_smiles_list(240)
    target_effect_combos = target_effect_combinations

    all_results = pipeline.generate_all(starting_molecules, target_effect_combos, molecules_per_pair=5)

    output_json = json.dumps(all_results, indent=4)
    print(output_json)
    with open("generated_molecules_results.json", "w") as f:
        f.write(output_json)


def analyse():
    json_filename = "generated_molecules_results.json"
    try:
        with open(json_filename, "r") as f:
            generated_results = json.load(f)
        print(f"Loaded generated results from '{json_filename}' successfully.")
    except Exception as e:
        print(f"Error loading generated results from '{json_filename}': {e}")
        return

    analyzer = Analyzer(classifier)

    analysed_results = analyzer.analyze_results(generated_results)

    with open("updated_generated_molecules.json", "w") as f:
        json.dump(analysed_results, f, indent=4)
    
    print("Analysis complete and saved to 'updated_generated_molecules.json'.")

def visualise():
    with open("updated_generated_molecules.json", "r") as f:
        analysed_results = json.load(f)

    seed_smiles = extract_seed_smiles_from_updated_json("updated_generated_molecules.json")
    print(f"Found {len(seed_smiles)} unique starting SMILES in updated_generated_molecules.json")

    visualiser = Visualiser(analysed_results)
    visualiser.plot_groupwise_novelty_across_starting_smiles(
        radius=2,
        nbits=2048,
        method="average",
        save_fig=True,
        filename="groupwise_novelty_boxplot.png",
        limit_most_novel=24,
        limit_least_novel=24
    )
if __name__ == "__main__":
    # generate()
    # analyse()
    visualise()
    pass
