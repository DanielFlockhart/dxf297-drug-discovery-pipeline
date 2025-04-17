import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
from system.constants import *
from system.utils import *


def truncate_label(label, max_length=20):
    return label if len(label) <= max_length else label[:max_length-3] + "..."

def plot_full_dataset_distribution(csv_file, effects_dict, classes_dict, threshold=0.05):
    df = pd.read_csv(csv_file)
    
    df['EFFECTS'] = df['EFFECTS'].apply(ast.literal_eval)
    df['CLASSES'] = df['CLASSES'].apply(ast.literal_eval)
    
    effects_array = np.array(df['EFFECTS'].tolist())
    classes_array = np.array(df['CLASSES'].tolist())
    
    avg_effects = effects_array.mean(axis=0) * 100
    avg_classes = classes_array.mean(axis=0) * 100

    effect_counts = np.sum(effects_array >= threshold, axis=0)
    valid_effect_indices = [i for i, count in enumerate(effect_counts) if count >= 1]
    valid_avg_effects = avg_effects[valid_effect_indices]
    sorted_order_effects = np.argsort(valid_avg_effects)[::-1]
    sorted_effect_indices = [valid_effect_indices[i] for i in sorted_order_effects]
    sorted_effect_names = [truncate_label(effects_dict[i]) for i in sorted_effect_indices]
    sorted_avg_effects = avg_effects[sorted_effect_indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_effect_names, sorted_avg_effects)
    plt.xticks(rotation=90)
    plt.ylabel("Average Activation (%)")
    plt.title("Distribution of Effects in Full Dataset")
    plt.tight_layout()
    plt.savefig("full_dataset_effects_distribution.png")
    plt.show()

    class_counts = np.sum(classes_array >= threshold, axis=0)
    valid_class_indices = [i for i, count in enumerate(class_counts) if count >= 1]
    valid_avg_classes = avg_classes[valid_class_indices]
    sorted_order_classes = np.argsort(valid_avg_classes)[::-1]
    sorted_class_indices = [valid_class_indices[i] for i in sorted_order_classes]
    sorted_class_names = [truncate_label(classes_dict[i]) for i in sorted_class_indices]
    sorted_avg_classes = avg_classes[sorted_class_indices]
    
    plt.figure(figsize=(12, 6))
    plt.bar(sorted_class_names, sorted_avg_classes)
    plt.xticks(rotation=90)
    plt.ylabel("Average Activation (%)")
    plt.title("Distribution of Classes in Full Dataset")
    plt.tight_layout()
    plt.savefig("full_dataset_classes_distribution.png")
    plt.show()


def plot_distribution(effect_preds, class_preds, effects_dict, classes_dict):
    avg_effects = effect_preds.mean(axis=0) * 100
    avg_classes = class_preds.mean(axis=0) * 100

    threshold = 0.05

    effect_counts = np.sum(effect_preds >= threshold, axis=0)
    valid_effect_indices = [i for i, count in enumerate(effect_counts) if count >= 1]
    valid_avg_effects = avg_effects[valid_effect_indices]
    sorted_order = np.argsort(valid_avg_effects)[::-1]
    sorted_effect_indices = [valid_effect_indices[i] for i in sorted_order]
    sorted_effect_names = [effects_dict[i] for i in sorted_effect_indices]
    sorted_avg_effects = avg_effects[sorted_effect_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_effect_names, sorted_avg_effects)
    plt.xticks(rotation=90)
    plt.ylabel("Average Predicted Probability (%)")
    plt.title("Distribution of Predicted Effects Across Generated Molecules")
    plt.tight_layout()
    plt.savefig("effects_distribution.png")
    plt.show()

    class_counts = np.sum(class_preds >= threshold, axis=0)
    valid_class_indices = [i for i, count in enumerate(class_counts) if count >= 1]
    valid_avg_classes = avg_classes[valid_class_indices]
    sorted_order = np.argsort(valid_avg_classes)[::-1]
    sorted_class_indices = [valid_class_indices[i] for i in sorted_order]
    sorted_class_names = [classes_dict[i] for i in sorted_class_indices]
    sorted_avg_classes = avg_classes[sorted_class_indices]

    plt.figure(figsize=(12, 6))
    plt.bar(sorted_class_names, sorted_avg_classes)
    plt.xticks(rotation=90)
    plt.ylabel("Average Predicted Probability (%)")
    plt.title("Distribution of Predicted Classes Across Generated Molecules")
    plt.tight_layout()
    plt.savefig("classes_distribution.png")
    plt.show()



# ------------------------- Main Function -------------------------

def plot_similarity_distribution(start_smile, generated_mols):
    similarities = [calculate_smiles_similarity(start_smile, mol) * 100 for mol in generated_mols]
    thresholds = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5]
    percentages = []
    for thresh in thresholds:
        count = sum(1 for sim in similarities if sim >= thresh)
        percentages.append(100 * count / len(similarities))
    
    plt.figure(figsize=(8, 6))
    plt.bar([f">={t}%" for t in thresholds], percentages)
    plt.ylabel("Percentage of Molecules (%)")
    plt.title("Structural Similarity Distribution to Starting Molecule")
    plt.tight_layout()
    plt.savefig("similarity_distribution.png")
    plt.show()

# # ------------------------- Main Function -------------------------
# def analyse():
#     # Hardcoded inputs examples

#     # ----------- Sleep targets ----------- 
#     # start_smile = "CC(=O)NCCC1=CNC2=C1C=C(C=C2)OC"
#     # desired_indices = [0, 25] 

#     # ----------- Stimulant targets -----------
#     start_smile = "CN1C2CCC1C(C(C2)OC(=O)C3=CC=CC=C3)C(=O)OC"
#     desired_indices = [17, 27]  # Example: desired effects at indices 1 and 2
    
    
#     num_molecules = 100

#     # Create a binary vector for desired effects of length EFFECTS_COUNT
#     desired_effect = [1.0 if i in desired_indices else 0.0 for i in range(EFFECTS_COUNT)]

#     print(f"Generating {num_molecules} molecules from SMILES: {start_smile}")
#     print(f"Desired effect indices: {desired_indices}")

#     generated_mols, classifier = generate_molecules(start_smile, desired_effect, num_molecules)
#     print("Generated molecules")

#     effect_preds, class_preds = classify_molecules(generated_mols, classifier)
#     print("Classified molecules")

#     # Plot the predicted effects and classes distributions
#     plot_distribution(effect_preds, class_preds, effects, classes)
#     # Additionally, plot the structural similarity distribution comparing each generated molecule to the starting SMILES.
#     plot_similarity_distribution(start_smile, generated_mols)



# # Provide the path to your dataset CSV file
# csv_file = "dataset.csv"

# # Generate the plots
# plot_full_dataset_distribution(csv_file, effects, classes)
