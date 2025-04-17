from tqdm import tqdm
import warnings

warnings.simplefilter("ignore", category=FutureWarning)

def custom_tqdm(iterable, **kwargs):
    defaults = {
        'ncols': 100,
        'colour': 'green',
        'bar_format': '{l_bar}{bar:40}{r_bar} | {postfix}',
        'dynamic_ncols': True,
        'leave': True,
    }
    defaults.update(kwargs)
    return tqdm(iterable, **defaults)



smiles_vocab = {
    # Aliphatic and Aromatic atoms
    'C': 0,   # Aliphatic carbon
    'c': 1,   # Aromatic carbon
    'O': 2,   # Aliphatic oxygen
    'o': 3,   # Aromatic oxygen
    'N': 4,   # Aliphatic nitrogen
    'n': 5,   # Aromatic nitrogen
    'Cl': 6,
    'Br': 7,
    'S': 8,   # Aliphatic sulfur
    's': 9,   # Aromatic sulfur
    'P': 10,  # Aliphatic phosphorus
    'p': 11,  # Aromatic phosphorus (if needed)
    'F': 12,
    'I': 13,
    
    # Bonds, branches, and ring closures
    '=': 14,
    '#': 15,
    '(': 16,
    ')': 17,
    '[': 18,
    ']': 19,
    '@': 20,
    '+': 21,
    '-': 22,
    '/': 23,
    '\\': 24,
    
    # Ring closure digits and two-digit closures
    '1': 25,
    '2': 26,
    '3': 27,
    '4': 28,
    '5': 29,
    '6': 30,
    '7': 31,
    '8': 32,
    '9': 33,
    '%10': 34,
    '%11': 35,
    '%12': 36,
    '%13': 37,
    '%14': 38,
    '%15': 39,
    
    # Additional elements and metals
    'Si': 40,
    'B': 41,
    'Se': 42,
    'Zn': 43,
    'Cu': 44,
    'Fe': 45,
    'Mn': 46,
    'Na': 47,
    'K': 48,
    'Li': 49,
    'Ca': 50,
    'Mg': 51,
    'Al': 52,
    'As': 53,
    'Ba': 54,
    'Co': 55,
    'Ni': 56,
    'Pb': 57,
    'Sn': 58,
    'Ti': 59,
    'V': 60,
    'W': 61,
    'Y': 62,
    'Zr': 63,
    'Au': 64,
    'Ag': 65,
    'Cd': 66,
    'Cr': 67,
    'Ga': 68,
    'Ge': 69,
    'H': 70,
    'He': 71,
    'Ne': 72,
    'Ar': 73,
    'Xe': 74,
    'Kr': 75,
    'Ra': 76,
    'Rb': 77,
    'Sr': 78,
    'Te': 79,
    'Tl': 80,
    'Cs': 81,
    'Be': 82,
    'Sc': 83,
    'Pt': 84,
    'Hg': 85,
    'Re': 86,
    'Ru': 87,
    'Sb': 88,
    'Tc': 89,
    'Th': 90,
    'U': 91,
    'Ac': 92,
    'Am': 93,
    'Np': 94,
    'Pu': 95,
    'Cm': 96,
    'Bk': 97,
    'Cf': 98,
    'Es': 99,
    'Fm': 100,
    'Md': 101,
    'No': 102,
    'Lr': 103,
    'Rf': 104,
    'Db': 105,
    'Sg': 106,
    'Bh': 107,
    'Hs': 108,
    'Mt': 109,
    'Ds': 110,
    'Rg': 111,
    'Cn': 112,
    'Fl': 113,
    'Lv': 114,
    'Ts': 115,
    'Og': 116,
    'Nb': 117,
    'Mo': 118,
    'Rh': 119,
    'Pd': 120,
    'In': 121,
    'Ta': 122,
    'Os': 123,
    'Ir': 124,
    'Bi': 125,
    'Po': 126,
    'At': 127,
    'Rn': 128,
    'Fr': 129,
    'La': 130,
    'Ce': 131,
    'Pr': 132,
    'Nd': 133,
    'Pm': 134,
    'Sm': 135,
    'Eu': 136,
    'Gd': 137,
    'Tb': 138,
    'Dy': 139,
    'Ho': 140,
    'Er': 141,
    'Tm': 142,
    'Yb': 143,
    'Lu': 144,
    'Nh': 145,
    'Mc': 146,
    
    # Special tokens
    '<START>': 147,
    '<END>': 148,
    '<PAD>': 149,
    '<UNK>': 150
}


idx_to_token = {v: k for k, v in smiles_vocab.items()}
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

classes = {
    0: "Amphetamines",
    1: "Benzodiazepines",
    2: "Phenethylamines",
    3: "Tryptamines",
    4: "Cannabinoids",
    5: "Synthetic Cannabinoids",
    6: "Opioids/Opiates",
    7: "Piperazines",
    8: "Dissociatives (Arylcyclohexylamines, etc.)",
    9: "Lysergamides",
    10: "Cathinones",
    11: "Barbiturates",
    12: "GABAergics",
    13: "SARMs/Steroidal Compounds",
    14: "Nootropics",
    15: "Indoles",
    16: "Peptides/Proteins",
    17: "Beta-blockers",
    18: "Antidepressants (SSRIs, SNRIs, etc.)",
    19: "Anesthetics",
    20: "Muscle Relaxants",
    21: "Steroids",
    22: "Analgesics",
    23: "Psychedelics",
    24: "Antipsychotics",
    25: "Anti-inflammatory Agents (NSAIDs, Steroidal)",
    26: "Anti-cancer Agents (Chemotherapy, Targeted therapy)",
    27: "Hormones/Hormonal Modulators",
    28: "Antibiotics",
    29: "Antivirals",
    30: "Cardiovascular Drugs",
    31: "Bronchodilators",
    32: "Diuretics",
    33: "Hypnotics/Sedatives",
    34: "Immunomodulators",
    35: "MAOIs (Monoamine Oxidase Inhibitors)",
    36: "PDE5 Inhibitors (Erectile Dysfunction Agents)",
    37: "Cholinergics",
    38: "Adrenergics",
    39: "Xanthines (e.g., Caffeine, Theophylline)",
    40: "Miscellaneous Research Chemicals (Novel Psychoactive Substances)"
}


MODEL_NAME = "drugs"
CLASSIFIER_PATH = f"models/{MODEL_NAME}/classifier/classifier_model.pth"
DECODER_PATH = f"models/{MODEL_NAME}/decoder/decoder_model.pth"
AUTOENCODER_PATH = f"models/{MODEL_NAME}/autoencoder/autoencoder.pth"
PRETRAINED_TRANSFORMER_PATH = f"models/{MODEL_NAME}/pretransformer/pretrained_transformer_model.pth"
RL_TRANSFORMER_PATH = f"models/{MODEL_NAME}/scientists/best_scientist_model.pth"
SCIENTISTS_PATH = f"models/{MODEL_NAME}/scientists"
RESTORE_SCIENTISTS_PATH = f"models/{MODEL_NAME}/scientists/gen_1.pth"
BEST_SCIENTIST_PATH = f"models/{MODEL_NAME}/scientists/best_scientist_model.pth"
DATASET = f"../datasets/{MODEL_NAME}/dataset.csv"
LATENT_PATH = f"../datasets/{MODEL_NAME}/latent_vectors.csv"
SCIENTISTS_RESTORE_EPOCH = 42


# ---------------- Constants ----------------
SAVE_FREQUENCY = 3
CLASSES_COUNT = len(classes)
EFFECTS_COUNT = len(effects)
MOLECULE_LENGTH = 768
MAX_SMILES_LENGTH = 350
VOCAB_SIZE = len(smiles_vocab)
LEARNING_RATE = 1e-4
TEACHING_FORCE_RATIO = 0.95
TEACHING_FORCE_DECAY = True
ACCUMULATION_STEPS = 4
NUM_SCIENTISTS = 32
NUM_GENERATED_MOLECULES = 18
EFFECT_SIMILARITY_SIGNIFICANCE = 1.0
CLASS_SIMILARITY_SIGNIFICANCE = 0.3
SMILES_SIGNIFICANCE = 0.8
INVALID_SMILES_PENALTY = 0.05
LENGTH_BONUS = 0.1
# Max score -> 1.0 + 0.3 + 0.8 + 0.1 = 2.2
# ---------------- EPOCHS ----------------
CLASSIFIER_EPOCHS = 30000
DECODING_EPOCHS = 30000
AUTOENCODER_EPOCHS = 30000
PRETRAINED_TRANSFORMER_EPOCHS = 30000
RL_TRANSFORMER_EPOCHS = 100

# ---------------- Hyperparameters ----------------
AUTOENCODER_INPUT_SIZE = EFFECTS_COUNT + MOLECULE_LENGTH + CLASSES_COUNT
SMILES_GENERATOR_HIDDEN_SIZE = 256
LATENT_DIM = 256
BATCH_SIZE = 128

# All Trainable in isolation
RESTORE_CLASSIFIER_EPOCH = True
RESTORE_DECODER_EPOCH = True
RESTORE_AUTOENCODER_EPOCH = True

# Trainable after autoencoder
RESTORE_PRETRAINED_TRANSFORMER_EPOCH = True

# Trainable with dependencies on other models
RESTORE_RL_TRANSFORMER_EPOCH = False 
import os
NUM_WORKERS = 0


