# Standard libraries
import os
import random
import math
import io
import csv
import ast
import json
import re
import uuid
from datetime import datetime
import base64

# Third-party libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
import requests
import plotly.express as px

# Flask and related
from flask import Flask, request, jsonify
from flask_cors import CORS

# RDKit libraries
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import DataStructs


# Machine Learning libraries
from transformers import AutoModel, AutoTokenizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Firebase
import firebase_admin
from firebase_admin import firestore

# OpenAI
import openai



# Set device for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import sys
import types

sys.modules["architectures"] = types.ModuleType("architectures")
sys.modules["architectures.classifier"] = sys.modules["__main__"]
sys.modules["architectures.autoencoder"] = sys.modules["__main__"]
sys.modules["architectures.transformer"] = sys.modules["__main__"]
sys.modules["architectures.decoder"] = sys.modules["__main__"]


# https://erudite-backend-798229031686.europe-west2.run.app

# Functions to update the site for my usage.

# docker buildx build --platform linux/amd64 -t gcr.io/erudite-8d040/erudite-backend:latest .
# docker push gcr.io/erudite-8d040/erudite-backend
# gcloud run deploy erudite-backend \
#     --image gcr.io/erudite-8d040/erudite-backend \
#     --platform managed \
#     --region europe-west2 \
#     --allow-unauthenticated
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
    'C': 0, 
    'c': 1,
    'O': 2,
    'o': 3,
    'N': 4,
    'n': 5,
    'Cl': 6,
    'Br': 7,
    'S': 8,
    's': 9,
    'P': 10,
    'p': 11,
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




# ---------------- Constants ----------------
CLASSES_COUNT = len(classes)
EFFECTS_COUNT = len(effects)
MOLECULE_LENGTH = 768
MAX_SMILES_LENGTH = 350
VOCAB_SIZE = len(smiles_vocab)
LEARNING_RATE = 1e-4



# ---------------- Hyperparameters ----------------
AUTOENCODER_INPUT_SIZE = EFFECTS_COUNT + MOLECULE_LENGTH + CLASSES_COUNT
SMILES_GENERATOR_HIDDEN_SIZE = 256
LATENT_DIM = 256

import os
NUM_WORKERS = 0


# ---------------- Models ----------------


class Classifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.3) -> None:
        super(Classifier, self).__init__()
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


class MoleculeClassifier:
    def __init__(self, input_dim: int, num_classes: int, num_effects: int, learning_rate: float) -> None:
        self.class_model = Classifier(input_dim, num_classes).to(device)
        self.effect_model = Classifier(input_dim, num_effects).to(device)
        self.class_criterion = nn.BCEWithLogitsLoss()
        self.effect_criterion = nn.BCEWithLogitsLoss()
        self.class_optimizer = optim.Adam(self.class_model.parameters(), lr=learning_rate * 0.2, weight_decay=1e-4)
        self.effect_optimizer = optim.Adam(self.effect_model.parameters(), lr=learning_rate * 0.2, weight_decay=1e-4)

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


class SMILESGenerator(nn.Module):
    def __init__(self, vector_size, hidden_size, vocab_size, max_smiles_length):
        super(SMILESGenerator, self).__init__()
        self.vector_size = vector_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_smiles_length = max_smiles_length

        self.fc_hidden = nn.Linear(vector_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=smiles_vocab['<PAD>'])
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, vector, target_smiles=None, teacher_forcing_ratio=0.8):
        batch_size = vector.size(0)
        hidden = self.fc_hidden(vector).unsqueeze(0)
        input_token = torch.full((batch_size, 1), smiles_vocab['<START>'], dtype=torch.long, device=vector.device)
        outputs = []

        for t in range(self.max_smiles_length):
            embedded = self.dropout(self.embedding(input_token))
            output, hidden = self.rnn(embedded, hidden)
            token_logits = self.fc_out(output.squeeze(1))
            outputs.append(token_logits)

            if target_smiles is not None and random.random() < teacher_forcing_ratio:
                input_token = target_smiles[:, t].unsqueeze(1)
            else:
                input_token = token_logits.argmax(dim=1, keepdim=True)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def beam_search_decode(self, vector, beam_width=10, temperature=0.8, remove_start=True):
        self.eval()
        with torch.no_grad():
            hidden = self.fc_hidden(vector.unsqueeze(0)).unsqueeze(0)
            start_token = torch.tensor([[smiles_vocab['<START>']]], device=vector.device)
            beams = [(start_token, 0.0, hidden)]
            completed = []

            for _ in range(self.max_smiles_length):
                new_beams = []
                for seq, log_prob, hidden_state in beams:
                    if seq[0, -1].item() == smiles_vocab['<END>']:
                        completed.append((seq, log_prob))
                        continue

                    emb = self.embedding(seq[:, -1].unsqueeze(1))
                    out, new_hidden = self.rnn(emb, hidden_state)
                    logits = self.fc_out(out.squeeze(1)) / temperature
                    log_probs = nn.functional.log_softmax(logits, dim=1)

                    topk_log_probs, topk_indices = log_probs.topk(beam_width)
                    for i in range(beam_width):
                        next_token = topk_indices[:, i].unsqueeze(0)
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_log_prob = log_prob + topk_log_probs[0, i].item()
                        new_beams.append((new_seq, new_log_prob, new_hidden))

                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                if all(seq[0, -1].item() == smiles_vocab['<END>'] for seq, _, _ in beams):
                    completed.extend([(seq, log_prob) for seq, log_prob, _ in beams])
                    break

            if not completed:
                completed = [(seq, log_prob) for seq, log_prob, _ in beams]

            best_seq, _ = max(completed, key=lambda x: x[1])
            token_list = best_seq.squeeze(0).tolist()

            if remove_start and token_list[0] == smiles_vocab['<START>']:
                token_list = token_list[1:]

            smiles = decode_smiles_token_sequence(token_list)
            return fix_smiles(smiles)

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


    
# ------------ utils ------------

model = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1').to(device)
tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
def smile_to_vector_ChemBERTa( smile):

    inputs = tokenizer(smile, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    vector = outputs.last_hidden_state.squeeze(0).mean(dim=0).cpu()
    return vector.tolist()


def isValidSMILES(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if (mol is not None) and (len(smiles) < MAX_SMILES_LENGTH):
            return True
        else :
            return False
    except:
        return False

def decode_smiles_token_sequence(token_sequence):
    smiles_string = ""
    for token_tensor in token_sequence:
        # Convert tensor to integer if needed.
        token_idx = token_tensor.item() if isinstance(token_tensor, torch.Tensor) else token_tensor
        token = idx_to_token.get(token_idx, "")
        if token == '<END>':
            break
        if token in ['<START>', '<PAD>']:
            continue
        smiles_string += token
    return smiles_string


    
def fix_smiles(smiles, max_bfs_iterations=20, max_candidates_per_level=40):
    if isValidSMILES(smiles):
        return smiles
    def clean_smiles(s):
        s = s.replace("<UNK>", "")
        s = re.sub(r"=+", "=", s)
        return s

    smiles = clean_smiles(smiles)
    if isValidSMILES(smiles):
        return smiles
    
    # A basic fix to balance parentheses.
    def basic_fix(s):
        fixed = []
        open_count = 0
        for char in s:
            if char == '(':
                open_count += 1
                fixed.append(char)
            elif char == ')':
                if open_count > 0:
                    open_count -= 1
                    fixed.append(char)
                else:
                    # Skip unmatched closing bracket.
                    continue
            else:
                fixed.append(char)
        # Append missing closing parentheses.
        fixed.extend(')' * open_count)
        return ''.join(fixed)

    # Step 1: Apply basic fix.
    candidate = basic_fix(smiles)
    if isValidSMILES(candidate):
        return candidate
    
    # Step 2: Attempt to fix ring closure digit imbalances.
    def fix_ring_digits(s):
        candidates = {s}
        for d in '0123456789':
            if s.count(d) % 2 != 0:
                new_candidates = set()
                for cand in candidates:
                    indices = [i for i, c in enumerate(cand) if c == d]
                    for idx in indices:
                        new_candidates.add(cand[:idx] + cand[idx+1:])
                candidates = candidates.union(new_candidates)
        return candidates
    
    candidate_set = fix_ring_digits(candidate)
    candidate2 = None
    for cand in candidate_set:
        fixed_cand = basic_fix(cand)
        if isValidSMILES(fixed_cand):
            return fixed_cand
        if candidate2 is None or len(fixed_cand) < len(candidate2):
            candidate2 = fixed_cand
    if candidate2 is None:
        candidate2 = candidate
    
    # Step 3: Use BFS to try removing one problematic character at a time.
    # We extend the problematic characters to include redundant "=" signs if they occur consecutively.
    visited = {smiles, candidate, candidate2}
    level = {candidate, candidate2}
    iterations = 0
    best_candidate = candidate2  # heuristic: candidate with shortest length
    
    while level and iterations < max_bfs_iterations:
        iterations += 1
        next_level = set()
        for s in level:
            for i, char in enumerate(s):
                # Remove characters if they are in the problematic set.
                # For '=' we check if it is part of a consecutive sequence.
                if char in "()0123456789" or (char == "=" and i+1 < len(s) and s[i+1] == "="):
                    new_s = s[:i] + s[i+1:]
                    new_s = basic_fix(new_s)
                    new_s = clean_smiles(new_s)  # Re-run cleaning in case removal creates new patterns.
                    if new_s in visited:
                        continue
                    visited.add(new_s)
                    if isValidSMILES(new_s):
                        return new_s
                    next_level.add(new_s)
                    if len(new_s) < len(best_candidate):
                        best_candidate = new_s
        # Limit candidates per level to avoid exponential explosion.
        if len(next_level) > max_candidates_per_level:
            next_level = set(list(next_level)[:max_candidates_per_level])
        level = next_level

    # If no valid candidate is found, return the best candidate found.
    return best_candidate


app = Flask(__name__)
CORS(app)  # Enables CORS for all routes and origins

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

encoder_checkpoint_path = "models/autoencoder/autoencoder.pth"
transformer_checkpoint_path = "models/transformer/best_scientist_model.pth"
decoder_checkpoint_path = "models/decoder/decoder_model.pth"
classifier_checkpoint_path = "models/classifier/classifier_model.pth"

autoencoder = MolecularAutoencoder(input_size=AUTOENCODER_INPUT_SIZE, latent_dim=LATENT_DIM).to(device)
decoder = SMILESGenerator(
        vector_size=MOLECULE_LENGTH,
        hidden_size=SMILES_GENERATOR_HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        max_smiles_length=MAX_SMILES_LENGTH
    ).to(device)
transformer = ConditionalTransformer(LATENT_DIM, EFFECTS_COUNT, MOLECULE_LENGTH)
classifier = MoleculeClassifier(MOLECULE_LENGTH, CLASSES_COUNT, EFFECTS_COUNT, LEARNING_RATE)


def load_all_models():
    global autoencoder, decoder, transformer, classifier
    load_model(autoencoder, encoder_checkpoint_path)
    load_model(decoder, decoder_checkpoint_path)
    # Load the transformer model

    checkpoint = torch.load(transformer_checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    transformer.load_state_dict(state_dict)
    transformer.eval()

    classifier_checkpoint = torch.load(
        classifier_checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False
    )

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
from rdkit.Chem import AllChem




def smile_to_img(smile, size=(500, 500), bond_length=25):
    smile = standardise_smiles(smile)
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    AllChem.Compute2DCoords(mol)
    drawing_options = Draw.MolDrawOptions()
    drawing_options.fixedBondLength = bond_length
    img = Draw.MolToImage(mol, size=size, options=drawing_options)
    return img


import base64

import io
import base64

import traceback

def rank_molecules_by_effect(molecules, desired_effect, classifier):
    desired_effect_tensor = torch.tensor(desired_effect, dtype=torch.float32)
    fixed_molecules = []
    for smiles in tqdm(molecules, desc="Fixing SMILES", unit="mol"):
        fixed = fix_smiles(smiles)
        if fixed is not None:
            fixed_molecules.append(fixed)

    if not fixed_molecules:
        raise ValueError("No valid canonical SMILES found.")

    vectors = []
    for smiles in tqdm(fixed_molecules, desc="Converting SMILES to vectors", unit="mol"):
        try:
            vector = torch.tensor(
                smile_to_vector_ChemBERTa(smiles),
                dtype=torch.float32
            )
            vectors.append(vector)
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {e}")


    vectors = torch.stack(vectors)
    with torch.no_grad():
        predicted_effects = classifier.effect_model(vectors)
    scores = torch.sum((predicted_effects - desired_effect_tensor) ** 2, dim=1)
    scored_molecules = list(zip(fixed_molecules, scores.tolist()))
    scored_molecules = list(set(scored_molecules))
    scored_molecules = sorted(scored_molecules, key=lambda x: x[1])
    return scored_molecules


firebase_admin.initialize_app()

db = firestore.client()
def save_group_to_firestore(group_id, starting_smiles, desired_effects,user_id):
    try:
        # Create document reference in Firestore
        doc_ref = db.collection("generated").document(group_id)
        
        # Prepare data
        group_data = {
            "user_id": user_id,
            "group_id": group_id,
            "starting_smiles": starting_smiles,
            "desired_effects": desired_effects,
            "time_created": datetime.utcnow(),
        }
        
        # Save to Firestore
        doc_ref.set(group_data)
    except Exception as e:
        print(f"Error saving group {group_id} to Firestore: {e}")
def save_molecule_to_firestore(group_id, molecule_id, effects, classes, user_id,molecule):
    try:
        group_ref = db.collection("generated").document(group_id)

        molecule_ref = group_ref.collection("molecules").document(molecule_id)

        molecule_data = {
            "id": molecule_id,
            "group_id": group_id,
            "user_id": user_id,
            "time_generated": datetime.utcnow(),
            "effects": effects,
            "classes": classes,
            "smiles": molecule
        }

        molecule_ref.set(molecule_data)
        print(f"Molecule {molecule_id} saved to Firestore under group {group_id}.")
    except Exception as e:
        print(f"Error saving molecule {molecule_id} to Firestore: {e}")


    
load_all_models()
@app.route("/predictClasses", methods=["POST"])
def predict_classes():
    try:
        data = request.json
        smiles = data.get("smiles")

        if not smiles or not isValidSMILES(smiles):
            return jsonify({"error": "Invalid or missing SMILES string"}), 400

        vector = torch.tensor(
            smile_to_vector_ChemBERTa(smiles),
            dtype=torch.float32
        ).unsqueeze(0)

        class_probs = classifier.predict_class_probabilities(vector, list(classes.values()), classifier.class_model)

        return jsonify({
            "smiles": smiles,
            "predicted_classes": class_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predictEffects", methods=["POST"])
def predict_effects():
    try:
        data = request.json
        smiles = data.get("smiles")

        if not smiles or not isValidSMILES(smiles):
            return jsonify({"error": "Invalid or missing SMILES string"}), 400

        vector = torch.tensor(
            smile_to_vector_ChemBERTa(smiles),
            dtype=torch.float32
        ).unsqueeze(0)
        effect_probs = classifier.predict_class_probabilities(vector, list(effects.values()), classifier.effect_model)

        return jsonify({
            "smiles": smiles,
            "predicted_effects": effect_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def standardise_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    return canonical_smiles



@app.route("/generateImage", methods=["POST"])
def generate_image():
    try:
        data = request.json
        smiles = data.get("smiles")

        if not smiles or not isValidSMILES(smiles):
            return jsonify({"error": "Invalid or missing SMILES string"}), 400

        try:
            img = smile_to_img(smiles)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            buffer.seek(0)
            img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            print("Image generated successfully.")

            return jsonify({
                "smiles": smiles,
                "image": img_str
            })
        except Exception as e:

            return jsonify({"error": f"Error generating image: {str(e)}"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500
def is_chemically_reasonable(smiles, max_ring_size=8):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    ring_info = mol.GetRingInfo()
    for ring in ring_info.AtomRings():
        if len(ring) > max_ring_size:
            return False
    return True

@app.route("/generateMolecules", methods=["POST"])
def generate_molecules():
    try:
        print("It ran")
        data = request.json
        smiles_input = data.get("smiles")
        desired_effect = data.get("desired_effect")
        generate_molecules_count = data.get("generate_molecules", 10)

        if not isValidSMILES(smiles_input):
            return jsonify({"error": "Invalid SMILES string"}), 400

        group_id = str(uuid.uuid4())

        smiles_in_vector = smile_to_vector_ChemBERTa(smiles_input)

       # Convert list to tensor if needed
        if isinstance(smiles_in_vector, list):
            smiles_in_vector = torch.tensor(smiles_in_vector, dtype=torch.float32)

        # Ensure smiles_in_vector has a batch dimension: [1, feature_dim]
        if smiles_in_vector.dim() == 1:
            smiles_in_vector = smiles_in_vector.unsqueeze(0)

        # Now pass through the classifiers
        effects_in_vector = classifier.effect_model(smiles_in_vector).squeeze(0)
        classes_in_vector = classifier.class_model(smiles_in_vector).squeeze(0)

        # And add a batch dim back to each if necessary: [1, effect_dim] and [1, class_dim]
        if effects_in_vector.dim() == 1:
            effects_in_vector = effects_in_vector.unsqueeze(0)
        if classes_in_vector.dim() == 1:
            classes_in_vector = classes_in_vector.unsqueeze(0)


        combined_input = torch.cat((effects_in_vector, classes_in_vector, smiles_in_vector), dim=1)
        vector = autoencoder.encoder(combined_input)
        desired_effect_in_tensor = torch.tensor(desired_effect, dtype=torch.float32).unsqueeze(0)
        generated_smiles = []
        target_pool_size = generate_molecules_count * 3

        with torch.no_grad():
            with tqdm(total=target_pool_size, desc="Generating molecules", unit="mol") as pbar:
        
                attempts = 0
                generated_smiles_set = set()

                while attempts < target_pool_size:
                    try:
                        # Use no noise on the first attempt and a small noise afterwards.
                        noise_scale = 0 if attempts == 0 else 0.05
                        noisy_vector = vector + (torch.randn_like(vector) * noise_scale)
                        latent_vector = transformer(noisy_vector, desired_effect_in_tensor)
                        new_smiles = decoder.beam_search_decode(latent_vector[0], beam_width=4, temperature=0.7)

                        isValid = isValidSMILES(new_smiles)
                        isChemicallyReasonable = is_chemically_reasonable(new_smiles)
                        if isValid and isChemicallyReasonable:
                            generated_smiles_set.add(new_smiles)
                            print(f"Valid SMILES found ({len(generated_smiles_set)}/{target_pool_size}): {new_smiles}, (attempts: {attempts}), with {noise_scale}% noise")
                        attempts += 1

                    except Exception as e:
                        print(f"Error generating SMILES: {e}")
                        attempts += 1
                        continue


        generated_smiles = list(generated_smiles_set)
        print(f"Generated {len(generated_smiles)} unique SMILES strings. With {attempts} attempts.")
        
        if (len(generated_smiles) > generate_molecules_count):
            ranked_molecules = rank_molecules_by_effect(generated_smiles, desired_effect, classifier)
            extracted_molecules = [mol for mol, score in ranked_molecules][:generate_molecules_count]
        else:
            print("Not enough molecules generated to rank.")
            extracted_molecules = generated_smiles
        

        molecule_images_base64 = []
        for molecule in extracted_molecules[:generate_molecules_count]:
            canon_smile = standardise_smiles(molecule)
            if canon_smile is not None:
                molecule = canon_smile
            try:
                img = smile_to_img(molecule)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                buffer.seek(0)
                img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
                molecule_images_base64.append(img_str)

            except Exception as e:
                print(f"Error generating or uploading image for SMILES {molecule}: {e}")

        return jsonify({
            "molecules": extracted_molecules[:generate_molecules_count],
            "images": molecule_images_base64
        })

    except Exception as e:
        # print the traceback to stderr
        tb = traceback.format_exc()
        app.logger.error(f"🔥 Exception in generateMolecules: {e}\n{tb}")
        return jsonify({"error": str(e)}), 500

DIMENSIONS = 3
CLUSTERS = 20
PERPLEXITY = 10

class DimensionalityReduction:
    def __init__(self, dataset):
        self.data, self.labels, self.names = self.load_data(dataset)

    def load_data(self, dataset):
        vectors, labels, names = [], [], []
        temp_path = f"../models/v2/latent_vectors.csv"
        with open(temp_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in tqdm(reader, desc="Reading CSV data", ncols=80, colour='green'):
                vectors.append(np.array(ast.literal_eval(row['VECTOR'])))
                labels.append(row['SMILES'])
                names.append(row['MOLECULE'])
        return np.array(vectors), labels, names

    def tSNE(self, perplexity=PERPLEXITY, n_components=DIMENSIONS):
        tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(self.data)
        return X_tsne

    def PCA(self, n_components=DIMENSIONS):
        pca = PCA(n_components=n_components, random_state=42)
        X_pca = pca.fit_transform(self.data)
        return X_pca

class Clustering:
    def __init__(self, data):
        self.data = data

    def kmeans(self, clusters=CLUSTERS):
        print("Clustering with KMeans...")
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        return kmeans.fit_predict(self.data)

    def calculate_optimal_kmeans_clusters(self, max_clusters=20):
        silhouette_scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(self.data)
            score = silhouette_score(self.data, cluster_labels)
            silhouette_scores.append(score)
        optimal_k = np.argmax(silhouette_scores) + 2
        return optimal_k, silhouette_scores

class Visualisation:
    def __init__(self, data, labels, names, dimension=DIMENSIONS):
        self.data = data
        self.labels = labels
        self.names = names
        self.dimension = dimension 

    def plot(self, X_model, cluster_labels):
        if self.dimension == 3:
            columns = ["Component 1", "Component 2", "Component 3"]
        else:
            columns = ["Component 1", "Component 2"]
        
        df = pd.DataFrame(X_model, columns=columns)
        df["Label"] = self.labels
        df["Cluster"] = cluster_labels
        df["Name"] = self.names
        
        colours = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray', 'olive']
        
        if self.dimension == 3:
            fig = px.scatter_3d(df, x="Component 1", y="Component 2", z="Component 3", color="Cluster", hover_data=["Label", "Name"], color_continuous_scale=colours)
        else:
            fig = px.scatter(df, x="Component 1", y="Component 2", color="Cluster", hover_data=["Label", "Name"], color_continuous_scale=colours)
        
        fig.show()
        return fig

    def save_plot(self, plot, clusters, model_type, clustering_type):
        file_path = f"graphs/recent.html"
        plot.write_html(file_path)

def calculate_best_hyperparameters(dimensionality_reduction, clustering):
    perplexity_range = [5, 10, 15, 25, 35, 40, 50]
    best_perplexity = None
    best_tsne_score = -1
    for p in perplexity_range:
        X_tsne = dimensionality_reduction.tSNE(perplexity=p)
        score = silhouette_score(X_tsne, KMeans(n_clusters=CLUSTERS, random_state=42).fit_predict(X_tsne))
        if score > best_tsne_score:
            best_tsne_score = score
            best_perplexity = p

    X_tsne = dimensionality_reduction.tSNE(perplexity=best_perplexity)

    clustering = Clustering(X_tsne)
    optimal_k, silhouette_scores = clustering.calculate_optimal_kmeans_clusters(max_clusters=20)
    
    return best_perplexity, optimal_k

    


@app.route("/cluster", methods=["POST"])
def cluster_dataset():
    dataset = "mols.csv"
    dimensionality_reduction = DimensionalityReduction(dataset)
    
    calculate_hyperparameters = False

    if calculate_hyperparameters:
        best_perplexity, optimal_k = calculate_best_hyperparameters(dimensionality_reduction, Clustering(dimensionality_reduction.data))
    else:
        best_perplexity = PERPLEXITY
        optimal_k = CLUSTERS

    X_tsne = dimensionality_reduction.tSNE(perplexity=best_perplexity)

    clustering = Clustering(X_tsne)
    cluster_labels_kmeans = clustering.kmeans(clusters=optimal_k)

    visualisation = Visualisation(X_tsne, dimensionality_reduction.labels, dimensionality_reduction.names)
    fig_tsne = visualisation.plot(X_tsne, cluster_labels_kmeans)
    visualisation.save_plot(fig_tsne, clusters=optimal_k, model_type="tsne", clustering_type="kmeans")


api_key = ''
client = openai.OpenAI(api_key=api_key)

def extract_json_from_response(response_text):
    try:
        match = re.search(r"\{[\s\S]*\}", response_text)  # Regex to find JSON block
        if match:
            return json.loads(match.group())  # Convert to Python dictionary
        else:
            return {"error": "Failed to extract valid JSON from response"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

def generate_synthesis_route(smiles):
    
    prompt = f"""
    Given the following molecule represented by the SMILES string: {smiles}, 
    propose a detailed multi-step synthesis route. Provide a step-by-step synthesis plan, 
    including the reagents, conditions, and expected yields ensuring starting materials are commercially available.

    Ensure the response is formatted as valid JSON with multiple steps (it may need to be lots of different steps).
    {{
        "steps": [
            {{
                "step": 1,
                "starting_material": "...",
                "starting_material_smiles": "{smiles}",
                "intermediate": "...",
                "intermediate_smiles": "...",
                "reagents": "...",
                "conditions": "...",
                "yield": "..."
            }},
            {{
                "step": 2,
                "starting_material": "...",
                "starting_material_smiles": "...",
                "final_product": "...",
                "final_product_smiles": "...",
                "reagents": "...",
                "conditions": "...",
                "yield": "..."
            }}
        ]
    }}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a chemistry expert specializing in organic synthesis."},
                      {"role": "user", "content": prompt}],
            temperature=0.7
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON properly
        synthesis_json = extract_json_from_response(response_text)

        return synthesis_json

    except Exception as e:
        return {"error": str(e)}
    
@app.route("/synthesis", methods=["POST"])
def synthesis():
    try:
        # Parse the incoming JSON request
        data = request.json
        smiles = data.get("smiles")

        # Validate SMILES input
        if not smiles or not isValidSMILES(smiles):
            return jsonify({"error": "Invalid or missing SMILES string"}), 400


        # Call the OpenAI function to generate the synthesis route
        synthesis_plan = generate_synthesis_route(smiles)

        # Check if an error occurred during synthesis generation
        if "error" in synthesis_plan:
            return jsonify({"error": synthesis_plan["error"]}), 500

        return jsonify(synthesis_plan), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
