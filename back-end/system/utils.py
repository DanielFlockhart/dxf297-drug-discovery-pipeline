from rdkit import Chem
from rdkit.Chem import Draw
import requests,torch,os
from rdkit import Chem
from transformers import AutoModel, AutoTokenizer
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
if __name__ == "__main__":
    from constants import *
else:
    from system.constants import *
from rdkit import RDLogger
import numpy as np
import re
RDLogger.DisableLog('rdApp.error')  # Suppress SMILES parsing errors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smile_to_img(smile, size=(500, 500), bond_length=25):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    drawing_options = Draw.MolDrawOptions()
    drawing_options.fixedBondLength = bond_length
    img = Draw.MolToImage(mol, size=size, options=drawing_options)
    
    return img

def get_smiles_from_name(substance_name):
    query = substance_name.replace(' ', '+')
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/CanonicalSMILES/JSON"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        
        print(f"SMILES notation for '{substance_name}': {smiles}")
        return smiles
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")

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
    
def calculate_smiles_similarity(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = FingerprintMols.FingerprintMol(mol1)
        fp2 = FingerprintMols.FingerprintMol(mol2)
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
        return similarity
    except Exception as e:
        print(f"Error calculating SMILES similarity: {e}")
        return 0.0
    
def print_molecule_results(smiles_string, predicted_effects, predicted_classes):
    print(f"\n--- Molecule Results ---")
    print(f"SMILES: {smiles_string}")
    
    sorted_effects = sorted(
        enumerate(predicted_effects.tolist()), key=lambda x: x[1], reverse=True
    )
    print("\nPredicted Effects (sorted by probabilities):")
    for idx, prob in sorted_effects:
        print(f"  - {effects[idx]}: {round(prob * 100, 2)}%")
    
    sorted_classes = sorted(
        enumerate(predicted_classes.tolist()), key=lambda x: x[1], reverse=True
    )
    print("\nPredicted Classes (sorted by probabilities):")
    for idx, prob in sorted_classes:
        print(f"  - {classes[idx]}: {round(prob * 100, 2)}%")
    print("\n-------------------------")

def preprocess_dataset(path,use_full_dataset=True,fraction=0.1):
    if use_full_dataset:
        df = pd.read_csv(path)
    else:
        print("Using a fraction of the dataset.")
        df = pd.read_csv(path).sample(frac=fraction)

    dataset = []
    smiles_list = []
    molecule_list = []
    for _, row in df.iterrows():
        molecule = row.get('MOLECULE', '')
        smiles = row['SMILES'] 
        molecule_list.append(molecule)
        smiles_list.append(smiles)
        vector = eval(row['VECTOR'])
        effects = eval(row['EFFECTS'])
        classes = eval(row['CLASSES'])
        with_data = effects + classes + vector
        dataset.append(with_data)
    return dataset, smiles_list, molecule_list
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




    
def fix_smiles(smiles, max_bfs_iterations=20, max_candidates_per_level=32):
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
                    continue
            else:
                fixed.append(char)
        fixed.extend(')' * open_count)
        return ''.join(fixed)
    # Apply basic fix.
    candidate = basic_fix(smiles)
    if isValidSMILES(candidate):
        return candidate
    # Attempt to fix ring closure digit imbalances.
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
    
    # Use BFS to try removing one problematic character at a time.
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




def classify_molecules(molecules, classifier):
    effect_predictions = []
    class_predictions = []
    for mol in molecules:
        vec_list = smile_to_vector_ChemBERTa(mol)
        vector = torch.tensor(vec_list, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            effects_pred = classifier.effect_model(vector).squeeze(0).cpu().numpy()
            classes_pred = classifier.class_model(vector).squeeze(0).cpu().numpy()
        effect_predictions.append(effects_pred)
        class_predictions.append(classes_pred)
    return np.array(effect_predictions), np.array(class_predictions)

def standardise_complex_smiles(smi: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None

        normalizer = rdMolStandardize.Normalizer()
        reionizer = rdMolStandardize.Reionizer()

        mol = normalizer.normalize(mol)
        mol = reionizer.reionize(mol)

        can_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

        return can_smi
    except Exception as e:
        print(f"Error in standardizing SMILES: {e}")
        return None
