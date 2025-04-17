import os,sys
sys.path.append(os.path.abspath(os.path.join('..')))
import pandas as pd
from tqdm import tqdm

import os, sys
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer
import csv
import requests
import difflib



def search_smiles(substance_name):
    query = substance_name.replace(' ', '+')
    
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/IsomericSMILES/JSON"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/cids/JSON"
            search_response = requests.get(search_url)
            search_response.raise_for_status()
            cids = search_response.json().get('IdentifierList', {}).get('CID', [])
            
            if not cids:
                return None
            
            cid_str = ','.join(map(str, cids))
            smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/CanonicalSMILES,MolecularFormula/JSON"
            smiles_response = requests.get(smiles_url)
            smiles_data = smiles_response.json().get('PropertyTable', {}).get('Properties', [])
            closest_match = None
            highest_similarity = 0
            for item in smiles_data:
                molecular_formula = item.get('MolecularFormula', '')
                smiles = item.get('CanonicalSMILES', '')
                similarity = difflib.SequenceMatcher(None, substance_name.lower(), molecular_formula.lower()).ratio()
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    closest_match = smiles

            return closest_match
        
        data = response.json()
        smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
        return smiles
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred for '{substance_name}': {http_err}")
        return None
    except Exception as err:
        print(f"An error occurred for '{substance_name}': {err}")
        return None

def process_molecules(input_file, output_file):
    with open(input_file, 'r') as file:
        molecule_names = [line.strip() for line in file.readlines()]
    
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['MOLECULE', 'SMILES'])
        
        for molecule_name in molecule_names:
            if molecule_name:
                print(f"Processing '{molecule_name}'...")
                
                smiles = search_smiles(molecule_name)
                
                if smiles:
                    csvwriter.writerow([molecule_name, smiles])
                else:
                    csvwriter.writerow([molecule_name, 'Not found'])

def get_vectors():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    input_csv = "dataset.csv"
    df = pd.read_csv(input_csv)
    
    df["SMILES"] = df["SMILES"].astype(str).str.replace('"', '').str.strip()
    
    smiles_list = df["SMILES"].tolist()
    
    batch_size = 32 
    vectors = []
    
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing Batches"):
        batch_smiles = smiles_list[i : i + batch_size]
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
        vectors.extend(batch_vectors)
    
    df["VECTOR"] = vectors
    
    output_csv = "molecules_with_vectors.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"Processing complete. Output saved to '{output_csv}'.")

