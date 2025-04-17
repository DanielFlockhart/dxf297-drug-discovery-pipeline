
# import pandas as pd
# import requests
# import difflib
# from tqdm import tqdm

# # Search for Isomeric SMILES via PubChem
# def search_smiles(substance_name):
#     query = substance_name.replace(' ', '+')
#     url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/property/IsomericSMILES/JSON"

#     try:
#         response = requests.get(url)
#         if response.status_code != 200:
#             search_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/cids/JSON"
#             search_response = requests.get(search_url)
#             search_response.raise_for_status()
#             cids = search_response.json().get('IdentifierList', {}).get('CID', [])

#             if not cids:
#                 return None

#             cid_str = ','.join(map(str, cids))
#             smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid_str}/property/IsomericSMILES,MolecularFormula/JSON"
#             smiles_response = requests.get(smiles_url)
#             smiles_data = smiles_response.json().get('PropertyTable', {}).get('Properties', [])

#             closest_match = None
#             highest_similarity = 0
#             for item in smiles_data:
#                 molecular_formula = item.get('MolecularFormula', '')
#                 smiles = item.get('IsomericSMILES', '')
#                 similarity = difflib.SequenceMatcher(None, substance_name.lower(), molecular_formula.lower()).ratio()

#                 if similarity > highest_similarity:
#                     highest_similarity = similarity
#                     closest_match = smiles

#             return closest_match

#         data = response.json()
#         smiles = data['PropertyTable']['Properties'][0]['IsomericSMILES']
#         return smiles

#     except Exception as err:
#         print(f"An error occurred for '{substance_name}': {err}")
#         return None


# # Main function to update SMILES in the existing CSV
# def update_smiles_in_csv(input_csv, output_csv):
#     df = pd.read_csv(input_csv)

#     updated_smiles = []
    
#     for molecule in tqdm(df['MOLECULE'], desc="Updating SMILES"):
#         smiles = search_smiles(molecule)
#         if smiles:
#             updated_smiles.append(smiles)
#         else:
#             updated_smiles.append('Not Found')

#     # Replace SMILES column
#     df['SMILES'] = updated_smiles

#     # Save updated dataframe
#     df.to_csv(output_csv, index=False, quoting=1)


# # Run the update
# if __name__ == '__main__':
#     input_csv = 'dataset.csv'    # your original dataset
#     output_csv = 'updated_dataset.csv' # your new, updated dataset

#     update_smiles_in_csv(input_csv, output_csv)

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

# # Main function to update VECTOR embeddings in the existing CSV
# def update_vectors(input_csv, output_csv):
#     df = pd.read_csv(input_csv)

#     # Set device for torch (GPU if available)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Load the ChemBERTa model and tokenizer
#     model_name = "seyonec/ChemBERTa-zinc-base-v1"
#     model = AutoModel.from_pretrained(model_name).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Clean SMILES strings
#     df["SMILES"] = df["SMILES"].astype(str).str.replace('"', '').str.strip()

#     smiles_list = df["SMILES"].tolist()
#     batch_size = 64
#     vectors = []

#     for i in tqdm(range(0, len(smiles_list), batch_size), desc="Computing Vectors"):
#         batch_smiles = smiles_list[i: i + batch_size]
#         inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)
#         with torch.no_grad():
#             outputs = model(**inputs)
#         batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
#         vectors.extend(batch_vectors)

#     df["VECTOR"] = vectors

#     # Save updated dataframe
#     df.to_csv(output_csv, index=False, quoting=1)


# # Run the update
# if __name__ == '__main__':
#     input_csv = 'dataset.csv'
#     output_csv = 'dataset_with_vectors.csv'

#     update_vectors(input_csv, output_csv)

import pandas as pd
import matplotlib.pyplot as plt

def smiles_length_statistics(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Ensure the 'SMILES' column is a string and compute its length
    df['SMILES'] = df['SMILES'].astype(str)
    df['length'] = df['SMILES'].apply(len)
    
    # Calculate descriptive statistics
    desc = df['length'].describe()
    mean_length = desc['mean']
    min_length = desc['min']
    max_length = desc['max']
    median = df['length'].quantile(0.5)
    q99 = df['length'].quantile(0.99)
    
    print("Summary of SMILES length statistics:")
    print(f"Mean (average) length:  {mean_length:.2f}")
    print(f"Min length:            {min_length}")
    print(f"Max length:            {max_length}")
    print(f"Median (50th percentile): {median}")
    print(f"99th percentile:       {round(q99)}")
    
    # Plotting the distribution graph (histogram)
    plt.figure(figsize=(10, 6))
    plt.hist(df['length'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Distribution of SMILES Lengths')
    plt.xlabel('SMILES Length')
    plt.ylabel('Frequency')
    plt.axvline(mean_length, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_length:.2f}')
    plt.axvline(median, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median}')
    plt.axvline(q99, color='purple', linestyle='dashed', linewidth=1, label=f'99th Percentile: {round(q99)}')
    plt.legend()
    plt.tight_layout()
    
    # Save the graph as PNG BEFORE showing it
    plt.savefig('smiles_length_distribution.png')
    
    # Then display the figure
    plt.show()

if __name__ == "__main__":
    csv_file_path = "../zinc/dataset.csv"  # Replace with the actual CSV file path
    smiles_length_statistics(csv_file_path)

# # Example usage:
# if __name__ == "__main__":
#     csv_file_path = "dataset.csv"  # <-- replace with the actual path
#     smiles_length_statistics(csv_file_path)

# import pandas as pd
# from rdkit import Chem


# def convert_to_aromatic(smiles):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         # If RDKit can't parse the SMILES, return the original string
#         return smiles
#     # Convert back to SMILES (default returns aromatic SMILES)
#     return Chem.MolToSmiles(mol)

# # Load the CSV file (adjust the filename as needed)
# df = pd.read_csv('dataset.csv')

# # Apply the conversion to the 'SMILES' column
# df['SMILES'] = df['SMILES'].apply(convert_to_aromatic)

# # Save the updated dataframe to a new CSV file
# df.to_csv('your_dataset_aromatic.csv', index=False)

# print("Conversion complete! Updated file saved as 'your_dataset_aromatic.csv'.")