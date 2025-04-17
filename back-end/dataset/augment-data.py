import openai
import pandas as pd
import csv
import os
import sys
import re
import time

sys.path.append(os.path.abspath(os.path.join('..')))
from system.constants import effects, classes

api_key = ''

file = "../../datasets/drugs/dataset.csv"
data = pd.read_csv(file)

data['MOLECULE'] = data['MOLECULE'].str.lower().str.strip()

# Enhanced prompt for classification
def generate_prompt(molecules, effects, classes):
    return f"""
    For each molecule listed below, classify it by selecting the most relevant effect(s) and class(es) from the provided lists.

    Instructions:
    - Make a best effort to determine at least one effect and one class per molecule, if there is more than one effect or class classification possible, list them all.
    - Only select from the provided lists.
    - If uncertain, infer based on similar or structurally related molecules.
    - Only use effects and classes from the provided lists; do not create new labels.
    

    Format exactly as follows (use lowercase letters only):

    Molecule: <molecule name>
    Effect(s): effect1, effect2
    Class(es): class1, class2

    Available Effects:
    {', '.join(effects.values())}

    Available Classes:
    {', '.join(classes.values())}

    Molecules:
    {chr(10).join(molecules)}
    """

def clean_response(response):
    molecule_data_dict = {}
    pattern = re.compile(
        r"Molecule:\s*(.+?)\s*Effect\(s\):\s*(.*?)\s*Class\(es\):\s*(.*?)(?=\nMolecule:|\Z)",
        re.IGNORECASE | re.DOTALL
    )

    matches = pattern.findall(response)
    for molecule, effects_str, classes_str in matches:
        molecule = molecule.lower().strip()
        effects_list = [e.strip().lower() for e in effects_str.split(',') if e.strip()] or ["no effect"]
        classes_list = [c.strip().lower() for c in classes_str.split(',') if c.strip()] or ["no class"]

        molecule_data_dict[molecule] = {
            "effects": effects_list,
            "classes": classes_list
        }

    return molecule_data_dict

def classify_molecule_batch(molecules, client, effects, classes, model="gpt-4o", max_retries=3):
    prompt = generate_prompt(molecules, effects, classes)

    retries = 0
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0,
            )

            content = response.choices[0].message.content
            molecule_data_dict = clean_response(content)

            labeled_data = []
            for molecule, molecule_data in molecule_data_dict.items():
                effects_vector = [1 if effect.lower() in molecule_data["effects"] else 0 for effect in effects.values()]
                classes_vector = [1 if cls.lower() in molecule_data["classes"] else 0 for cls in classes.values()]

                labeled_data.append({
                    'MOLECULE': molecule,
                    'EFFECTS': effects_vector,
                    'CLASSES': classes_vector
                })
            return labeled_data

        except Exception as e:
            retries += 1
            print(f"Attempt {retries}/{max_retries} failed with error: {str(e)}. Retrying...")
            time.sleep(2 ** retries)

    print(f"All {max_retries} retries failed for batch: {molecules}")
    return []

def classify_molecules(data, client, effects, classes, batch_size=10):
    labeled_molecules = []
    for i in range(0, len(data), batch_size):
        molecule_batch = [m.strip().lower() for m in data.iloc[i:i+batch_size]['MOLECULE'].tolist() if pd.notna(m)]
        print(f"Processing batch {i}/{len(data)}")
        labels = classify_molecule_batch(molecule_batch, client, effects, classes)
        labeled_molecules.extend(labels)

    return labeled_molecules

if __name__ == '__main__':
    client = openai.OpenAI(api_key=api_key)
    labeled_molecules = classify_molecules(data, client, effects, classes)

    labeled_data_df = pd.DataFrame(labeled_molecules)

    merge_df = pd.merge(data[['MOLECULE', 'SMILES', 'VECTOR']], labeled_data_df, on='MOLECULE', how='inner')

    final_df = merge_df[['MOLECULE', 'SMILES', 'VECTOR', 'EFFECTS', 'CLASSES']]

    final_df.to_csv("labeled_molecules.csv", index=False, quoting=csv.QUOTE_ALL)

    print("Classification complete, results saved to labeled_molecules.csv")
