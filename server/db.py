import firebase_admin
from firebase_admin import credentials, storage, firestore
import csv
cred = credentials.Certificate("key.json")
firebase_admin.initialize_app(cred, {
    "storageBucket": "erudite-8d040.firebasestorage.app" 
})

import rdkit
from rdkit import Chem
from rdkit.Chem import Draw

db = firestore.client()
bucket = storage.bucket() 

def add_starting_molecule(name, smile, img_path,effects,classes):
    data = {
        "smile": smile,
         "img": img_path,
        "name": name,
        "effects": effects,
        "classes": classes
    }

    try:
        db.collection("molecules").document(name).set(data)
        print(f"Molecule {name} added successfully.")
    except Exception as e:
        print(f"Error adding molecule {name}: {e}")


csv_File = "mols.csv"
# Iterate over the rows of the CSV file and extract the MOLECULE,SMILES,EFFECTS,CLASSES and add them to Firestore
# with open(csv_File, "r") as file:
#     reader = csv.DictReader(file)
#     for row in reader:
#         name = row["MOLECULE"]
#         smile = row["SMILES"]
#         img_path = "starting-images/" + name + ".png"
#         effects_cleaned = row["EFFECTS"].replace(" ", "").replace("[", "").replace("]", "")
#         classes_cleaned = row["CLASSES"].replace(" ", "").replace("[", "").replace("]", "")
#         effects = effects_cleaned.split(",")
#         classes = classes_cleaned.split(",")

#         # Add the molecule to Firestore
#         add_starting_molecule(name, smile, img_path, effects, classes)

def smile_to_img(smile, size=(500, 500), bond_length=25):
    """
    Converts a SMILES string to an image with consistent relative scaling using RDKit.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    
    drawing_options = Draw.MolDrawOptions()
    drawing_options.fixedBondLength = bond_length
    img = Draw.MolToImage(mol, size=size, options=drawing_options)
    return img

with open(csv_File, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        try:
            name = row["MOLECULE"]
            smile = row["SMILES"]
            destination_img_path = "starting-images/" + name + ".png"
            img = smile_to_img(smile)
            blob = bucket.blob(destination_img_path)
            img.save("temp.png")
            blob.upload_from_filename("temp.png")

            print(f"Image for molecule {name} uploaded successfully.")
        except Exception as e:
            print(f"Error uploading image for molecule {name}: {e}")
            