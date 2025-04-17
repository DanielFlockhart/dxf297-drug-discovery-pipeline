
from rdkit import Chem
from rdkit.Chem import Draw

def smile_to_img(smile, size=(500, 500), bond_length=25):
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smile}")
    drawing_options = Draw.MolDrawOptions()
    drawing_options.fixedBondLength = bond_length
    img = Draw.MolToImage(mol, size=size, options=drawing_options)
    
    return img

if __name__ == "__main__":
    # Example SMILES string
    smile = "CN1CC(=O)N2[C@H](c3ccc4c(c3)OCO4)c3[nH]c4ccccc4c3C[C@@H]2C1=O"  # Ethanol
    img = smile_to_img(smile)
    img.show()  # This will display the image in a window