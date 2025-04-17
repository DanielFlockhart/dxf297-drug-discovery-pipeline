import subprocess
import os
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_pdb(smiles, filename="ligand.pdb"):
    """Convert a SMILES string to a PDB file."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)
        Chem.MolToPDBFile(mol, filename)
        print(f"Ligand file {filename} created successfully.")
        return filename
    except Exception as e:
        print(f"Error converting SMILES to PDB: {e}")
        return None

def prepare_ligand_to_pdbqt(pdb_file, output_file="ligand.pdbqt"):
    """Convert a PDB file to a PDBQT file using AutoDockTools' prepare_ligand4.py."""
    try:
        command = [
            "~/Downloads/mgltools_1.5.7_MacOS-X/bin/python2.7",
            "~/Downloads/mgltools_1.5.7_MacOS-X/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py",
            "-l", pdb_file,
            "-o", output_file
        ]
        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Ligand PDBQT file {output_file} created successfully.")
            return output_file
        else:
            print(f"Error preparing ligand PDBQT: {result.stderr}")
            return None
    except Exception as e:
        print(f"Error running prepare_ligand4.py: {e}")
        return None

def run_docking(vina_executable, ligand_file, receptor_file, output_file="docked_output.pdbqt"):
    """Run AutoDock Vina for docking."""
    if not os.path.exists(ligand_file):
        print(f"Error: Ligand file {ligand_file} not found.")
        return None
    if not os.path.exists(receptor_file):
        print(f"Error: Receptor file {receptor_file} not found.")
        return None
    
    command = [
        vina_executable,
        "--ligand", ligand_file,
        "--receptor", receptor_file,
        "--out", output_file,
        "--center_x", str(docking_center[0]),
         "--center_y", str(docking_center[1]),
         "--center_z", str(docking_center[2]),
        "--size_x", str(docking_size[0]),
        "--size_y", str(docking_size[1]),
        "--size_z", str(docking_size[2]),
    ]
    
    print(f"Running docking with command: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    print("Vina Output:")
    print(result.stdout)
    print("Vina Errors:")
    print(result.stderr)
    return result.stdout

def parse_vina_results(output_file):
    """Parse the Vina output file for binding affinities."""
    scores = []
    with open(output_file, "r") as file:
        for line in file:
            if "REMARK VINA RESULT" in line:
                parts = line.split()
                scores.append({"affinity": float(parts[3])})
    return scores

# Main workflow
if __name__ == "__main__":
    receptor_pdbqt = "processed-5ht2a.pdbqt"  # Ensure this file exists
    ligand_pdbqt = "processed-ligand.pdbqt"
    docking_center = (10.0, 12.5, 15.0)  # Example approximate coordinates

    docking_size = (20, 20, 20)  # Example docking box size
    docking_output = run_docking("vina", ligand_pdbqt, receptor_pdbqt)
    if docking_output:
        scores = parse_vina_results("docked_output.pdbqt")
        print("Docking Results:")
        for i, score in enumerate(scores):
            print(f"Pose {i+1}: Affinity = {score['affinity']} kcal/mol")
    else:
        print("Docking failed.")
      
