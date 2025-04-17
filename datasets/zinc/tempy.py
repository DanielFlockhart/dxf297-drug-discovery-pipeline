import pandas as pd

# Replace these with your file paths
# input_csv = 'zinc_250k.csv'
# output_csv = 'filtered.csv'

# # Load the CSV file
# df = pd.read_csv(input_csv)

# # Keep only the 'smiles' column (case-sensitive)
# if 'smiles' in df.columns:
#     df_smiles = df[['smiles']]
#     # Save the reduced dataframe to a new CSV
#     df_smiles.to_csv(output_csv, index=False)
#     print(f"Successfully saved smiles-only CSV to '{output_csv}'")
# else:
#     print("Error: 'smiles' column not found.")



# # Load CSV, ensuring proper handling of newline characters
# df = pd.read_csv(input_csv, lineterminator='\n')

# # Clean whitespace and newlines from the "smiles" column
# df['smiles'] = df['smiles'].str.strip().replace('\n', '', regex=True)

# # Save back to CSV, quoting each SMILES properly
# df.to_csv(output_csv, index=False, quoting=1) # quoting=1 means QUOTE_ALL
import torch
import warnings
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


if __name__ == "__main__":
    # Set device for torch (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the ChemBERTa model and tokenizer.
    model_name = "seyonec/ChemBERTa-zinc-base-v1"
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Read the CSV file (update the filename as needed)
    input_csv = 'output_file.csv'
    df = pd.read_csv(input_csv)
    
    # Remove any extraneous quotes from the SMILES column.
    df["SMILES"] = df["SMILES"].astype(str).str.replace('"', '').str.strip()
    
    # Get a list of all SMILES strings (process the entire dataset)
    smiles_list = df["SMILES"].tolist()
    
    batch_size = 64  # Adjust based on your available memory
    vectors = []
    
    # Process SMILES in batches.
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Processing Batches"):
        batch_smiles = smiles_list[i : i + batch_size]
        inputs = tokenizer(batch_smiles, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        # Mean pooling across sequence length (dim=1)
        batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().tolist()
        vectors.extend(batch_vectors)
    
    # Add the computed vectors as a new column in the DataFrame.
    df["VECTOR"] = vectors
    
    # Write the DataFrame to an output CSV.
    output_csv = "molecules_with_vectors.csv"
    df.to_csv(output_csv, index=False)
    
    print(f"Processing complete. Output saved to '{output_csv}'.")

