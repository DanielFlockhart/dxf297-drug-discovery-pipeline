import openai
from rdkit import Chem
from rdkit.Chem import Draw
import json,re

api_key = ''
client = openai.OpenAI(api_key=api_key)

def is_valid_smiles(smiles):
    """Check if the input SMILES string is valid."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def extract_json_from_response(response_text):
    """Extract only the JSON portion from OpenAI's response."""
    try:
        match = re.search(r"\{[\s\S]*\}", response_text) 
        if match:
            return json.loads(match.group()) 
        else:
            return {"error": "Failed to extract valid JSON from response"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

def generate_synthesis_route(smiles):
    """Call OpenAI API to generate a multi-step synthesis route and return structured JSON."""
    
    prompt = f"""
    Given the following molecule represented by the SMILES string: {smiles}, 
    propose a detailed multi-step synthesis route. Provide a step-by-step synthesis plan, 
    including the reagents, conditions, and expected yields.

    Ensure the response is formatted as valid JSON with multiple steps:
    {{
        "steps": [
            {{
                "step": 1,
                "starting_material": "...",
                "intermediate": "...",
                "reagents": "...",
                "conditions": "...",
                "yield": "..."
            }},
            {{
                "step": 2,
                "starting_material": "...",
                "final_product": "...",
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


if __name__ == "__main__":
    smiles = input("Enter a SMILES string: ").strip()

    if not is_valid_smiles(smiles):
        print("Invalid SMILES input. Please enter a valid molecular structure.")
    else:
        print("\nGenerating synthesis route...")
        synthesis_plan = generate_synthesis_route(smiles)

        synthesis_json = json.dumps(synthesis_plan, indent=4)
