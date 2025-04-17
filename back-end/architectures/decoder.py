import os
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import os,sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from system.constants import *
from system.utils import *


class SMILESGenerator(nn.Module):
    def __init__(self, vector_size, hidden_size, vocab_size, max_smiles_length, num_layers=2, num_heads=4):
        super(SMILESGenerator, self).__init__()
        self.vector_size = vector_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_smiles_length = max_smiles_length

        self.fc_hidden = nn.Linear(vector_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=smiles_vocab['<PAD>'])
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, vector, target_smiles=None, teacher_forcing_ratio=0.8):
        batch_size = vector.size(0)
        hidden = self.fc_hidden(vector).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
        input_token = torch.full((batch_size, 1), smiles_vocab['<START>'], dtype=torch.long, device=vector.device)
        outputs = []
        accumulated_outputs = None

        for t in range(self.max_smiles_length):
            embedded = self.dropout(self.embedding(input_token)) 
            output, hidden = self.rnn(embedded, hidden)
            if accumulated_outputs is None:
                accumulated_outputs = output
            else:
                accumulated_outputs = torch.cat([accumulated_outputs, output], dim=1)
            
            attn_output, _ = self.attention(query=output, key=accumulated_outputs, value=accumulated_outputs)
            combined = output + attn_output 
            token_logits = self.fc_out(combined.squeeze(1))
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
            hidden = self.fc_hidden(vector).unsqueeze(0).repeat(self.rnn.num_layers, 1, 1)
            start_token = torch.tensor([[smiles_vocab['<START>']]], device=vector.device)
            beams = [(start_token, 0.0, hidden, None)]
            completed = []

            for _ in range(self.max_smiles_length):
                new_beams = []
                for seq, log_prob, hidden_state, acc_outputs in beams:
                    if seq[0, -1].item() == smiles_vocab['<END>']:
                        completed.append((seq, log_prob))
                        continue

                    last_token = seq[:, -1].unsqueeze(1)
                    embedded = self.dropout(self.embedding(last_token))
                    output, new_hidden = self.rnn(embedded, hidden_state)

                    if acc_outputs is None:
                        new_acc_outputs = output
                    else:
                        new_acc_outputs = torch.cat([acc_outputs, output], dim=1)

                    attn_output, _ = self.attention(query=output, key=new_acc_outputs, value=new_acc_outputs)
                    combined = output + attn_output
                    logits = self.fc_out(combined.squeeze(1)) / temperature
                    log_probs = nn.functional.log_softmax(logits, dim=1)
                    topk_log_probs, topk_indices = log_probs.topk(beam_width)

                    for i in range(beam_width):
                        next_token = topk_indices[:, i].unsqueeze(0)
                        new_seq = torch.cat([seq, next_token], dim=1)
                        new_log_prob = log_prob + topk_log_probs[0, i].item()
                        new_beams.append((new_seq, new_log_prob, new_hidden, new_acc_outputs))

                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
                if all(seq[0, -1].item() == smiles_vocab['<END>'] for seq, _, _, _ in beams):
                    completed.extend([(seq, log_prob) for seq, log_prob, _, _ in beams])
                    break

            if not completed:
                completed = [(seq, log_prob) for seq, log_prob, _, _ in beams]

            best_seq, _ = max(completed, key=lambda x: x[1])
            token_list = best_seq.squeeze(0).tolist()

            if remove_start and token_list[0] == smiles_vocab['<START>']:
                token_list = token_list[1:]

            smiles = decode_smiles_token_sequence(token_list)
            return fix_smiles(smiles)



    def save_model(self, path, optimizer, epoch, full_model=False):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, optimizer, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch']

class SMILESDataset(Dataset):
    def __init__(self, dataset_path: str, max_smiles_length: int, fraction: float = 1.0):
        self.data = self.preprocess_dataset(dataset_path, max_smiles_length, fraction)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        vector, tokenized_smiles = self.data[idx]
        vector_tensor = torch.tensor(vector, dtype=torch.float32)
        target_tensor = torch.tensor(tokenized_smiles, dtype=torch.long)
        return vector_tensor, target_tensor

    def tokenize_smiles(self, smiles: str, smiles_vocab: dict, max_length: int):
        multi_char_tokens = sorted([k for k in smiles_vocab if len(k) > 1], key=len, reverse=True)
        pattern = '|'.join(map(re.escape, multi_char_tokens))
        tokenized = []
        i = 0
        while i < len(smiles):
            match = re.match(pattern, smiles[i:])
            if match:
                token = match.group(0)
                tokenized.append(smiles_vocab.get(token, smiles_vocab['<UNK>']))
                i += len(token)
            else:
                tokenized.append(smiles_vocab.get(smiles[i], smiles_vocab['<UNK>']))
                i += 1
        tokenized.append(smiles_vocab['<END>'])
        tokenized += [smiles_vocab['<PAD>']] * (max_length - len(tokenized))
        return tokenized[:max_length]

    def preprocess_dataset(self, path: str, max_smiles_length: int, fraction: float):
        df = pd.read_csv(path)
        if fraction < 1.0:
            df = df.sample(frac=fraction, random_state=42)
        dataset = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing dataset"):
            vector = eval(row['VECTOR'])
            tokenized_smiles = self.tokenize_smiles(row['SMILES'], smiles_vocab, max_smiles_length)
            dataset.append((vector, tokenized_smiles))
        return dataset
    def collate_fn(self, batch):
        vectors = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        vectors = torch.stack(vectors, dim=0)
        targets = torch.stack(targets, dim=0)
        return vectors, targets
    
from matplotlib import pyplot as plt
def train_decoder():
    # DATASET = "../datasets/zinc/dataset.csv"
    dataset_obj = SMILESDataset(DATASET, MAX_SMILES_LENGTH, fraction=1.0)

    model = SMILESGenerator(MOLECULE_LENGTH, SMILES_GENERATOR_HIDDEN_SIZE, VOCAB_SIZE, MAX_SMILES_LENGTH).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=smiles_vocab['<PAD>'])

    if RESTORE_DECODER_EPOCH and os.path.exists(DECODER_PATH):
        start_epoch = model.load_checkpoint(optimizer, DECODER_PATH)
        print(f"Checkpoint loaded. Resuming at epoch {start_epoch}")
    else:
        start_epoch = 0

    epoch_range = range(start_epoch, DECODING_EPOCHS)
    data_loader = DataLoader(dataset_obj, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    loss_history = []

    for epoch in tqdm(epoch_range, desc="Generator Training"):
        total_loss = 0.0
        teacher_forcing_ratio = max(0.9 - epoch * 0.005, 0.2)

        for vector, target_smiles in tqdm(data_loader, leave=False):
            vector, target_smiles = vector.to(device), target_smiles.to(device)
            optimizer.zero_grad()

            outputs = model(vector, target_smiles, teacher_forcing_ratio)
            outputs = outputs.view(-1, VOCAB_SIZE)
            targets = target_smiles.view(-1)

            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        loss_history.append(total_loss)
        
        print(f"Epoch {epoch} - Loss: {total_loss:.4f}")

        model.eval()
        idx = random.randint(0, len(dataset_obj) - 1)
        vector, target_smiles = dataset_obj[idx]
        vector = torch.tensor(vector, dtype=torch.float32).to(device)
        generated_smiles = model.beam_search_decode(vector, beam_width=4)
        original_smile = decode_smiles_token_sequence(target_smiles)
        print(f"Epoch {epoch} Original: {original_smile}\nGenerated: {generated_smiles}")
        model.train()

        model.save_model(DECODER_PATH, optimizer, epoch + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(list(epoch_range), loss_history, linewidth=2)
    plt.title("Decoder Training Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig("decoder_loss_graph.png")
    plt.close()
def test_generator(remove_start=True, toGenerate=10):

    DATASET = "../../datasets/zinc/dataset.csv"
    DECODER_PATH = "../models/drugs/decoder/decoder_model.pth"
    dataset_obj = SMILESDataset(DATASET, MAX_SMILES_LENGTH, fraction=0.01)
    
    model = SMILESGenerator(
        vector_size=MOLECULE_LENGTH,
        hidden_size=SMILES_GENERATOR_HIDDEN_SIZE,
        vocab_size=VOCAB_SIZE,
        max_smiles_length=MAX_SMILES_LENGTH
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    try:
        model.load_checkpoint(optimizer, DECODER_PATH)
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
    
    model.eval()
    
    all_results = []
    with torch.no_grad():
        for i in range(toGenerate):
            sample_vector, target_smiles = dataset_obj[random.randint(0, len(dataset_obj) - 1)]
            sample_vector = torch.tensor(sample_vector, dtype=torch.float32).to(device)
            
            generated_smiles = model.beam_search_decode(
                sample_vector,
                beam_width=3,
                remove_start=remove_start
            )
            original_smile = decode_smiles_token_sequence(target_smiles)

            print(f"--- Generation {i+1}/{toGenerate} ---")
            print(f"Target SMILES:    {original_smile}")
            print(f"Generated SMILES: {generated_smiles}")
            print("-----------------------------\n")
            
            all_results.append((target_smiles, generated_smiles))

    return all_results

if __name__ == "__main__":
    results = test_generator(remove_start=True, toGenerate=20)
