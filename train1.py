import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from tqdm import tqdm
import ast

# Internal imports
from model1 import build_transformer_classifier
from config1 import get_config, get_weights_file_path
from dataset1 import EmotionDataset

def parse_kaggle_string(data_str):
    try:
        return ast.literal_eval(data_str)
    except:
        return str(data_str).strip().split()

def get_all_sentences(flat_data):
    for item in flat_data:
        yield item['text']

def get_or_build_tokenizer(config, flat_data):
    tokenizer_path = Path(config['tokenizer_file'])
    if not tokenizer_path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(flat_data), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def train_model():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # FIXED PATHS
    data_files = {
        "train": r"C:\Users\Shaakir\OneDrive\Documents\dialog emotion\train.csv", 
        "validation": r"C:\Users\Shaakir\OneDrive\Documents\dialog emotion\validation.csv",
        "test": r"C:\Users\Shaakir\OneDrive\Documents\dialog emotion\test.csv"
    }
    
    raw_datasets = load_dataset('csv', data_files=data_files)
    
    # 1. PROCESS TRAIN DATA
    train_flat = [] # This is the variable name we will use consistently
    for row in raw_datasets['train']:
        # Changed 'dialogue' to 'dialog' to match your CSV header
        utts = parse_kaggle_string(row['dialog']) 
        emotions = parse_kaggle_string(row['emotion'])
        for t, l in zip(utts, emotions):
            train_flat.append({'text': str(t), 'label': int(l)})

    # 2. PROCESS TEST DATA
    test_flat = []
    for row in raw_datasets['test']:
        utts = parse_kaggle_string(row['dialog']) 
        emotions = parse_kaggle_string(row['emotion'])
        for t, l in zip(utts, emotions):
            test_flat.append({'text': str(t), 'label': int(l)})

    # Pass train_flat (not flat_data) to the tokenizer
    tokenizer = get_or_build_tokenizer(config, train_flat)
    
    # Create DataLoaders
    train_loader = DataLoader(EmotionDataset(train_flat, tokenizer, config['seq_len']), batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(EmotionDataset(test_flat, tokenizer, config['seq_len']), batch_size=config['batch_size'], shuffle=False)

    model = build_transformer_classifier(tokenizer.get_vocab_size(), config['seq_len'], config['num_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Training Loop
    for epoch in range(config['num_epochs']):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch:02d}")
        for batch in batch_iterator:
            inputs, masks, labels = batch['input'].to(device), batch['mask'].to(device), batch['label'].to(device)
            outputs = model(inputs, masks)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
        
        torch.save(model.state_dict(), get_weights_file_path(config, f"{epoch:02d}"))

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            inputs, masks, labels = batch['input'].to(device), batch['mask'].to(device), batch['label'].to(device)
            outputs = model(inputs, masks)
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
    print(f"Final Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train_model()