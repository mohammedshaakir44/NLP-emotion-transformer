import torch
from tokenizers import Tokenizer
from model1 import build_transformer_classifier
from config1 import get_config, get_weights_file_path

def predict_emotion(input_text):
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Tokenizer
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])
    
    # 2. Build Model and Load Weights
    # We use '09' for the last epoch (assuming 10 epochs total)
    model = build_transformer_classifier(
        tokenizer.get_vocab_size(), 
        config['seq_len'], 
        config['num_classes']
    ).to(device)
    
    model_path = get_weights_file_path(config, "09")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3. Preprocess Input
    tokens = tokenizer.encode(input_text).ids
    # Pad or truncate manually to match training seq_len
    if len(tokens) < config['seq_len']:
        tokens += [tokenizer.token_to_id("[PAD]")] * (config['seq_len'] - len(tokens))
    else:
        tokens = tokens[:config['seq_len']]
        
    input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
    mask = (input_tensor != tokenizer.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(device)

    # 4. Predict
    # Label Map for DailyDialog
    labels = {0: "No emotion", 1: "Anger", 2: "Disgust", 3: "Fear", 4: "Happiness", 5: "Sadness", 6: "Surprise"}
    
    with torch.no_grad():
        output = model(input_tensor, mask)
        prediction = torch.argmax(output, dim=1).item()
        
    return labels.get(prediction, "Unknown")

if __name__ == "__main__":
    print("--- Emotion Detector ---")
    while True:
        user_input = input("\nEnter a sentence (or 'q' to quit): ")
        if user_input.lower() == 'q':
            break
        emotion = predict_emotion(user_input)
        print(f"Predicted Emotion: {emotion}")