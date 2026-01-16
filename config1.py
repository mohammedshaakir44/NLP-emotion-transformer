from pathlib import Path

def get_config():
    return {
        "batch_size": 16,
        "num_epochs": 10,
        "lr": 10**-5,
        "seq_len": 128,
        "d_model": 512,
        "lang_src": "en", # DailyDialog is English
        "num_classes": 7, # 0: no emotion, 1: anger, 2: disgust, 3: fear, 4: happiness, 5: sadness, 6: surprise
        "model_folder": "weights_emotion",
        "model_basename": "emomodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_emotion.json",
        "experiment_name": "runs/emotion_model"
    }

def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)