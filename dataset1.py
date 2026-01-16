import torch
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, flat_ds, tokenizer, seq_len):
        self.ds = flat_ds
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_token = tokenizer.token_to_id("[PAD]")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        text = item['text']
        label = item['label']

        tokens = self.tokenizer.encode(text).ids
        num_padding = self.seq_len - len(tokens)

        if num_padding < 0:
            tokens = tokens[:self.seq_len]
            num_padding = 0

        input_tensor = torch.cat([
            torch.tensor(tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * num_padding, dtype=torch.int64)
        ])

        return {
            "input": input_tensor,
            "mask": (input_tensor != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "label": torch.tensor(label, dtype=torch.long)
        }