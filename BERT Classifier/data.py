from torch.utils.data import Dataset, DataLoader
import torch
from transformers import BertTokenizer
import pandas as pd

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class ContextDataset(Dataset):
    def __init__(self, x_inputs, y_inputs, tokenizer, max_len):
        self.x = x_inputs
        self.y = y_inputs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        label = torch.tensor(self.y[index], dtype=torch.long)
        text = self.x[index]

        embedding = self.tokenizer.encode_plus(
            str(text),
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            "text": text,
            "input_ids": embedding['input_ids'].flatten(),
            "attention_mask": embedding['attention_mask'].flatten(),
            "token_type_ids": embedding['token_type_ids'].flatten(),
            "labels": label
        }


def get_data(args, max_len, batch_size, type = 'train'):
    if type == 'train':
        train = pd.read_csv(f'data/{args.dataset}/train.csv')
    elif type == 'val':
        train = pd.read_csv(f"data/{args.dataset}/val.csv")
    else:
        train = pd.read_csv(f"data/{args.dataset}/test.csv")
    X_train = train['X']
    Y_train = train['Y']
    train_dataset = ContextDataset(X_train, Y_train, tokenizer, max_len)
    train_loader = DataLoader(dataset=train_dataset, batch_size= batch_size, shuffle=True, num_workers=args.num_workers)
    return train_loader

class Recorder:
    def __init__(self):
        self.best_valid_epoch = 0
        self.best_valid_recall = 0
        self.patience = 0
