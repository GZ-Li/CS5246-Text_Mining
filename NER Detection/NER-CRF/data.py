import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

def align_label(texts, labels, labels_to_ids):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=512, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
                
        previous_word_idx = word_idx

    return label_ids


class NERDataset(torch.utils.data.Dataset):

    def __init__(self, df):

        corpus = df['corpus'].values.tolist()

        labels = [i.split(" ") for i in df['label'].values.tolist()]

        # Check how many labels are there in the dataset
        unique_labels = set()

        for lb in labels:
            [unique_labels.add(i) for i in lb if i not in unique_labels]

        self.labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
        self.ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
        
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = 512, truncation=True, return_tensors="pt") for i in corpus]
        self.labels = [align_label(i,j, self.labels_to_ids) for i,j in zip(corpus, labels)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def get_label(self, idx):

        return self.ids_to_labels[idx]
    
    def get_label_idx(self, label):

        return self.labels_to_ids[label]

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels


# Method 2: Generate a Dataset class
def get_data(BATCH_SIZE, train_datapath, test_datapath):

    """
    Generate dataloader given training data and testing data
    
    Parameters:
        BATCH_SIZE: an integer
        train_datapath: a string
        test_datapath: a string
    
    Returns:
        train_loader, test_loader, val_loader
    """

    train_df = pd.read_csv(train_datapath)
    df = pd.read_csv(test_datapath)

    # Further split the test data into test data and validation data
    test_df, val_df = train_test_split(df, test_size=0.5)

    # Dataloader for training dataset
    train_dataset = NERDataset(train_df)
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size= BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        drop_last = True,
    )

    # Dataloader for testing dataset
    test_dataset = NERDataset(test_df)
    test_loader = DataLoader(
        dataset=test_dataset, 
        batch_size= BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        drop_last = True,
    )

    # Dataloader for validation dataset
    val_dataset = NERDataset(val_df)
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size= BATCH_SIZE, 
        shuffle=True, 
        num_workers=2,
        drop_last = True,
    )

    return train_loader, test_loader, val_loader


# Example of using get_data function
# train_loader, test_loader, val_loader = get_data(32, training data csv file path, testing data csv file path)

# batch1 = next(iter(train_loader))
# batch2 = next(iter(test_loader))
# batch3 = next(iter(val_loader))

# print(batch1, batch2, batch3)
