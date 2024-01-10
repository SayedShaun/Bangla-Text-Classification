import torch
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
import string
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence



class CustomDataset(Dataset):
    def __init__(self, file_path, column, target):
        self.df = pd.read_excel(file_path)
        self.sentences = self.df[column]
        self.label = self.df[target]
        self.vocabs = build_vocab_from_iterator(
            self.token_genarator(self.sentences)
        )
        extra_tokens = ["<PAD>", "<UNK>"]
        for token in extra_tokens:
            self.vocabs.append_token(token)

    def token_genarator(self, sentences):
        for text in sentences:
            clean_text = "".join(
                [word for word in text 
                 if word not in string.punctuation]
            )
            tokens = word_tokenize(clean_text)
            yield tokens

    def text_to_sequences(self, sentences):
        sequence = [
            self.vocabs[token] if token in self.vocabs
            else self.vocabs["<UNK>"]
            for token in word_tokenize(sentences)
            ]
        return sequence
    
    @staticmethod
    def collate_fn(batch):
        texts = [item[0] for item in batch]
        label = [item[1] for item in batch]
        padded = pad_sequence(texts, padding_value=0)
        return padded, torch.tensor(label) 

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, index):
        features = self.sentences[index]
        sequence = self.text_to_sequences(features)
        label = self.label[index]
        return torch.tensor(sequence), torch.tensor(label)




if __name__=="__main__":
    dataset = CustomDataset(
        file_path="Data/BanglaNewsText.xlsx",
        column="Data",
        target="Label"
    )
    dataloader = DataLoader(
        dataset, batch_size=32,
        shuffle=True, 
        collate_fn=dataset.collate_fn
    )

    text, label = next(iter(dataloader))
    print(text, label)
