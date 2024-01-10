from nltk.tokenize import word_tokenize
import torch
from model import MyModel
from dataloader import CustomDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = CustomDataset(
    file_path="Data/BanglaNewsText.xlsx",
    column="Data", 
    target="Label")

def prediction(text):
    vocab_size = len(dataset.vocabs)
    embed_size = 300
    hidden_size = 128
    layer_size = 2
    nums_class = 3
    model = MyModel(vocab_size, 
                    embed_size, 
                    hidden_size, 
                    layer_size, 
                    nums_class).to(device)
    
    model.load_state_dict(
        torch.load("Saved Model/saved_model.pth", 
        map_location=device
        )
    )
    model.eval()
    sequence = torch.tensor(dataset.text_to_sequences(text))

    output = model(sequence)
    prediction = torch.argmax(output)
    
    if prediction == 0:
        print("Job Releted News")
    elif prediction == 1:
        print("Politics Releted News")
    elif prediction == 2:
        print("Education Related News")


text = "প্রায় এক বছর ধরে করোনা মহামারীর জন্য শিক্ষার অনেক অংশ অনলাইনে চলছে।"
prediction(text)