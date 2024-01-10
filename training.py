from dataloader import CustomDataset
from model import MyModel
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(dataset.vocabs)
embed_size = 300
hidden_size = 128
layer_size = 2
nums_class = 3
model = MyModel(vocab_size,
                embed_size,
                hidden_size,
                layer_size,
                nums_class
                ).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 10
for epoch in tqdm(range(epochs)):
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for batch, (features, label) in enumerate(dataloader):
        features = features.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output = model(features)
        loss = loss_fn(output, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(output, 1)
        correct_predictions += (predicted == label).sum().item()
        total_samples += label.shape[0]

        if batch % 100 == 0:
            avg_loss = total_loss / (batch + 1)
            accuracy = correct_predictions / total_samples * 100.0
            print(f"Epoch: {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")