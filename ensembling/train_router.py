import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
import json
import pandas as pd

ROUTER_MODEL_PATH = "./task_router.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a text embedding model
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")

class TaskRouter(nn.Module):
    def __init__(self, input_dim):
        super(TaskRouter, self).__init__()
        self.fc = nn.Linear(input_dim, 2)  # Output logits for Task 1 (0) and Task 2 (1)

    def forward(self, x):
        return self.fc(x)  # No softmax (CrossEntropyLoss will handle it)

router = TaskRouter(input_dim=384).to(device)  # MiniLM produces 384-dim embeddings

def load_json_WVS_dataset(json_path, label):
    texts = []
    with open(json_path, 'r') as f:
            for line in f:
                data = json.loads(line.strip()) 
                texts.append(data['messages'][0]['content'] + " " + data['messages'][1]['content'])
    labels = [label] * len(texts)
    return texts, labels

task1_texts, task1_labels = load_json_WVS_dataset("../data/WVQ_China_Train.jsonl", label=0)

df = pd.read_excel("../data/finetune/prompts/text-infilling.xlsx")
task2_texts = df["Prompt"].tolist()
task2_labels = [1] * len(task2_texts)

# Combine datasets
train_texts = task1_texts + task2_texts
train_labels = task1_labels + task2_labels

# Define Dataset class for loading text samples
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_embedding = text_encoder.encode(self.texts[idx], convert_to_tensor=True).to(device)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return text_embedding, label

train_dataset = TextDataset(train_texts, train_labels)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(router.parameters(), lr=0.001)

def train_router(epochs=5):
    router.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for embeddings, labels in train_loader:
            optimizer.zero_grad()
            outputs = router(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

train_router()

torch.save(router.state_dict(), ROUTER_MODEL_PATH)
print(f"Router model saved to {ROUTER_MODEL_PATH}")