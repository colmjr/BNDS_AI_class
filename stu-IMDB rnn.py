import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import re
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
from collections import Counter
from tqdm import tqdm

# ================= 1. 配置与超参数 =================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_WORDS, MAX_LEN, BATCH_SIZE = 10000, 60, 64
LR = 0.0005

# ================= 2. 数据处理与划分 =================
def preprocess_imdb(file_path, sample_size=5000):
    df = pd.read_csv(file_path).sample(sample_size, random_state=42).reset_index(drop=True)
    def clean(text):
        text = text.lower().replace("<br />", " ")
        return re.sub(r"[^a-z ]", "", text).split()
    all_tokens = [w for text in df['review'] for w in clean(text)]
    vocab = {w: i+2 for i, (w, _) in enumerate(Counter(all_tokens).most_common(MAX_WORDS))}
    vocab["<PAD>"], vocab["<UNK>"] = 0, 1
    return df, vocab, clean

df, vocab, clean_fn = preprocess_imdb('IMDB Dataset.csv')

class IMDBDataset(Dataset):
    def __init__(self, df, vocab, max_len):
        self.reviews, self.labels = df['review'].values, [1 if l=='positive' else 0 for l in df['sentiment']]
        self.vocab, self.max_len = vocab, max_len
    def __len__(self): return len(self.reviews)
    def __getitem__(self, idx):
        ids = [self.vocab.get(w, 1) for w in clean_fn(self.reviews[idx])][:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids), torch.tensor(self.labels[idx], dtype=torch.float32)

# 划分训练集(80%)与测试集(20%)
full_dataset = IMDBDataset(df, vocab, MAX_LEN)
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, test_ds = random_split(full_dataset, [train_size, test_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ================= 3. 优化版双向 RNN 模型 =================
class OptimizedRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, 
                          num_layers=2, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name: nn.init.orthogonal_(param)

    def forward(self, x):
        embedded = self.embedding(x)
        out, h_n = self.rnn(embedded)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return torch.sigmoid(self.fc(hidden))

# ================= 4. 训练与实时评估逻辑 =================
model = OptimizedRNN(len(vocab), 64, 128).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCELoss()

history = {'train_acc': [], 'test_acc': []}

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts).squeeze()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

print("Starting Balanced RNN Training...")
for epoch in range(10):  # 增加轮数观察曲线
    model.train()
    train_correct, train_total = 0, 0
    for texts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        texts, labels = texts.to(DEVICE), labels.to(DEVICE)
        outputs = model(texts).squeeze()
        loss = criterion(outputs, labels)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
        
        train_correct += ((outputs > 0.5).float() == labels).sum().item()
        train_total += labels.size(0)
    
    train_acc = train_correct / train_total
    test_acc = evaluate(model, test_loader)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    print(f"-> Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

# ================= 5. 绘制最终曲线 =================
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), history['train_acc'], label='Train Accuracy', marker='o')
plt.plot(range(1, 11), history['test_acc'], label='Test Accuracy', marker='s')
plt.title('RNN Training vs Testing Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()