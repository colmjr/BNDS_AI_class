#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
优化版 LSTM 文本生成（防过拟合 + 教学版）
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ------------------------------
# 1. 设备
# ------------------------------
def get_device():
    try:
        import torch_musa
        if torch_musa.is_available():
            return torch.device("musa")
    except:
        pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = get_device()
print("Device:", device)

# ------------------------------
# 2. 超参数（防过拟合版本）
# ------------------------------
DATA_PATH = "tiny_shakespeare.txt"

SEQUENCE_LEN = 30        
BATCH_SIZE = 64
VAL_SPLIT = 0.1
USE_SUBSET = True
SUBSET_SIZE = 150000

EMBED_DIM = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.2

LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5     

NUM_EPOCHS = 30
PATIENCE = 5

GENERATION_LENGTH = 200
GENERATION_START = "KING:"

SAVE_BEST_MODEL = True
MODEL_PATH = "best_lstm.pth"

# ------------------------------
# 3. 数据
# ------------------------------
def load_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def build_vocab(text):
    chars = sorted(set(text))
    return {c:i for i,c in enumerate(chars)}, {i:c for i,c in enumerate(chars)}, len(chars)

def encode(text, vocab):
    return torch.tensor([vocab[c] for c in text], dtype=torch.long)

class DatasetSeq(Dataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

# ------------------------------
# 4. 模型
# ------------------------------
class LSTMModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, EMBED_DIM)
        self.lstm = nn.LSTM(
            EMBED_DIM,
            HIDDEN_DIM,
            NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT
        )
        self.fc = nn.Linear(HIDDEN_DIM, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        out, _ = self.lstm(x)
        return self.fc(out)

# ------------------------------
# 5. 采样
# ------------------------------
def sample(logits, temperature=0.7):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()

# ------------------------------
# 6. 训练
# ------------------------------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out.view(-1, vocab_size), y.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        total += loss.item()
    return total / len(loader)

@torch.no_grad()
def eval_epoch(model, loader, loss_fn):
    model.eval()
    total = 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out.view(-1, vocab_size), y.view(-1))
        total += loss.item()
    return total / len(loader)

# ------------------------------
# 7. 生成
# ------------------------------
def generate(model, start, vocab, ivocab, length=200):
    model.eval()
    x = torch.tensor([[vocab[c] for c in start]], device=device)
    hidden = None
    out_text = start

    with torch.no_grad():
        for _ in range(length):
            logits = model(x)[:, -1, :]
            idx = sample(logits[0])
            out_text += ivocab[idx]
            x = torch.tensor([[idx]], device=device)

    return out_text

# ------------------------------
# 8. 主程序
# ------------------------------
if __name__ == "__main__":

    text = load_text(DATA_PATH)

    if USE_SUBSET:
        text = text[:SUBSET_SIZE]

    vocab, ivocab, vocab_size = build_vocab(text)
    data = encode(text, vocab)

    train_size = int(len(data) * (1 - VAL_SPLIT))
    train_data = data[:train_size]
    val_data = data[train_size:]

    train_ds = DatasetSeq(train_data, SEQUENCE_LEN)
    val_ds = DatasetSeq(val_data, SEQUENCE_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = LSTMModel(vocab_size).to(device)

    opt = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val = 1e9
    patience = 0

    train_loss_list = []
    val_loss_list = []

    print("Training...")

    for epoch in range(NUM_EPOCHS):

        train_loss = train_epoch(model, train_loader, opt, loss_fn)
        val_loss = eval_epoch(model, val_loader, loss_fn)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        print(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience += 1

        if patience >= PATIENCE:
            print("Early stopping triggered")
            break

    # ------------------------------
    # 可视化
    # ------------------------------
    plt.plot(train_loss_list, label="train")
    plt.plot(val_loss_list, label="val")
    plt.legend()
    plt.title("Loss Curve")
    plt.show()

    # ------------------------------
    # 生成文本
    # ------------------------------
    model.load_state_dict(torch.load(MODEL_PATH))

    print(generate(model, GENERATION_START, vocab, ivocab))