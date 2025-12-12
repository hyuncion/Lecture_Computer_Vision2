# src/classifiers/fc.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from tqdm import tqdm
from typing import Optional


class FCNet(nn.Module):
    """
    TWO layers = 1 hidden + 1 output
    in_dim ➜ hidden(4096) ➜ ReLU ➜ Dropout(0.4) ➜ out(num_classes)
    """
    def __init__(self, in_dim: int, hidden: int = 4096, num_classes: int = 20):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.4)
        self.fc2 = nn.Linear(hidden, num_classes)

        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity="linear")
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        return self.fc2(x)


class FCClassifier:
    """
    2-layer FC head (conv weights frozen)
      • 10 % validation split + early stopping
      • Adam + cosine LR decay
    """
    def __init__(self,
                 epochs: int = 50,
                 lr: float = 3e-4,
                 batch_size: int = 128,
                 patience: int = 6,
                 device: str = "cpu"):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net: Optional[FCNet] = None

    # ─────────────────────────── train ───────────────────────────
    def fit(self, X, y):
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        val_len = max(1, int(0.1 * len(X)))
        train_ds, val_ds = random_split(
            TensorDataset(X, y),
            [len(X) - val_len, val_len],
            generator=torch.Generator().manual_seed(0))

        self.net = FCNet(X.shape[1],
                         num_classes=int(y.max().item() + 1)).to(self.device)

        opt = optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        criterion = nn.CrossEntropyLoss()

        best_acc, patience_left = 0.0, self.patience
        best_state = None

        for epoch in range(self.epochs):
            self.net.train()
            loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()
                loss = criterion(self.net(xb), yb)
                loss.backward()
                opt.step()
            scheduler.step()

            val_acc = self._accuracy(val_ds)
            if val_acc > best_acc:
                best_acc, best_state = val_acc, self.net.state_dict()
                patience_left = self.patience
            else:
                patience_left -= 1
                if patience_left == 0:
                    break  # early stop

        self.net.load_state_dict(best_state)

    # ──────────────────────── helper ────────────────────────
    def _accuracy(self, dataset):
        self.net.eval()
        loader = DataLoader(dataset, batch_size=256)
        correct = total = 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.net(xb).argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        return correct / total

    # ───────────────────────── predict ─────────────────────────
    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.net(X).argmax(1).cpu().numpy()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path, in_dim, num_classes=20):
        self.net = FCNet(in_dim, num_classes=num_classes)
        self.net.load_state_dict(torch.load(path, map_location="cpu"))
