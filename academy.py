import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import trange


def train(model, X, y, task='regression', epochs=100, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss()

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()

    if X.ndim == 1:
        X = X.unsqueeze(1)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    if y.ndim == 1:
        y = y.unsqueeze(1)

    t = trange(epochs, desc="Log")
    for e in t:
        out = model(X)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        t.set_description(f"[Epoch {e+1}/{epochs}] loss: {loss.item():.4f}")


@torch.no_grad()
def forecast(model, x, n):
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    pred = []
    for i in range(n):
        out = model(x).detach()
        pred.append(out.numpy())
        x = torch.cat((x[1:], out))

    return np.array(pred)
