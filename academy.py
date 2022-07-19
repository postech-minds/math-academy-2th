import numpy as np
import torch
import torch.nn as nn
from tqdm.notebook import trange


class Music:
    def __init__(self, cycles=None, music=None, nodelist=None):
        self.cycles = cycles
        self.music = music
        self.nodelist = nodelist

    def __repr__(self):
        repr = 'Music( '
        if self.music is not None:
            repr += f'length={self.length} '
        if self.nodelist is not None:
            repr += f'num_nodes={self.num_nodes} '
        if self.cycles is not None:
            repr += f'num_cycles={self.num_cycles} '
        repr += ')'
        return repr

    @property
    def length(self):
        return len(self.music)

    @property
    def num_nodes(self):
        return len(self.nodelist)

    @property
    def num_cycles(self):
        return len(self.cycles)

    @property
    def node2idx(self):
        return {node: idx for idx, node in enumerate(self.nodelist)}

    @property
    def idx2node(self):
        return {idx: node for idx, node in enumerate(self.nodelist)}

    @property
    def indexed_music(self):
        return np.array([self.node2idx[node] for node in self.music])

    def _split_music_by_beat(self):
        col = []
        inx_s = 0
        sumleng = 0
        for i, node in enumerate(self.music):
            leng = round((node-int(node))*100)
            sumleng += leng
            if sumleng == 36:
                inx_e = i+1
                col.append(self.music[inx_s:inx_e])
                inx_s = i+1
                sumleng = 0

        col_inx = []
        for beat in col:
            col_inx.append(list(map(lambda x: self.node2idx[x], beat)))

        return col_inx

    def beat_start_indices(self):
        col_inx = self._split_music_by_beat()
        length_list = list(map(len, col_inx))

        cum_sum = 0
        start_indices = []
        for length in length_list:
            start_indices.append(cum_sum)
            cum_sum += length

        return start_indices

    def _node_overlap(self):
        matrix = []
        for cycle in self.cycles:
            row = []
            for node in self.music:
                inx = self.node2idx[node]
                if inx in cycle:
                    row.append(cycle.index(inx)+1)
                else:
                    row.append(0)
            matrix.append(row)

        for q in [3]:
            mat = []
            for j, row in enumerate(matrix):
                new, leng = [], 0
                # construct new row
                for i, node in enumerate(row):
                    if node == 0:
                        if leng == 0:
                            new.append(0)
                        else:
                            if leng > q:
                                new += row[i-leng:i]+[0]
                            else:
                                new += [0]*leng+[0]
                            leng = 0
                    else:
                        leng += 1
                if leng != 0:
                    if leng > q:
                        new += row[len(row)-leng:]
                    else:
                        new += [0]*leng
                mat.append(new)

        return mat

    @property
    def overlap_matrix(self):
        mat = self._node_overlap()
        overlap_matrix = np.zeros((self.length, self.num_cycles))
        for i in range(self.length):
            for j in range(self.num_cycles):
                if mat[j][i] != 0:
                    overlap_matrix[i, j] = 1

        return overlap_matrix

    def load_music(self, fpath):
        self.music = np.loadtxt(fpath)

    def load_nodelist(self, fpath):
        # self.nodelist = sorted(np.loadtxt(fpath))
        self.nodelist = np.loadtxt(fpath)


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
