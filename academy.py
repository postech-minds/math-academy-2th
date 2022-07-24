import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def load_cycles(self, fpath):
        """
        Args:
            fpath (_type_): input_matlabinfo

        Returns:
            _type_: _description_
        """
        import json
        f = open(fpath, 'r')
        l = len(f.readlines())
        f = open(fpath, 'r')

        # s = f.readlines()
        s = []
        for i in range(l):
            s.append(f.readline())

        # dim1 info
        d1s = []
        for i in range(l):
            if s[i] == 'Dimension: 1\n':
                start_inx = i+1
        for j in range(start_inx, l):
            if s[j] == 'Dimension: 2\n' or s[j] == '\n':
                end_inx = j
                break
        for k in range(start_inx, end_inx):
            d1s.append(s[k])

        # get cycle info
        c_info = []
        for i in range(len(d1s)):
            ci_int, ci = d1s[i].split(': ')
            ci = ci.split('\n')[0]
            ci_edge = ci.split(' + ')
            for j in range(len(ci_edge)):
                if ci_edge[j][0] == '-':
                    ci_edge[j] = json.loads(ci_edge[j][1:])
                    ci_edge[j] = ci_edge[j][::-1]
                else:
                    ci_edge[j] = json.loads(ci_edge[j])
            ci_node = []
            for k in range(len(ci_edge)):
                for node in ci_edge[k]:
                    if not(node in ci_node):
                        ci_node.append(node)
            ci_node.sort()
            c_info.append([ci_node, ci_int, ci_edge])
        c = []
        e = []
        for i in range(len(c_info)):
            c.append(c_info[i][0])
            e.append(c_info[i][2])

        self.cycles = c


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional=False):
        super(LSTM, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bidirectional=bidirectional)
        if bidirectional:
            hidden_size *= 2
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = x.squeeze(1)
        x = self.head(x)

        return x


class OverlapMatrixModel(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output, num_nodes, dropout=0.3):
        def fanin_(size):
            """
            Take a look "Experiment Details" in the DDPG paper
            code from: https://blog.paperspace.com/physics-control-tasks-with-deep-reinforcement-learning/
            """
            fan_in = size[0]
            weight = 2. / math.sqrt(fan_in)
            return torch.Tensor(size).uniform_(-weight, weight)
        super(OverlapMatrixModel, self).__init__()
        self.fc1 = nn.Linear(dim_input, dim_hidden)
        self.fc1.weight.data = fanin_(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(dim_hidden, 2 * dim_hidden)
        self.fc2.weight.data = fanin_(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(2 * dim_hidden, dim_output)
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.num_nodes = num_nodes
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view((-1, self.num_nodes))

        return x


def train(model, X, y, task='regression', epochs=100, lr=0.001, weight=None):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if task == 'regression' else nn.CrossEntropyLoss(weight=weight)

    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()

    if X.ndim == 1:
        X = X.unsqueeze(1)

    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    if y.ndim == 1 and task == 'regression':
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
def forecast(model, x, n, recurrent=False):
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()

    if not recurrent:
        pred = []
        for i in range(n):
            out = model(x).detach()
            pred.append(out.numpy())
            x = torch.cat((x[1:], out))

        return np.array(pred)

    else:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 0:
            x = x.unsqueeze(0)
        if x.ndim == 1:
            x = x.unsqueeze(0)
            x = x.unsqueeze(0)

        for i in range(n):
            x_next = model(x)
            x = torch.cat((x, x_next[-1].unsqueeze(0).unsqueeze(0)))

        return x.numpy()[:, 0, :]
