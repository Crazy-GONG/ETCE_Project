"""Phase 2A predictive models and deterministic validation-only selection."""
from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


def set_pilot_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))


class TabularMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers = []
        current = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(current, hidden), nn.ReLU(), nn.Dropout(dropout)))
            current = hidden
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values):
        return self.network(values).squeeze(-1)


class ActivityLSTM(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, time_dim, hidden_size, num_layers,
                 dropout, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(embedding_dim + time_dim, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout, bidirectional=bidirectional)
        self.head = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, activity_ids, time_features):
        embedded = self.embedding(activity_ids)
        output, _ = self.lstm(torch.cat((embedded, time_features), dim=-1))
        return self.head(output[:, -1, :]).squeeze(-1)


def _probabilities(model, loader, kind):
    model.eval()
    result = []
    with torch.no_grad():
        for batch in loader:
            if kind == 'mlp':
                logits = model(batch[0])
            else:
                logits = model(batch[0], batch[1])
            result.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(result)


def train_torch_candidate(kind, train_data, val_data, config, seed, vocabulary_size=None):
    """Only train updates weights; validation selects epoch and returns best checkpoint."""
    set_pilot_seed(seed)
    if kind == 'mlp':
        model = TabularMLP(train_data.tabular.shape[1], tuple(config['hidden_dims']), config['dropout'])
        train_ds = TensorDataset(torch.from_numpy(train_data.tabular),
                                 torch.from_numpy(train_data.labels.astype(np.float32)))
        val_ds = TensorDataset(torch.from_numpy(val_data.tabular),
                               torch.from_numpy(val_data.labels.astype(np.float32)))
    elif kind == 'lstm':
        model = ActivityLSTM(vocabulary_size, config['embedding_dim'], 2, config['hidden_size'],
                             config['num_layers'], config['dropout'], config['bidirectional'])
        train_ds = TensorDataset(torch.from_numpy(train_data.activity_sequence),
                                 torch.from_numpy(train_data.time_sequence),
                                 torch.from_numpy(train_data.labels.astype(np.float32)))
        val_ds = TensorDataset(torch.from_numpy(val_data.activity_sequence),
                               torch.from_numpy(val_data.time_sequence),
                               torch.from_numpy(val_data.labels.astype(np.float32)))
    else:
        raise ValueError(kind)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                              generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    loss_fn = nn.BCEWithLogitsLoss()
    best_score, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            if kind == 'mlp':
                logits, labels = model(batch[0]), batch[1]
            else:
                logits, labels = model(batch[0], batch[1]), batch[2]
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(labels)
        val_prob = _probabilities(model, val_loader, kind)
        val_score = f1_score(val_data.labels, (val_prob >= 0.5).astype(int), average='macro')
        history.append({'epoch': epoch, 'train_loss': total_loss / len(train_ds),
                        'validation_macro_f1': float(val_score)})
        if val_score > best_score + 1e-12:
            best_score, best_epoch = float(val_score), epoch
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= config['early_stopping_patience']:
            break
    model.load_state_dict(best_state)
    return model, {'best_epoch': best_epoch, 'best_validation_macro_f1': best_score,
                   'epochs_ran': len(history), 'history': history}


def predict_torch(model, kind, data, batch_size):
    if kind == 'mlp':
        dataset = TensorDataset(torch.from_numpy(data.tabular), torch.from_numpy(data.labels))
    else:
        dataset = TensorDataset(torch.from_numpy(data.activity_sequence),
                                torch.from_numpy(data.time_sequence), torch.from_numpy(data.labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return _probabilities(model, loader, kind)
