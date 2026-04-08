import os
import yaml
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(pred, target)

class AKIDataset(Dataset):
    def __init__(self, csv_path, config, scalers=None, is_train=False):
        self.config = config
        self.is_train = is_train
        self.max_len = config['data']['max_seq_len']
        self.dynamic_cols = config['data']['dynamic_cols']
        self.static_cols = config['data']['static_cols']
        self.target_col = config['data']['target_col']
        self.group_col = config['data']['group_col']
        self.dynamic_dim = len(self.dynamic_cols)
        self.static_dim = len(self.static_cols)
        aug_conf = config['data'].get('augmentation', {})
        self.enable_feature_masking = aug_conf.get('enable_feature_masking', False)
        self.feature_mask_prob = aug_conf.get('feature_mask_prob', 0.5)
        masked_col_names = aug_conf.get('masked_cols', [])
        self.feature_mask_indices = [self.dynamic_cols.index(col) for col in masked_col_names if
                                     col in self.dynamic_cols]
        self.enable_time_masking = aug_conf.get('enable_time_masking', False)
        self.time_mask_prob = aug_conf.get('time_mask_prob', 0.5)
        self.max_mask_steps = aug_conf.get('max_mask_steps', 2)

        df = pd.read_csv(csv_path)
        all_cols = self.dynamic_cols + self.static_cols
        df[all_cols] = df[all_cols].fillna(0)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        if is_train:
            self.dyn_scaler = StandardScaler()
            self.stat_scaler = StandardScaler()
            df[self.dynamic_cols] = self.dyn_scaler.fit_transform(df[self.dynamic_cols])
            df[self.static_cols] = self.stat_scaler.fit_transform(df[self.static_cols])
            self.scalers = {'dyn': self.dyn_scaler, 'stat': self.stat_scaler}
        else:
            self.scalers = scalers
            if self.scalers is not None:
                df[self.dynamic_cols] = self.scalers['dyn'].transform(df[self.dynamic_cols])
                df[self.static_cols] = self.scalers['stat'].transform(df[self.static_cols])

        self.samples = []
        self.labels = []
        grouped = df.groupby(self.group_col)

        for stay_id, group in tqdm(grouped, desc="Generating Windows"):
            dyn_data = group[self.dynamic_cols].values.astype(np.float32)
            stat_data = group[self.static_cols].values.astype(np.float32)
            labels_data = group[self.target_col].values
            num_steps = len(dyn_data)
            for i in range(num_steps):
                current_label = labels_data[i]
                if current_label == -1: continue
                end_idx = i + 1
                start_idx = end_idx - self.max_len
                static_vec = stat_data[i]
                if start_idx >= 0:
                    window = dyn_data[start_idx:end_idx, :]
                    mask = np.ones(self.max_len)
                else:
                    valid_data = dyn_data[0:end_idx, :]
                    pad_len = self.max_len - valid_data.shape[0]
                    padding = np.zeros((pad_len, self.dynamic_dim))
                    window = np.concatenate([padding, valid_data], axis=0)
                    mask = np.concatenate([np.zeros(pad_len), np.ones(valid_data.shape[0])])
                self.samples.append((window.astype(np.float32), static_vec, mask.astype(np.float32)))
                self.labels.append(current_label)

        self.labels = np.array(self.labels)
        self.indices = np.arange(len(self.labels))
        if is_train and config['data'].get('use_oversampling', False):
            self._apply_oversampling()

    def _apply_oversampling(self):
        pos_indices = np.where(self.labels == 1)[0]
        neg_indices = np.where(self.labels == 0)[0]
        if len(pos_indices) > 0 and len(pos_indices) < len(neg_indices):
            diff = len(neg_indices) - len(pos_indices)
            oversampled = np.random.choice(pos_indices, size=diff, replace=True)
            self.indices = np.concatenate([pos_indices, neg_indices, oversampled])
            np.random.shuffle(self.indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        dyn_window, static_vec, mask = self.samples[real_idx]
        label = self.labels[real_idx]
        dyn_window = dyn_window.copy()
        if self.is_train:
            if self.enable_feature_masking and random.random() < self.feature_mask_prob:
                dyn_window[:, self.feature_mask_indices] = 0.0
            if self.enable_time_masking and random.random() < self.time_mask_prob:
                seq_len = dyn_window.shape[0]
                mask_len = random.randint(1, self.max_mask_steps)
                start_step = random.randint(0, seq_len - mask_len)
                dyn_window[start_step: start_step + mask_len, :] = 0.0
        return (torch.FloatTensor(dyn_window),
                torch.FloatTensor(static_vec),
                torch.FloatTensor([label]),
                torch.FloatTensor(mask))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class DualStreamAKIModel(nn.Module):
    def __init__(self, config):
        super(DualStreamAKIModel, self).__init__()
        model_conf = config['model']
        data_conf = config['data']
        self.dyn_input_dim = len(data_conf['dynamic_cols'])
        self.d_model = model_conf['d_model']
        self.nhead = model_conf['nhead']
        self.num_layers = model_conf['num_layers']
        self.dropout_p = model_conf['dropout']
        self.embedding = nn.Linear(self.dyn_input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=data_conf['max_seq_len'] + 50)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=self.nhead,
            dim_feedforward=model_conf['dim_feedforward'],
            dropout=self.dropout_p, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        self.stat_input_dim = len(data_conf['static_cols'])
        self.stat_hidden_dim = model_conf.get('static_hidden_dim', 32)
        self.static_encoder = nn.Sequential(
            nn.Linear(self.stat_input_dim, self.stat_hidden_dim),
            nn.BatchNorm1d(self.stat_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.stat_hidden_dim, self.d_model)
        )
        self.gate_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model * 2, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(64, 1)
        )

    def forward(self, dyn, stat, mask=None):
        x_dyn = self.embedding(dyn)
        x_dyn = x_dyn.permute(1, 0, 2)
        x_dyn = self.pos_encoder(x_dyn)
        x_dyn = x_dyn.permute(1, 0, 2)
        key_padding_mask = (mask == 0).bool() if mask is not None else None
        out_dyn = self.transformer_encoder(x_dyn, src_key_padding_mask=key_padding_mask)
        emb_dyn = out_dyn[:, -1, :]
        emb_stat = self.static_encoder(stat)
        gate = self.gate_net(emb_stat)
        gated_dyn = emb_dyn * gate
        combined = torch.cat([gated_dyn, emb_stat], dim=1)
        logits = self.classifier(combined)
        return logits.squeeze(1)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['train']['device'])
        self.save_dir = config['train']['save_dir']
        if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
        self.train_dataset = AKIDataset(config['data']['train_path'], config, is_train=True)
        self.val_dataset = AKIDataset(config['data']['val_path'], config, scalers=self.train_dataset.scalers,
                                      is_train=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=config['dataloader']['batch_size'], shuffle=True,
                                       num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=config['dataloader']['batch_size'], shuffle=False,
                                     num_workers=0)
        self.model = DualStreamAKIModel(config).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=float(config['train']['lr']), weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        self.criterion = LabelSmoothingLoss(smoothing=0.1)
        self.val_criterion = nn.BCEWithLogitsLoss()
        self.best_auc = 0.0
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'epoch': []}

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        loop = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.config['train']['epochs']}")
        for dyn, stat, y, mask in loop:
            dyn, stat = dyn.to(self.device), stat.to(self.device)
            y, mask = y.to(self.device), mask.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(dyn, stat, mask)
            loss = self.criterion(logits, y.squeeze(-1))
            loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch + loop.n / len(self.train_loader))
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])
        return total_loss / len(self.train_loader)

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for dyn, stat, y, mask in self.val_loader:
                dyn, stat = dyn.to(self.device), stat.to(self.device)
                y, mask = y.to(self.device), mask.to(self.device)
                logits = self.model(dyn, stat, mask)
                loss = self.val_criterion(logits, y.squeeze(-1))
                total_loss += loss.item()
                all_probs.extend(torch.sigmoid(logits).cpu().numpy())
                all_labels.extend(y.squeeze(-1).cpu().numpy())
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.5
        return total_loss / len(self.val_loader), auc

    def plot_training_curve(self):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['epoch'], self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['epoch'], self.history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss Curve', fontname='Times New Roman')
        plt.subplot(1, 2, 2)
        plt.plot(self.history['epoch'], self.history['val_auc'], label='Val AUC')
        plt.legend()
        plt.title('AUC Curve', fontname='Times New Roman')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_curve.png"))
        plt.close()

    def run(self):
        for epoch in range(self.config['train']['epochs']):
            t_loss = self.train_epoch(epoch)
            v_loss, v_auc = self.validate(epoch)
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(t_loss)
            self.history['val_loss'].append(v_loss)
            self.history['val_auc'].append(v_auc)
            self.plot_training_curve()
            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'scalers': self.train_dataset.scalers,
                'config': self.config
            }
            torch.save(state, os.path.join(self.save_dir, "last.pth"))
            if v_auc > self.best_auc:
                self.best_auc = v_auc
                torch.save(state, os.path.join(self.save_dir, "best.pth"))

def main():
    if not os.path.exists('config.yaml'): raise FileNotFoundError("config.yaml not found!")
    with open('config.yaml', 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    Trainer(config).run()

if __name__ == "__main__":
    main()
