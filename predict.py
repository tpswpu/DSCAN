import os
import yaml
import time
import torch
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from thop import profile, clever_format
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
    f1_score, recall_score, accuracy_score, roc_auc_score, precision_score
)
from torch.utils.data import DataLoader

from main import DualStreamAKIModel, AKIDataset

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['predict']['device'])
        self.output_dir = config['predict']['output_dir']

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        weight_path = config['predict'].get('weight_path')
        if not weight_path or not os.path.exists(weight_path):
            weight_path = os.path.join(config['train']['save_dir'], "best.pth")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight file not found: {weight_path}")

        ckpt = torch.load(weight_path, map_location=self.device, weights_only=False)
        scalers = ckpt.get('scalers', None)
        self.test_dataset = AKIDataset(config['data']['test_path'], config, scalers=scalers, is_train=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=config['dataloader']['batch_size'], shuffle=False)

        self.model = DualStreamAKIModel(config).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()

    def calculate_flops(self):
        dyn_dim = len(self.config['data']['dynamic_cols'])
        seq_len = self.config['data']['max_seq_len']
        dummy_dyn = torch.randn(1, seq_len, dyn_dim).to(self.device)
        stat_dim = len(self.config['data']['static_cols'])
        dummy_stat = torch.randn(1, stat_dim).to(self.device)
        dummy_mask = torch.ones(1, seq_len).to(self.device)
        try:
            flops, params = profile(self.model, inputs=(dummy_dyn, dummy_stat, dummy_mask), verbose=False)
            return clever_format([flops, params], "%.3f")
        except:
            return "N/A", "N/A"

    def calculate_metrics(self, y_true, y_probs, threshold, strategy_name):
        y_pred = (y_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            'Strategy': strategy_name,
            'Threshold': threshold,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Sensitivity': recall_score(y_true, y_pred, zero_division=0),
            'Specificity': tn / (tn + fp + 1e-8),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'F1': f1_score(y_true, y_pred, zero_division=0),
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
        }

    def evaluate(self):
        flops_str, params_str = self.calculate_flops()
        all_probs, all_labels = [], []
        start_time = time.time()

        with torch.no_grad():
            for dyn, stat, y, mask in tqdm(self.test_loader, desc="Inference"):
                dyn, stat = dyn.to(self.device), stat.to(self.device)
                mask = mask.to(self.device)
                logits = self.model(dyn, stat, mask)
                probs = torch.sigmoid(logits)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.numpy())

        total_time = time.time() - start_time
        avg_time = (total_time / len(self.test_dataset)) * 1000
        y_true = np.array(all_labels).ravel()
        y_probs = np.array(all_probs)
        auc_score = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5

        thresholds_dict = {}
        if len(np.unique(y_true)) > 1:
            fpr, tpr, roc_thresh = roc_curve(y_true, y_probs)
            prec_pr, rec_pr, pr_thresh = precision_recall_curve(y_true, y_probs)
            thresholds_dict['Default (0.5)'] = 0.5
            J = tpr - fpr
            ix_youden = np.argmax(J)
            thresholds_dict['Max Youden'] = roc_thresh[ix_youden]
            f1_scores = 2 * (prec_pr[:-1] * rec_pr[:-1]) / (prec_pr[:-1] + rec_pr[:-1] + 1e-8)
            ix_f1 = np.argmax(f1_scores)
            thresholds_dict['Max F1'] = pr_thresh[ix_f1]
            valid_idx = np.where(tpr >= 0.80)[0]
            if len(valid_idx) > 0:
                ix_clinical = valid_idx[0]
                thresholds_dict['Clinical (Sens>=0.8)'] = roc_thresh[ix_clinical]
            else:
                thresholds_dict['Clinical (Sens>=0.8)'] = roc_thresh[-1]
        else:
            thresholds_dict = {k: 0.5 for k in ['Default (0.5)', 'Max Youden', 'Max F1', 'Clinical (Sens>=0.8)']}

        results_list = []
        for strategy_name, thresh in thresholds_dict.items():
            metrics = self.calculate_metrics(y_true, y_probs, thresh, strategy_name)
            metrics['AUC'] = auc_score
            results_list.append(metrics)

        results_df = pd.DataFrame(results_list)
        results_df.to_csv(os.path.join(self.output_dir, "multistrategy_metrics.csv"), index=False)

        with open(os.path.join(self.output_dir, "metrics_report.txt"), "w", encoding='utf-8') as f:
            f.write("DUAL-STREAM MODEL COMPREHENSIVE REPORT\n")
            f.write("============================================\n")
            f.write(f"Model Structure:      {params_str}\n")
            f.write(f"Inference FLOPs:      {flops_str}\n")
            f.write(f"Overall AUC Score:    {auc_score:.4f}\n\n")
            for index, row in results_df.iterrows():
                f.write("-" * 40 + "\n")
                f.write(f"STRATEGY: {row['Strategy']}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Threshold:            {row['Threshold']:.4f}\n")
                f.write(f"Accuracy:             {row['Accuracy']:.4f}\n")
                f.write(f"F1-Score:             {row['F1']:.4f}\n")
                f.write(f"Sensitivity (Recall): {row['Sensitivity']:.4f}\n")
                f.write(f"Specificity:          {row['Specificity']:.4f}\n")
                f.write(f"Precision (PPV):      {row['Precision']:.4f}\n")
                f.write(f"Confusion Matrix:     TP={row['TP']}, FP={row['FP']}, TN={row['TN']}, FN={row['FN']}\n\n")
            f.write("-" * 40 + "\n")
            f.write("EFFICIENCY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Time:           {total_time:.4f} s\n")
            f.write(f"Avg Time/Sample:      {avg_time:.4f} ms\n")

        if len(np.unique(y_true)) > 1:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='#d62728', lw=2, label=f'AUC = {auc_score:.3f}')
            markers = ['^', 's', 'o']
            colors = ['blue', 'green', 'purple']
            plot_strategies = ['Max Youden', 'Max F1', 'Clinical (Sens>=0.8)']
            for i, strategy in enumerate(plot_strategies):
                thresh_val = thresholds_dict[strategy]
                idx = np.argmin(np.abs(roc_thresh - thresh_val))
                plt.scatter(fpr[idx], tpr[idx], marker=markers[i], s=100, color=colors[i],
                            label=f'{strategy} (Thr={thresh_val:.2f})', zorder=5)
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=14)
            plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=14)
            plt.title('ROC Curve with Multi-Strategies', fontweight='bold', fontname='Times New Roman', fontsize=16)
            plt.legend(prop={'family': 'Times New Roman', 'size': 11})
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.savefig(os.path.join(self.output_dir, "roc_curve_multi.png"), dpi=300, bbox_inches='tight')
            plt.close()

            for index, row in results_df.iterrows():
                strategy_name_safe = row['Strategy'].replace(" ", "_").replace(">=", "ge").replace(".", "_")
                plt.figure(figsize=(5, 4))
                cm = np.array([[row['TN'], row['FP']], [row['FN'], row['TP']]])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"family": "Times New Roman", "size": 14})
                plt.title(f'CM: {row["Strategy"]}\n(Thresh={row["Threshold"]:.2f})', fontweight='bold',
                          fontname='Times New Roman')
                plt.xlabel('Predicted', fontname='Times New Roman')
                plt.ylabel('True', fontname='Times New Roman')
                plt.savefig(os.path.join(self.output_dir, f"cm_{strategy_name_safe}.png"), dpi=300, bbox_inches='tight')
                plt.close()

if __name__ == "__main__":
    if not os.path.exists("config.yaml"):
        exit()
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    Evaluator(config).evaluate()
