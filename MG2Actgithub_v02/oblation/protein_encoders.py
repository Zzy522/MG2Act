"""
蛋白质序列编码器消融实验模块
支持不同的蛋白质编码方法：
1. ACC (自协互协) - Auto Cross-Covariance
2. OneHot - One-hot编码
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Dict

# 氨基酸列表和索引映射
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: idx for idx, aa in enumerate(AA_LIST)}


class ProteinEncoder(nn.Module):
    """
    蛋白质序列编码器基类
    """

    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.output_dim = output_dim

    def forward(self, seq_list: List[str]) -> torch.Tensor:
        """
        编码蛋白质序列列表

        参数:
            seq_list: 蛋白质序列列表

        返回:
            torch.Tensor: [batch_size, output_dim] 的编码向量
        """
        raise NotImplementedError


class OneHotProteinEncoder(ProteinEncoder):
    """
    One-hot编码蛋白质序列编码器

    将每个氨基酸编码为20维one-hot向量，然后通过CNN或MLP降维
    """

    def __init__(self, output_dim: int = 64, method: str = "cnn"):
        super().__init__(output_dim)
        self.method = method.lower()
        self.aa_dim = len(AA_LIST)  # 20

        if self.method == "cnn":
            # 使用CNN编码器
            self.encoder = nn.Sequential(
                nn.Conv1d(self.aa_dim, 128, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),  # 全局平均池化
                nn.Flatten(),
                nn.Linear(64, output_dim),
                nn.LayerNorm(output_dim)
            )
        elif self.method == "mlp":
            # 使用MLP编码器（需要固定序列长度）
            self.max_length = 1024  # 最大序列长度
            self.encoder = nn.Sequential(
                nn.Flatten(),  # [batch, seq_len * aa_dim]
                nn.Linear(self.max_length * self.aa_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            raise ValueError(f"不支持的编码方法: {method}")

    def sequence_to_onehot(self, seq: str) -> torch.Tensor:
        """
        将蛋白质序列转换为one-hot编码

        参数:
            seq: 蛋白质序列字符串

        返回:
            torch.Tensor: [seq_len, aa_dim] 的one-hot向量
        """
        onehot = []
        for aa in seq:
            vec = torch.zeros(self.aa_dim)
            idx = AA_INDEX.get(aa)
            if idx is not None:
                vec[idx] = 1.0
            onehot.append(vec)
        return torch.stack(onehot)  # [seq_len, aa_dim]

    def forward(self, seq_list: List[str]) -> torch.Tensor:
        """
        编码蛋白质序列列表

        参数:
            seq_list: 蛋白质序列列表

        返回:
            torch.Tensor: [batch_size, output_dim]
        """
        device = next(self.parameters()).device

        if self.method == "cnn":
            # CNN方法：处理变长序列
            encoded_seqs = []
            for seq in seq_list:
                onehot = self.sequence_to_onehot(seq).to(device)  # [seq_len, aa_dim]
                if onehot.size(0) == 0:
                    # 空序列使用零向量
                    encoded = torch.zeros(self.output_dim, device=device)
                else:
                    # 转置为 [aa_dim, seq_len] 以适应Conv1d
                    onehot_t = onehot.transpose(0, 1).unsqueeze(0)  # [1, aa_dim, seq_len]
                    encoded = self.encoder(onehot_t).squeeze(0)  # [output_dim]
                encoded_seqs.append(encoded)
            return torch.stack(encoded_seqs, dim=0)  # [batch_size, output_dim]

        elif self.method == "mlp":
            # MLP方法：固定长度
            batch_onehot = []
            for seq in seq_list:
                onehot = self.sequence_to_onehot(seq)  # [seq_len, aa_dim]
                seq_len = onehot.size(0)

                if seq_len > self.max_length:
                    # 截断
                    onehot = onehot[:self.max_length]
                    seq_len = self.max_length
                elif seq_len < self.max_length:
                    # 填充
                    padding = torch.zeros(self.max_length - seq_len, self.aa_dim)
                    onehot = torch.cat([onehot, padding], dim=0)

                batch_onehot.append(onehot.flatten())  # [max_length * aa_dim]

            batch_tensor = torch.stack(batch_onehot, dim=0).to(device)  # [batch_size, max_length * aa_dim]
            return self.encoder(batch_tensor)  # [batch_size, output_dim]


class ACCProteinEncoder(ProteinEncoder):
    """
    ACC (Auto Cross-Covariance) 蛋白质序列编码器

    基于序列的自协方差和互协方差特征
    """

    def __init__(self, output_dim: int = 64, max_lag: int = 3):
        super().__init__(output_dim)
        self.max_lag = max_lag
        self.aa_dim = len(AA_LIST)

        # ACC特征维度：均值(20) + 每个lag的协方差矩阵(20x20)
        self.acc_dim = self.aa_dim + self.max_lag * self.aa_dim * self.aa_dim

        # 投影层：将ACC特征降维到output_dim
        self.projection = nn.Sequential(
            nn.Linear(self.acc_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def sequence_to_acc(self, seq: str) -> torch.Tensor:
        """
        计算序列的ACC (Auto Cross-Covariance) 特征

        参数:
            seq: 蛋白质序列字符串

        返回:
            torch.Tensor: ACC特征向量
        """
        if not seq:
            return torch.zeros(self.acc_dim)

        # 转换为one-hot
        onehots = []
        for ch in seq:
            idx = AA_INDEX.get(ch)
            vec = np.zeros(self.aa_dim, dtype=np.float32)
            if idx is not None:
                vec[idx] = 1.0
            onehots.append(vec)

        if not onehots:
            return torch.zeros(self.acc_dim)

        X = np.stack(onehots)  # [seq_len, aa_dim]
        mean = X.mean(axis=0, keepdims=True)  # [1, aa_dim]

        # ACC特征：均值 + 协方差矩阵展开
        feats = [mean.ravel()]

        for lag in range(1, self.max_lag + 1):
            if X.shape[0] <= lag:
                feats.append(np.zeros(self.aa_dim * self.aa_dim, dtype=np.float32))
                continue

            x1 = X[:-lag] - mean  # [seq_len-lag, aa_dim]
            x2 = X[lag:] - mean   # [seq_len-lag, aa_dim]

            # 计算协方差矩阵
            cov = np.einsum("ni,nj->ij", x1, x2) / x1.shape[0]  # [aa_dim, aa_dim]
            feats.append(cov.astype(np.float32).ravel())

        return torch.tensor(np.concatenate(feats, axis=0), dtype=torch.float32)

    def forward(self, seq_list: List[str]) -> torch.Tensor:
        """
        编码蛋白质序列列表

        参数:
            seq_list: 蛋白质序列列表

        返回:
            torch.Tensor: [batch_size, output_dim]
        """
        device = next(self.parameters()).device

        acc_features = []
        for seq in seq_list:
            acc_feat = self.sequence_to_acc(seq).to(device)
            acc_features.append(acc_feat)

        batch_tensor = torch.stack(acc_features, dim=0)  # [batch_size, acc_dim]
        return self.projection(batch_tensor)  # [batch_size, output_dim]


class AblationProteinEncoder(nn.Module):
    """
    消融实验蛋白质编码器
    支持选择不同的编码方法：ESMC, ACC, OneHot
    """

    def __init__(self,
                 method: str = "esmc",
                 output_dim: int = 64,
                 device: torch.device = torch.device("cpu"),
                 **kwargs):
        """
        参数:
            method: 编码方法 ("esmc", "acc", "onehot")
            output_dim: 输出维度
            device: 设备
            **kwargs: 传递给具体编码器的参数
        """
        super().__init__()
        self.method = method.lower()
        self.output_dim = output_dim
        self.device = device

        if self.method == "esmc":
            # 使用ESMC编码器
            from ..encoder import SequenceEncoder
            self.encoder = SequenceEncoder(
                device=device,
                proj_method=kwargs.get("proj_method", "conv"),
                embed_dim=output_dim,
                dropout=kwargs.get("dropout", 0.1)
            )
        elif self.method == "acc":
            self.encoder = ACCProteinEncoder(
                output_dim=output_dim,
                max_lag=kwargs.get("max_lag", 3)
            )
        elif self.method == "onehot":
            self.encoder = OneHotProteinEncoder(
                output_dim=output_dim,
                method=kwargs.get("onehot_method", "cnn")
            )
        else:
            raise ValueError(f"不支持的蛋白质编码方法: {method}")

        self.to(device)

    def forward(self, seq_list: List[str], return_token_level: bool = False):
        """
        编码蛋白质序列

        参数:
            seq_list: 蛋白质序列列表
            return_token_level: 是否返回token级别特征（仅ESMC支持）

        返回:
            如果return_token_level=True且method="esmc": token级别特征列表
            否则: [batch_size, output_dim] 的编码向量
        """
        if self.method == "esmc" and return_token_level:
            return self.encoder(seq_list, return_token_level=True)
        else:
            return self.encoder(seq_list)

    def preload_cache(self, seq_list: List[str], device=None):
        """预加载缓存（仅ESMC支持）"""
        if self.method == "esmc":
            if device is None:
                device = self.device
            self.encoder.preload_cache(seq_list, device)

    def clear_cache(self):
        """清除缓存（仅ESMC支持）"""
        if self.method == "esmc":
            self.encoder.clear_cache()

    def get_cache_info(self):
        """获取缓存信息（仅ESMC支持）"""
        if self.method == "esmc":
            return self.encoder.get_cache_info()
        return {"method": self.method, "cache": "not supported"}
