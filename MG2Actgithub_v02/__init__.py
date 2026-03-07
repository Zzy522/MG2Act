"""
MG2Act: 分子胶降解活性预测模型

这是一个用于预测分子胶（molecular glue）降解活性的深度学习模型。
模型使用多模态融合架构，结合了：
- E3连接酶序列编码（ESMC预训练模型）
- 靶点蛋白序列编码（ESMC预训练模型）  
- 分子结构编码（GNN + 官能团检测）
- 交叉注意力机制进行特征融合

主要模块：
- model: 主要的MG2ActModel类
- encoder: 分子和蛋白质编码器
- attention_fusion: 注意力融合机制
- dataset: 数据集加载和处理
- train: 训练脚本
- evaluate: 评估工具
"""

from .model import MG2ActModel
from .encoder import EnhancedGNNMolecularEncoder, SequenceEncoder, FunctionalGroupDetector
from .attention_fusion import MultiScaleAttentionFusion, ConcatenationFusion
from .dataset import MG2ActDataset, collate_samples
from .evaluate import evaluate
from .utils import LayerNorm, ConvPooler

__version__ = "1.0.0"
__author__ = "MG2Act Team"

__all__ = [
    "MG2ActModel",
    "EnhancedGNNMolecularEncoder", 
    "SequenceEncoder",
    "FunctionalGroupDetector",
    "MultiScaleAttentionFusion",
    "ConcatenationFusion", 
    "MG2ActDataset",
    "collate_samples",
    "evaluate",
    "LayerNorm",
    "ConvPooler"
]

