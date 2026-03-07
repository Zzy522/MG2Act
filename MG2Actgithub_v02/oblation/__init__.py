"""
MG2Act 消融实验模块
包含蛋白质编码消融实验的代码
"""

from .protein_encoders import AblationProteinEncoder, OneHotProteinEncoder, ACCProteinEncoder
from .model_ablation import MG2ActAblationModel
from .pooled_attention_fusion import PooledAttentionFusion

__all__ = [
    'AblationProteinEncoder',
    'OneHotProteinEncoder',
    'ACCProteinEncoder',
    'MG2ActAblationModel',
    'PooledAttentionFusion'
]