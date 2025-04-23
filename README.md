# CLIP冲突检测算法
CLIP冲突检测算法是基于CLIP（Contrastive Language-Image Pre-training）模型的多模态语义对齐分析技术，旨在识别图像与文本之间的语义不一致性（冲突）。

clip_alignment_v1.py是CLIP语义对齐度计算模块基础模块

clip_alignment_v2.py是优化动态阈值算法（基于混淆矩阵反馈）后的版本

clip_alignment_v3.py是集成冲突权重衰减逻辑（权重0.7→0.3）后的版本
