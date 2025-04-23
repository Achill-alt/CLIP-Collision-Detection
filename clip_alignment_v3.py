import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import unittest
import time
import math
from typing import List, Dict
from PIL import Image
from tqdm import tqdm
from collections import deque
import seaborn as sns
from transformers import ViTModel, BertModel, BertTokenizer, ViTConfig
warnings.filterwarnings('ignore')

# ========== 1.梯度冲突检测模块 ======================================================
class GradientConflictDetector:
    """梯度冲突检测器，基于参数更新方向一致性"""
    def __init__(self, model_params, decay_init=0.7, decay_final=0.3):
        self.decay_init = decay_init
        self.decay_final = decay_final
        self.model_params = list(model_params)
        self.grad_cos_sim = []

    def _cosine_similarity(self, grad1, grad2):
        return F.cosine_similarity(grad1.flatten(), grad2.flatten(), dim=0)

    def calculate_conflict(self):
        """计算参数间平均梯度余弦相似度"""
        conflict_scores = []
        for param in self.model_params:
            if param.grad is None:
                continue
            # 分层采样关键参数
            if 'projection' in str(param) or 'logit_scale' in str(param):
                for other_param in self.model_params:
                    if other_param is param or other_param.grad is None:
                        continue
                    score = self._cosine_similarity(param.grad, other_param.grad)
                    conflict_scores.append(score.item())
        return torch.tensor(conflict_scores).mean() if conflict_scores else torch.tensor(0.0)

# ========== 2.模型编码器修改 ======================================================
class ImageEncoder(nn.Module):
    """视觉编码器，基于ViT实现，支持分层冻结"""
    def __init__(self, freeze_vit_layers=6):
        super().__init__()
        self.visual_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # 冻结指定层
        for layer in self.visual_backbone.encoder.layer[:freeze_vit_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, image):
        features = self.visual_backbone(pixel_values=image).last_hidden_state[:, 0, :]
        return F.normalize(self.projection(features), dim=-1)

class TextEncoder(nn.Module):
    """文本编码器，基于BERT实现，支持分层冻结"""
    def __init__(self, freeze_bert_layers=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # 冻结指定层
        for layer in self.bert.encoder.layer[:freeze_bert_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)
        
        # 特征投影层
        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        features = outputs.last_hidden_state[:, 0, :]
        return F.normalize(self.projection(features), dim=-1)

# ========== 3.动态权重衰减集成 ====================================================
class MemoryOptimizedCLIP(nn.Module):
    """改进版CLIP模型，集成动态阈值优化"""
    def __init__(self, use_dynamic_threshold=True, freeze_vit=6, freeze_bert=3, device='cpu'):
        super().__init__()
        self.device = device
        self.image_encoder = ImageEncoder(freeze_vit_layers=freeze_vit)
        self.text_encoder = TextEncoder(freeze_bert_layers=freeze_bert)
        self.use_dynamic = use_dynamic_threshold
        self.threshold_module = MemoryEfficientDynamicThreshold(device=device) if use_dynamic_threshold else None
        self.logit_scale = nn.Parameter(torch.ones(1, device=device) * np.log(1 / 0.07))
        self.cm_tracker = OptimizedConfusionMatrixTracker(device=device)
        self.last_metrics = {}
        
        # 初始化梯度冲突检测器
        self.conflict_detector = GradientConflictDetector(
            self.parameters(),
            decay_init=0.7,
            decay_final=0.3
        )
        self.decay_scheduler = None  # 延迟初始化
        self.current_decay = 0.7

    def get_dynamics(self):
        return {
            'scale': self.threshold_module.base_scale.exp().item() if self.use_dynamic else self.logit_scale.exp().item(),
            **{k: v.item() for k, v in self.last_metrics.items()}
        }

    def forward(self, image, text_inputs, labels=None):
        with torch.cuda.amp.autocast():
            image_features = self.image_encoder(image)
            text_features = self.text_encoder(**text_inputs)
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            if self.use_dynamic and self.training:
                current_metrics = self.cm_tracker.get_metrics()
                logit_scale = self.threshold_module(logits_per_image.detach(), cm_metrics=current_metrics)
            else:
                logit_scale = self.logit_scale.exp()

            scaled_logits = logit_scale * logits_per_image

            if self.training:
                batch_size = image_features.size(0)
                target = torch.eye(batch_size, device=self.device)
                self.cm_tracker.update(scaled_logits.detach(), target, threshold=0.5)
                self.last_metrics = self.cm_tracker.get_metrics()

            return scaled_logits

# ========== 4.优化器与训练逻辑 ====================================================
class OptimizedCLIPLoss(nn.Module):
    """改进版CLIP损失函数，含动态惩罚项"""
    def __init__(self, model, alpha=0.5):
        super().__init__()
        self.model = model
        self.alpha = alpha

    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0), device=self.model.device)
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)
        base_loss = (loss_img + loss_txt) / 2

        dynamics = self.model.get_dynamics()
        conflict_level = self.model.conflict_detector.calculate_conflict()

        # 动态衰减逻辑
        current_step = self.model.decay_scheduler.last_epoch if self.model.decay_scheduler else 0
        scheduled_decay = self.model.decay_scheduler.get_last_lr()[0] if self.model.decay_scheduler else 0.7
        
        # 基于指标的自适应衰减
        f1_factor = (1.0 - dynamics['f1']) * 0.5
        adaptive_decay = scheduled_decay * (1 - f1_factor)

        # 最终衰减系数
        final_decay = torch.clamp(
            torch.tensor(adaptive_decay),
            min=0.3,
            max=0.7
        ).to(self.model.device)

        scale_penalty = F.mse_loss(
            torch.tensor(dynamics['scale'], device=self.model.device),
            torch.tensor(1.0, device=self.model.device)
        )
        metric_balance = (1.0 - dynamics['f1']) * self.alpha

        return base_loss + final_decay * scale_penalty + 0.3 * metric_balance

def optimized_train_example():
    """训练示例"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MemoryOptimizedCLIP(device=device, use_dynamic_threshold=True)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    analyzer = CLIPAlignmentAnalyzer(model, tokenizer, device)
    loss_fn = OptimizedCLIPLoss(model, alpha=0.5)

    # 参数分组优化器
    projection_params = [
        {"params": model.image_encoder.projection.parameters(), "weight_decay": 0.7},
        {"params": model.text_encoder.projection.parameters(), "weight_decay": 0.7}
    ]
    other_params = [
        {"params": [p for n, p in model.named_parameters() if 'projection' not in n], "weight_decay": 0.0}
    ]
    optimizer = torch.optim.AdamW(projection_params + other_params, lr=1e-4)
    
    # 初始化学习率调度器
    model.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=100,
        eta_min=0.3
    )

    # 测试数据生成
    images = torch.randn(16, 3, 224, 224).to(device)
    texts = ["a photo of cat"] * 8 + ["a picture of dog"] * 8
    text_inputs = analyzer._prepare_text(texts)

    model.train()
    decay_values = []
    conflict_levels = []

    for epoch in range(10):
        optimizer.zero_grad()
        logits = model(images, text_inputs)
        loss = loss_fn(logits, logits.t())
        loss.backward()
        optimizer.step()
        model.decay_scheduler.step()  # 正确的位置更新学习率

        # 动态调整权重衰减
        conflict_level = model.conflict_detector.calculate_conflict()
        current_decay = 0.7 - (0.7 - 0.3) * (epoch / 10)  # 线性衰减
        
        # 更新优化器参数
        for param_group in optimizer.param_groups:
            if 'projection' in str(param_group['params'][0]):
                param_group['weight_decay'] = current_decay * conflict_level.item()

        # 记录衰减值
        decay_values.append(current_decay)
        conflict_levels.append(conflict_level.item())
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Decay: {current_decay:.3f}')

    # 可视化监控
    plt.figure(figsize=(10, 6))
    plt.plot(decay_values, label='Weight Decay')
    plt.plot(conflict_levels, label='Conflict Level', linestyle='--')
    plt.title("Dynamic Weight Decay Monitoring")
    plt.xlabel("Training Steps")
    plt.ylabel("Value")
    plt.legend()
    plt.savefig('decay_dynamics.png')

# ========== 5.监控测试类 ========================================================
class EnhancedCLIPTest(unittest.TestCase):
    """增强型测试模块，集成双可视化功能"""
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = MemoryOptimizedCLIP(device=self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.analyzer = CLIPAlignmentAnalyzer(self.model, self.tokenizer, self.device)
        self.test_images = torch.randn(4, 3, 224, 224).to(self.device)
        self.text_descriptions = [
            "a photo of cat", "a picture of dog",
            "an animal in forest", "a furry mammal"
        ]

    def test_decay_monitoring(self):
        """权重衰减动态监控测试"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        loss_fn = OptimizedCLIPLoss(self.model, alpha=0.5)
        
        # 初始化调度器
        self.model.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer,
            T_max=100,
            eta_min=0.3
        )
        
        self.model.train()
        decay_values = []
        conflict_levels = []

        for epoch in range(5):
            optimizer.zero_grad()
            text_inputs = self.analyzer._prepare_text(self.text_descriptions)
            logits = self.model(self.test_images, text_inputs)
            loss = loss_fn(logits, logits.t())
            loss.backward()
            optimizer.step()
            self.model.decay_scheduler.step()

            # 记录衰减值
            for param_group in optimizer.param_groups:
                if 'projection' in str(param_group['params'][0]):
                    decay_values.append(param_group['weight_decay'])
            conflict_levels.append(self.model.conflict_detector.calculate_conflict().item())

        # 可视化验证
        plt.figure(figsize=(10, 6))
        plt.plot(decay_values, label='Weight Decay')
        plt.plot(conflict_levels, label='Conflict Level', linestyle='--')
        plt.title("Dynamic Weight Decay Test")
        plt.xlabel("Training Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.savefig('test_decay_dynamics.png')
        plt.close()

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(EnhancedCLIPTest)
    unittest.TextTestRunner(verbosity=2).run(suite)
    optimized_train_example()
