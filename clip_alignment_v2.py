import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTConfig, BertModel, BertTokenizer
from collections import deque
import numpy as np
import os
import warnings
import unittest
from tqdm import tqdm  # 新增进度条库
warnings.filterwarnings('ignore')

# ------------------ 内存优化增强的混淆矩阵跟踪器 ------------------
class OptimizedConfusionMatrixTracker:
    def __init__(self, num_classes=2, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.reset()
        
        self.history_size = 1000
        self.precision_history = deque(maxlen=self.history_size)
        self.recall_history = deque(maxlen=self.history_size)
        self.f1_history = deque(maxlen=self.history_size)
        self.threshold_history = deque(maxlen=self.history_size)

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, preds, labels, threshold=0.5):
        with torch.no_grad():
            preds = (preds >= threshold).float()
            labels = labels.float()
            
            tp = torch.sum((preds == 1) & (labels == 1)).item()
            fp = torch.sum((preds == 1) & (labels == 0)).item()
            fn = torch.sum((preds == 0) & (labels == 1)).item()
            tn = torch.sum((preds == 0) & (labels == 0)).item()
            
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.tn += tn

            self.threshold_history.append(threshold)
            metrics = self.get_metrics()
            self.precision_history.append(metrics['precision'].item())
            self.recall_history.append(metrics['recall'].item())
            self.f1_history.append(metrics['f1'].item())

    def get_metrics(self):
        eps = 1e-8
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        return {
            'precision': torch.tensor(precision, device=self.device),
            'recall': torch.tensor(recall, device=self.device),
            'f1': torch.tensor(f1, device=self.device)
        }

    def visualize(self, window_size=100):
        plt.figure(figsize=(12,6))
        plt.plot(list(self.precision_history)[-window_size:], label='Precision')
        plt.plot(list(self.recall_history)[-window_size:], label='Recall')
        plt.plot(list(self.f1_history)[-window_size:], label='F1')
        plt.plot(list(self.threshold_history)[-window_size:], label='Threshold', linestyle='--')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title('Training Dynamics Monitoring')
        plt.legend()
        plt.show()

# ------------------ 显存优化的动态阈值模块 ------------------
class MemoryEfficientDynamicThreshold(nn.Module):
    def __init__(self, window_size=1000, init_temp=0.07, momentum=0.9, 
                 update_freq=50, max_scale=5.0, device='cpu'):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.momentum = momentum
        self.update_freq = update_freq
        self.max_scale = max_scale
        
        self.base_scale = nn.Parameter(torch.tensor(np.log(init_temp), dtype=torch.float32, device=device))
        self.register_buffer('running_mean', torch.zeros(1, dtype=torch.float32, device=device))
        self.register_buffer('running_std', torch.ones(1, dtype=torch.float32, device=device))
        
        self.scale_history = deque(maxlen=500)
        self.stability_warning_count = 0
        self.cm_alpha = 0.3
        self.update_counter = 0

    def _calculate_adaptive_scale(self, logits):
        with torch.cuda.amp.autocast():
            z_scores = (logits - self.running_mean) / (self.running_std + 1e-7)
            return torch.sigmoid(z_scores) * 2.0 + 0.5

    def update_statistics(self, similarities):
        if len(self.scale_history) >= 500:
            self.scale_history.popleft()
        self.scale_history.append(self.base_scale.exp().item())
        
        if self.update_counter % self.update_freq == 0:
            recent_scales = np.array(self.scale_history)
            if len(recent_scales) >= 100:
                current_std = np.std(recent_scales[-100:])
                current_mean = np.mean(recent_scales[-100:])
                if current_std > 0.15 * current_mean:
                    self.stability_warning_count += 1
                    print(f"[Warning] Threshold instability! STD={current_std:.4f}, Mean={current_mean:.4f}")
        self.update_counter += 1

    def forward(self, logits, cm_metrics=None):
        with torch.no_grad():
            base_scale = torch.clamp(self.base_scale.exp(), min=1e-4, max=self.max_scale)
            
            if self.training and cm_metrics is not None:
                f1_factor = 1.0 + (cm_metrics['f1'] - 0.5) * self.cm_alpha
                f1_factor = torch.clamp(f1_factor, 0.5, 2.0).to(self.device)
                adaptive_scale = self._calculate_adaptive_scale(logits)
                final_scale = base_scale * f1_factor * adaptive_scale
                self.update_statistics(final_scale)
                return final_scale
            
            return base_scale

# ------------------ 显存优化的模型架构 ------------------
class MemoryOptimizedCLIP(nn.Module):
    def __init__(self, use_dynamic_threshold=True, device='cpu'):
        super().__init__()
        self.device = device
        
        config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
        config.update({'use_pooler': False})
        self.image_encoder = ViTModel.from_pretrained(
            'google/vit-base-patch16-224',
            config=config,
            add_pooling_layer=False
        ).to(device)
        
        self.image_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        ).to(device)
        
        self.text_encoder = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.text_proj = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1)
        ).to(device)
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.use_dynamic = use_dynamic_threshold
        self.threshold_module = MemoryEfficientDynamicThreshold(device=device) if use_dynamic_threshold else None
        self.logit_scale = nn.Parameter(torch.ones(1, device=device) * np.log(1/0.07))
        self.cm_tracker = OptimizedConfusionMatrixTracker(device=device)
        self.last_metrics = {}

    def forward(self, image, text_inputs, labels=None):
        with torch.cuda.amp.autocast():
            image_features = self.image_encoder(pixel_values=image).last_hidden_state[:, 0, :]
            image_features = F.normalize(self.image_proj(image_features), dim=-1)
            
            text_features = self.text_encoder(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            ).last_hidden_state[:, 0, :]
            text_features = F.normalize(self.text_proj(text_features), dim=-1)
            
            del image, text_inputs
            torch.cuda.empty_cache()
            
            logits_per_image = image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            if self.use_dynamic and self.training:
                current_metrics = self.cm_tracker.get_metrics()
                logit_scale = self.threshold_module(
                    logits_per_image.detach(),
                    cm_metrics=current_metrics
                )
            else:
                logit_scale = self.logit_scale.exp()

            scaled_logits = logit_scale * logits_per_image
            
            if labels is not None and self.training:
                batch_size = image_features.size(0)
                target = torch.eye(batch_size, device=self.device)
                self.cm_tracker.update(
                    scaled_logits.detach(),
                    target,
                    threshold=0.5
                )
                self.last_metrics = self.cm_tracker.get_metrics()

            return scaled_logits

    def get_dynamics(self):
        return {
            'scale': self.threshold_module.base_scale.exp().item() if self.use_dynamic else self.logit_scale.exp().item(),
            **{k: v.item() for k, v in self.last_metrics.items()}
        }

# ------------------ 内存优化的损失函数 ------------------
class OptimizedCLIPLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.alpha = 0.5
        self.grad_accum_steps = 4

    def forward(self, logits_per_image, logits_per_text):
        labels = torch.arange(logits_per_image.size(0), device=self.model.device)
        
        chunk_size = max(1, logits_per_image.size(0) // 2)
        loss_img = 0
        loss_txt = 0
        
        for i in range(0, logits_per_image.size(0), chunk_size):
            chunk_img = logits_per_image[i:i+chunk_size]
            chunk_txt = logits_per_text[i:i+chunk_size]
            loss_img += F.cross_entropy(chunk_img, labels[i:i+chunk_size])
            loss_txt += F.cross_entropy(chunk_txt, labels[i:i+chunk_size])
        
        base_loss = (loss_img + loss_txt) / 2

        dynamics = self.model.get_dynamics()
        scale = dynamics['scale']
        
        scale_penalty = F.mse_loss(torch.tensor(scale, device=self.model.device), 
                                 torch.tensor(1.0, device=self.model.device))
        metric_balance = (1.0 - dynamics['f1']) * self.alpha
        
        return (base_loss + 0.1*scale_penalty + 0.3*metric_balance) / self.grad_accum_steps

# ------------------ 单元测试模块 ------------------
class TestCLIPComponents(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_confusion_matrix(self):
        """测试混淆矩阵跟踪器"""
        tracker = OptimizedConfusionMatrixTracker(device=self.device)
        
        # 测试完美预测
        tracker.reset()
        preds = torch.tensor([1, 0, 1, 0], device=self.device)
        labels = torch.tensor([1, 0, 1, 0], device=self.device)
        tracker.update(preds, labels)
        metrics = tracker.get_metrics()
        self.assertAlmostEqual(metrics['precision'].item(), 1.0)
        self.assertAlmostEqual(metrics['recall'].item(), 1.0)
        
        # 测试部分错误
        tracker.reset()
        preds = torch.tensor([1, 0, 1, 0], device=self.device)
        labels = torch.tensor([1, 0, 0, 1], device=self.device)
        tracker.update(preds, labels)
        metrics = tracker.get_metrics()
        self.assertAlmostEqual(metrics['precision'], 0.5, delta=0.01)
        self.assertAlmostEqual(metrics['recall'], 0.5, delta=0.01)

# ------------------ 训练示例（带进度条）------------------
def optimized_train_example():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache()
    model = MemoryOptimizedCLIP(device=device).to(device)
    loss_fn = OptimizedCLIPLoss(model)
    
    # 生成增强测试数据
    images = torch.randn(16, 3, 224, 224).to(device)
    texts = ["a photo of cat"]*8 + ["a picture of dog"]*8  # 平衡样本
    text_inputs = model.tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    
    # 训练参数设置
    total_steps = 200
    grad_accum_steps = loss_fn.grad_accum_steps
    
    # 初始化进度条
    pbar = tqdm(range(total_steps), desc="Training Progress", unit="step")
    
    for step in pbar:
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast():
            logits = model(images, text_inputs, labels=torch.eye(16, device=device))
            loss = loss_fn(logits, logits.t())
        
        scaler.scale(loss).backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
        
        # 实时更新进度条信息
        if step % 2 == 0:
            dynamics = model.get_dynamics()
            pbar.set_postfix({
                'Loss': f"{loss.item() * grad_accum_steps:.3f}",
                'Scale': f"{dynamics['scale']:.3f}",
                'F1': f"{dynamics['f1']:.3f}"
            })
        
        # 定期清理显存
        if step % 50 == 0:
            torch.cuda.empty_cache()
    
    # 可视化训练动态
    model.cm_tracker.visualize(window_size=50)
    print("Training completed!")

if __name__ == "__main__":
    # 运行单元测试
    unittest.main(argv=[''], exit=False)
    
    # 执行训练示例
    optimized_train_example()