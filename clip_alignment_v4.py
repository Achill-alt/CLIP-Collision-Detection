
# -*- coding: utf-8 -*-
"""
��ģ̬CLIPģ�͸Ľ���ʵ�֣��������º��Ĺ��ܣ�
1. ����ViT��BERT��˫ģ̬������
2. �ֲ�����������
3. ��̬���������������ֵ�Ż�
4. ��Ͼ���ѵ��֧��
5. �ݶȳ�ͻ����붯̬Ȩ��˥��
6. ���������ӻ�ģ��
7. ������֧����ģ�ͳ־û�
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
import math
import time
from typing import List, Dict
from collections import deque
from PIL import Image
from tqdm import tqdm
from transformers import ViTModel, BertModel, BertTokenizer, ViTConfig

warnings.filterwarnings('ignore')

#%% �ݶȳ�ͻ���ģ��
class GradientConflictDetector:
    """�ݶȳ�ͻ����������ڲ������·���һ����"""
    def __init__(self, model_params, decay_init=0.7, decay_final=0.3):
        self.decay_init = decay_init
        self.decay_final = decay_final
        self.model_params = list(model_params)
        self.grad_cos_sim = []
    
    def _cosine_similarity(self, grad1, grad2):
        return F.cosine_similarity(grad1.flatten(), grad2.flatten(), dim=0)
    
    def calculate_conflict(self):
        """���������ƽ���ݶ��������ƶ�"""
        conflict_scores = []
        for i, param in enumerate(self.model_params):
            if param.grad is None:
                continue
            for j in range(i+1, len(self.model_params)):
                other_param = self.model_params[j]
                if other_param.grad is None:
                    continue
                score = self._cosine_similarity(param.grad, other_param.grad)
                conflict_scores.append(score.item())
        return torch.tensor(conflict_scores).mean() if conflict_scores else torch.tensor(0.0)

#%% ģ�ͱ�����
class ImageEncoder(nn.Module):
    """�Ӿ�������������ViTʵ�֣�֧�ֲַ㶳��"""
    def __init__(self, freeze_vit_layers=6):
        super().__init__()
        self.visual_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        
        # �ֲ㶳�����
        for layer in self.visual_backbone.encoder.layer[:freeze_vit_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)
        
        # ����ͶӰ��
        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, image):
        # ��ȡȫ������(CLS token)
        features = self.visual_backbone(pixel_values=image).last_hidden_state[:, 0, :]
        return F.normalize(self.projection(features), dim=-1)

class TextEncoder(nn.Module):
    """�ı�������������BERTʵ�֣�֧�ֲַ㶳��"""
    def __init__(self, freeze_bert_layers=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        
        # ����ָ����
        for layer in self.bert.encoder.layer[:freeze_bert_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)
        
        # ����ͶӰ��
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

#%% ��̬��ֵ�Ż�ģ��
class OptimizedConfusionMatrixTracker:
    """��̬���������������֧����ֵ����"""
    def __init__(self, num_classes=2, device='cpu'):
        self.num_classes = num_classes
        self.device = device
        self.history_size = 1000
        self.reset()
        self.precision_history = deque(maxlen=self.history_size)
        self.recall_history = deque(maxlen=self.history_size)
        self.f1_history = deque(maxlen=self.history_size)
        self.threshold_history = deque(maxlen=self.history_size)
    
    def reset(self):
        self.tp = self.fp = self.fn = self.tn = 0
    
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
        plt.plot(list(self.threshold_history)[-window_size:], 
                label='Threshold', linestyle='--')
        plt.xlabel('Steps')
        plt.ylabel('Value')
        plt.title('Training Dynamics Monitoring')
        plt.legend()
        plt.show()

class MemoryEfficientDynamicThreshold(nn.Module):
    """��̬��ֵ�㷨�����ڻ�������������"""
    def __init__(self, window_size=1000, init_temp=0.07, momentum=0.9, 
                update_freq=50, max_scale=5.0, device='cpu'):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.momentum = momentum
        self.update_freq = update_freq
        self.max_scale = max_scale
        self.base_scale = nn.Parameter(torch.tensor(np.log(init_temp), 
                                  dtype=torch.float32, device=device))
        self.register_buffer('running_mean', torch.zeros(1, dtype=torch.float32, device=device))
        self.register_buffer('running_std', torch.ones(1, dtype=torch.float32, device=device))
        self.scale_history = deque(maxlen=500)
        self.stability_warning_count = 0
        self.cm_alpha = 0.3
        self.update_counter = 0
    
    def _calculate_adaptive_scale(self, logits):
        logits_flat = logits.mean()
        z_score = (logits_flat - self.running_mean) / (self.running_std + 1e-7)
        return torch.sigmoid(z_score) * 2.0 + 0.5
    
    def update_statistics(self, final_scale):
        if len(self.scale_history) >= 500:
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
                self.scale_history.append(final_scale.item())
                self.update_statistics(final_scale)
                return final_scale
            return base_scale

#%% ����CLIPģ��
class CLIPModel(nn.Module):
    """�Ľ���CLIPģ�ͣ����ɶ�̬��ֵ�Ż����ݶȳ�ͻ���"""
    def __init__(self, use_dynamic_threshold=True, freeze_vit=6, freeze_bert=3, device='cpu'):
        super().__init__()
        self.device = device
        self.image_encoder = ImageEncoder(freeze_vit_layers=freeze_vit)
        self.text_encoder = TextEncoder(freeze_bert_layers=freeze_bert)
        self.use_dynamic = use_dynamic_threshold
        self.threshold_module = MemoryEfficientDynamicThreshold(device=device) if use_dynamic_threshold else None
        self.logit_scale = nn.Parameter(torch.ones(1, device=device) * np.log(1 / 0.07))
        self.cm_tracker = OptimizedConfusionMatrixTracker(device=device)
        self.conflict_detector = GradientConflictDetector(self.parameters())
        self.decay_scheduler = None
        self.last_metrics = {}
    
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
    
    def get_dynamics(self):
        return {
            'scale': self.threshold_module.base_scale.exp().item() if self.use_dynamic 
                     else self.logit_scale.exp().item(),
            **{k: v.item() for k, v in self.last_metrics.items()}
        }

#%% ���������
class CLIPAlignmentAnalyzer:
    """�������ȼ�������ӻ�ģ��"""
    def __init__(self, clip_model, tokenizer, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = clip_model.to(self.device).eval()
        self.tokenizer = tokenizer
    
    def _prepare_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            texts, padding=True, return_tensors="pt",
            max_length=77, truncation=True, return_token_type_ids=True
        ).to(self.device)
    
    def calculate_similarity(self, images: torch.Tensor, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            text_inputs = self._prepare_text(texts)
            image_features = self.clip_model.image_encoder(images.to(self.device))
            text_features = self.clip_model.text_encoder(**text_inputs)
            logit_scale = self.clip_model.logit_scale.exp()
            similarity = logit_scale * (image_features @ text_features.t())
        return similarity.cpu()
    
    def visualize_alignment(self, similarity_matrix: torch.Tensor, texts: List[str], figsize=(12,10)):
        plt.figure(figsize=figsize, dpi=120)
        matrix = similarity_matrix.numpy() if torch.is_tensor(similarity_matrix) else similarity_matrix
        heatmap = plt.imshow(matrix, cmap='plasma', aspect='auto')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(texts)), texts, rotation=55, ha='right')
        plt.yticks(np.arange(len(texts)), [f'Image{i+1}' for i in range(len(texts))])
        plt.title("Cross-Modal Semantic Alignment")
        plt.xlabel("Text Descriptions")
        plt.ylabel("Visual Content")
        plt.tight_layout()
        plt.show()
    
    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tokenizer_dir = os.path.join(os.path.dirname(path), "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_dir)
        torch.save({
            'clip_model': self.clip_model.state_dict(),
            'tokenizer_dir': tokenizer_dir
        }, path)
    
    @classmethod
    def load_model(cls, path: str, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        tokenizer = BertTokenizer.from_pretrained(checkpoint['tokenizer_dir'])
        model = CLIPModel(device=device)
        model.load_state_dict(checkpoint['clip_model'])
        return cls(model, tokenizer, device)

#%% ѵ���Ż�ģ��
class OptimizedCLIPLoss(nn.Module):
    """�Ľ���CLIP��ʧ����������̬�ͷ���"""
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
        
        # ��̬˥���߼�
        current_step = self.model.decay_scheduler.last_epoch if self.model.decay_scheduler else 0
        scheduled_decay = self.model.decay_scheduler.get_last_lr()[0] if self.model.decay_scheduler else 0.7
        
        # ����ָ�������Ӧ˥��
        f1_factor = (1.0 - dynamics['f1']) * 0.5
        adaptive_decay = scheduled_decay * (1 - f1_factor)
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

#%% �ۺϲ���ģ��
class CLIPExtendedTests:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clip_model = CLIPModel(device=device).to(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.analyzer = CLIPAlignmentAnalyzer(self.clip_model, self.tokenizer, device)
    
    def test_basic_alignment(self):
        """��֤������������"""
        print("\n=== ����������� ===")
        real_images = [
            torch.rand(3, 224, 224).to(self.device) * 0.3 + 0.4,
            torch.rand(3, 224, 224).to(self.device) * 0.2 + 0.6,
            torch.rand(3, 224, 224).to(self.device) * 0.5 + 0.3
        ]
        batch_images = torch.stack(real_images)
        texts = [
            "A gray tabby cat on carpet",
            "Golden retriever playing with ball",
            "Sports car on racing track"
        ]
        similarity = self.analyzer.calculate_similarity(batch_images, texts)
        print("�Խ���ƥ�����:", similarity.diag().tolist())
        self.analyzer.visualize_alignment(similarity, texts)
    
    def test_edge_cases(self):
        """��֤�쳣���봦������"""
        print("\n=== �߽�������� ===")
        # ���ı�����
        empty_text_similarity = self.analyzer.calculate_similarity(
            torch.rand(1, 3, 224, 224).to(self.device), [""])
        print("���ı����ƶ�:", empty_text_similarity.item())
        
        # ��Чͼ�����
        try:
            invalid_image = torch.rand(2, 3, 512, 512).to(self.device)
            self.analyzer.calculate_similarity(invalid_image, ["Test"])
        except Exception as e:
            print(f"�����쳣: {str(e)}")
    
    def test_multilingual_support(self):
        """��֤������������������"""
        print("\n=== ������֧�ֲ��� ===")
        chinese_texts = [
            "һֻ��ɳ����˯������è",
            "�ݵ��ϱ��ܵĽ�ëȮ",
            "���ٹ�·�ϵĺ�ɫ�ܳ�"
        ]
        similarity = self.analyzer.calculate_similarity(
            torch.rand(3, 3, 224, 224).to(self.device), chinese_texts)
        print("�����Զ������:\n", similarity.numpy().round(2))
    
    def test_model_persistence(self):
        """��֤ģ�ͱ���/����������"""
        print("\n=== ģ�ͳ־û����� ===")
        temp_path = "temp_model.pth"
        self.analyzer.save_model(temp_path)
        loaded_analyzer = CLIPAlignmentAnalyzer.load_model(temp_path)
        test_image = torch.rand(1, 3, 224, 224).to(self.device)
        orig_feature = self.clip_model.image_encoder(test_image)
        loaded_feature = loaded_analyzer.clip_model.image_encoder(test_image)
        cos_sim = F.cosine_similarity(orig_feature, loaded_feature).item()
        print("�����������ƶ�:", round(cos_sim, 4))
    
    def benchmark_performance(self, batch_size=16):
        """ִ�����ܻ�׼����"""
        print(f"\n=== ���ܻ�׼����(batch_size={batch_size}) ===")
        large_images = torch.rand(batch_size, 3, 224, 224).to(self.device)
        texts = ["Test description"] * batch_size
        
        if self.device == "cuda":
            print("��ʼ�Դ�ռ��:", torch.cuda.memory_allocated() // 1024**2, "MB")
        
        start_time = time.time()
        _ = self.analyzer.calculate_similarity(large_images, texts)
        print(f"�����ʱ: {time.time() - start_time:.2f}s")
        
        if self.device == "cuda":
            print("��ֵ�Դ�ռ��:", torch.cuda.max_memory_allocated() // 1024**2, "MB")

#%% ���������
if __name__ == "__main__":
    # ��ʼ�����Ի���
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache() if device == "cuda" else None
        
        # ��ʼ�����
        clip_model = CLIPModel(device=device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        analyzer = CLIPAlignmentAnalyzer(clip_model, tokenizer, device)
        
        # ִ�к��Ĳ���
        tester = CLIPExtendedTests(device)
        test_scenarios = [
            tester.test_basic_alignment,
            tester.test_edge_cases,
            tester.test_multilingual_support,
            tester.test_model_persistence,
            lambda: tester.benchmark_performance(32)
        ]
        
        for test in test_scenarios:
            try:
                test()
            except Exception as e:
                print(f"����ʧ��: {str(e)}")
        
        # ��չѵ������
        optimizer = torch.optim.AdamW(clip_model.parameters(), lr=1e-4)
        loss_fn = OptimizedCLIPLoss(clip_model, alpha=0.5)
        images = torch.randn(16, 3, 224, 224).to(device)
        texts = ["a photo of cat"]*8 + ["a picture of dog"]*8
        text_inputs = analyzer._prepare_text(texts)
        
        clip_model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            logits = clip_model(images, text_inputs)
            loss = loss_fn(logits, logits.t())
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
            if epoch % 2 == 0:
                clip_model.cm_tracker.visualize(window_size=50)
                
    except Exception as e:
        print(f"����ʱ����: {str(e)}")
        if "CUDA" in str(e):
            print("���� GPU������ CUDA�汾������")
