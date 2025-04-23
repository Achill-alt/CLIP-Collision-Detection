#!pip install numpy pandas torch  # 安装多个库
#!apt-get install -y libgl1-mesa-glx  # 安装系统依赖

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import ViTModel, BertModel, BertTokenizer
from PIL import Image
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

# ------------------ 核心模型架构（修复路径处理）------------------
class ImageEncoder(nn.Module):
    def __init__(self, freeze_vit_layers=6):
        super().__init__()
        self.visual_backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        # 分层冻结策略
        for layer in self.visual_backbone.encoder.layer[:freeze_vit_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)

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
    def __init__(self, freeze_bert_layers=3):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # 冻结前N层参数
        for layer in self.bert.encoder.layer[:freeze_bert_layers]:
            for param in layer.parameters():
                param.requires_grad_(False)

        self.projection = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_ids, attention_mask):
        features = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        return F.normalize(self.projection(features), dim=-1)

# ------------------ 对齐度计算模块（修复路径处理）------------------
class CLIPAlignmentAnalyzer:
    def __init__(self, clip_model, tokenizer, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = clip_model.to(self.device)
        self.tokenizer = tokenizer
        self.clip_model.eval()

    def _prepare_text(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            max_length=77,  # CLIP标准长度
            truncation=True,
            return_token_type_ids=False
        ).to(self.device)

    def calculate_similarity(self, images, texts):
        """支持批量与单样本的灵活处理"""
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_inputs = self._prepare_text(texts)
            image_features = self.clip_model.image_encoder(images.to(self.device))
            text_features = self.clip_model.text_encoder(**text_inputs)

            logit_scale = self.clip_model.logit_scale.exp()
            similarity = logit_scale * image_features @ text_features.t()
        return similarity.cpu()

    def visualize_alignment(self, similarity_matrix, texts, figsize=(12, 10)):
        """优化可视化显示"""
        plt.figure(figsize=figsize, dpi=120)
        matrix = similarity_matrix.numpy() if torch.is_tensor(similarity_matrix) else similarity_matrix

        heatmap = plt.imshow(matrix, cmap='plasma', aspect='auto')
        plt.colorbar(heatmap, fraction=0.046, pad=0.04)
        plt.xticks(np.arange(len(texts)), texts, rotation=55, ha='right')
        plt.yticks(np.arange(len(texts)), [f"Image {i+1}" for i in range(len(texts))])
        plt.title("Cross-Modal Semantic Alignment", pad=20)
        plt.xlabel("Text Descriptions", labelpad=15)
        plt.ylabel("Visual Content", labelpad=15)
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        """修复保存逻辑：创建专用tokenizer子目录"""
        # 确保父目录存在
        save_dir = os.path.dirname(path) or '.'  # 处理无目录的情况
        os.makedirs(save_dir, exist_ok=True)

        # 创建专用tokenizer子目录
        tokenizer_dir = os.path.join(save_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_dir)

        # 保存模型时记录tokenizer路径
        torch.save({
            'clip_model': self.clip_model.state_dict(),
            'tokenizer_dir': tokenizer_dir,
        }, path)

    @classmethod
    def load_model(cls, path, device=None):
        """修复加载逻辑：使用完整tokenizer路径"""
        current_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 添加安全全局声明（必须在加载前执行）
        torch.serialization.add_safe_globals([BertTokenizer])

        try:
            checkpoint = torch.load(path,
                                  weights_only=True,
                                  map_location=current_device)
        except Exception as e:
            print(f"安全模式加载失败: {str(e)}")
            print("尝试非安全模式加载（仅限可信来源！）")
            checkpoint = torch.load(path,
                                  weights_only=False,
                                  map_location=current_device)

        # 从专用目录加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(checkpoint['tokenizer_dir'])

        # 初始化模型
        clip_model = CLIPModel()
        clip_model.load_state_dict(checkpoint['clip_model'])

        return cls(clip_model, tokenizer, current_device)

# ------------------ 完整CLIP模型（保持原样）------------------
class CLIPModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))

    def forward(self, image, text):
        image_features = self.image_encoder(image)
        text_features = self.text_encoder(**text)
        return image_features, text_features

# ------------------ 使用示例（添加路径验证）------------------
if __name__ == "__main__":
    try:
        # 设备检测与内存优化
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.cuda.empty_cache() if device == "cuda" else None

        # 初始化组件
        clip_model = CLIPModel().to(device)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # 创建分析器
        analyzer = CLIPAlignmentAnalyzer(clip_model, tokenizer, device)

        # 测试数据
        test_images = torch.randn(3, 3, 224, 224).to(device) * 0.5 + 0.5
        test_texts = [
            "A black cat sitting on wooden floor",
            "Golden retriever playing in park",
            "Red sports car on highway"
        ]

        # 计算对齐度
        similarity_matrix = analyzer.calculate_similarity(test_images, test_texts)

        # 结果可视化
        print("\n语义对齐矩阵:")
        print(similarity_matrix.numpy().round(2))
        analyzer.visualize_alignment(similarity_matrix, test_texts)

        # 模型保存与加载测试（指定完整路径）
        model_path = "saved_models/clip_align_model.pth"  # 显式指定目录结构
        analyzer.save_model(model_path)
        loaded_analyzer = CLIPAlignmentAnalyzer.load_model(model_path)

        # 单样本测试
        single_image = torch.randn(1, 3, 224, 224).to(device) * 0.5 + 0.5
        caption = "Cute puppy looking at camera"
        similarity = analyzer.calculate_similarity(single_image, [caption])
        print(f"\n单样本对齐分数: {similarity.item():.2f}")

    except Exception as e:
        print(f"运行时错误: {str(e)}")
        if "CUDA" in str(e):
            print("请检查GPU驱动和CUDA版本兼容性")
# ------------------ 扩展测试模块 ------------------
class CLIPExtendedTests:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.clip_model = CLIPModel().to(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.analyzer = CLIPAlignmentAnalyzer(self.clip_model, self.tokenizer, device)
        
    # ------------------ 基础功能验证 ------------------
    def test_basic_alignment(self):
        """验证基础对齐能力"""
        print("\n=== 基础对齐测试 ===")
        
        # 构造特征匹配数据
        real_images = [
            torch.rand(3, 224, 224).to(self.device) * 0.3 + 0.4,  # 猫图像模拟
            torch.rand(3, 224, 224).to(self.device) * 0.2 + 0.6,  # 狗图像模拟
            torch.rand(3, 224, 224).to(self.device) * 0.5 + 0.3   # 汽车图像模拟
        ]
        batch_images = torch.stack(real_images)
        
        texts = [
            "A gray tabby cat on carpet",
            "Golden retriever playing with ball", 
            "Sports car on racing track"
        ]
        
        similarity = self.analyzer.calculate_similarity(batch_images, texts)
        print("对角线匹配分数:", similarity.diag().tolist())
        self.analyzer.visualize_alignment(similarity, texts)

    # ------------------ 边界情况测试 ------------------
    def test_edge_cases(self):
        """验证异常输入处理能力"""
        print("\n=== 边界情况测试 ===")
        
        # 空文本测试
        empty_text_similarity = self.analyzer.calculate_similarity(
            torch.rand(1, 3, 224, 224).to(self.device), 
            [""]
        )
        print("空文本相似度:", empty_text_similarity.item())
        
        # 无效图像测试
        try:
            invalid_image = torch.rand(2, 3, 512, 512).to(self.device)  # 错误尺寸
            self.analyzer.calculate_similarity(invalid_image, ["Test"])
        except Exception as e:
            print(f"捕获异常：{str(e)}")

    # ------------------ 多语言支持测试 ------------------
    def test_multilingual_support(self):
        """验证多语言描述对齐能力"""
        print("\n=== 多语言支持测试 ===")
        
        chinese_texts = [
            "一只在沙发上睡觉的橘猫",
            "草地上奔跑的金毛犬",
            "高速公路上的红色跑车"
        ]
        
        similarity = self.analyzer.calculate_similarity(
            torch.rand(3, 3, 224, 224).to(self.device),
            chinese_texts
        )
        print("跨语言对齐矩阵:\n", similarity.numpy().round(2))

    # ------------------ 模型持久化测试 ------------------
    def test_model_persistence(self):
        """验证模型保存/加载完整性"""
        print("\n=== 模型持久化测试 ===")
        
        # 保存并重新加载模型
        temp_path = "temp_model.pth"
        self.analyzer.save_model(temp_path)
        loaded_analyzer = CLIPAlignmentAnalyzer.load_model(temp_path)
        
        # 验证特征一致性
        test_image = torch.rand(1, 3, 224, 224).to(self.device)
        orig_feature = self.clip_model.image_encoder(test_image)
        loaded_feature = loaded_analyzer.clip_model.image_encoder(test_image)
        
        print("特征余弦相似度:", 
              F.cosine_similarity(orig_feature, loaded_feature).item())

    # ------------------ 性能基准测试 ------------------
    def benchmark_performance(self, batch_size=16):
        """执行性能基准测试"""
        print(f"\n=== 性能基准测试 (batch_size={batch_size}) ===")
        
        large_images = torch.rand(batch_size, 3, 224, 224).to(self.device)
        texts = ["Test description"] * batch_size
        
        # GPU显存监控
        if self.device == "cuda":
            print("初始显存占用:", torch.cuda.memory_allocated()//1024**2, "MB")
        
        import time
        start_time = time.time()
        _ = self.analyzer.calculate_similarity(large_images, texts)
        print(f"处理耗时: {time.time()-start_time:.2f}s")
        
        if self.device == "cuda":
            print("峰值显存占用:", torch.cuda.max_memory_allocated()//1024**2, "MB")

# ------------------ 执行扩展测试 ------------------
if __name__ == "__main__":
    tester = CLIPExtendedTests()
    
    # 执行测试套件
    test_scenarios = [
        tester.test_basic_alignment,
        tester.test_edge_cases,
        tester.test_multilingual_support,
        tester.test_model_persistence,
        lambda: tester.benchmark_performance(32)  # 大批量测试
    ]
    
    for test in test_scenarios:
        try:
            test()
        except Exception as e:
            print(f"测试失败: {str(e)}")
