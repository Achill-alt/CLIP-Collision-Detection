import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # 视觉主干网络（示例用线性层代替）
        self.visual_backbone = nn.Sequential(
            nn.Linear(3*224*224, 768),  # 假设输入为224x224 RGB图像
            nn.ReLU()
        )
        
        # 语义分支（处理语义输入）
        self.semantic_proj = nn.Linear(768, 768)  # 保持维度一致
        
        # 多模态融合层（调整输入维度匹配）
        self.fusion_layer = nn.Linear(768 * 2, 768)  # 拼接后维度翻倍
        
        # 最终投影层（调整输出维度）
        self.final_proj = nn.Linear(768, 512)  # 正确维度配置

    def forward(self, image, semantic=None, weights=None):
        # 视觉特征提取
        visual_feat = self.visual_backbone(image.flatten(1))  # [B, 768]
        
        # 分支处理
        if semantic is not None:
            semantic_feat = self.semantic_proj(semantic)  # [B, 768]
            # 多模态融合（拼接）
            fused_feat = torch.cat([visual_feat, semantic_feat], dim=1)  # [B, 1536]
            visual_feat = self.fusion_layer(fused_feat)  # [B, 768]
        
        # 加权融合（示例）
        if weights is not None:
            # 假设权重应用于多尺度特征（需调整实际实现）
            pass  
        
        # 最终投影
        output = self.final_proj(visual_feat)  # [B, 512]
        return output

# 验证代码
if __name__ == "__main__":
    image_encoder = ImageEncoder()
    dummy_image = torch.randn(2, 3, 224, 224)  # 输入形状 [2, 3, 224, 224]
    dummy_semantic = torch.randn(2, 768)
    
    # 三种调用方式验证
    output1 = image_encoder(dummy_image)  # 仅视觉
    output2 = image_encoder(dummy_image, dummy_semantic)  # 视觉+语义
    output3 = image_encoder(dummy_image, dummy_semantic, weights=[0.8, 0.2])  # 加权
    
    print(f"Output shapes: {output1.shape}, {output2.shape}, {output3.shape}")