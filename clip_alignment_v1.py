import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # �Ӿ��������磨ʾ�������Բ���棩
        self.visual_backbone = nn.Sequential(
            nn.Linear(3*224*224, 768),  # ��������Ϊ224x224 RGBͼ��
            nn.ReLU()
        )
        
        # �����֧�������������룩
        self.semantic_proj = nn.Linear(768, 768)  # ����ά��һ��
        
        # ��ģ̬�ںϲ㣨��������ά��ƥ�䣩
        self.fusion_layer = nn.Linear(768 * 2, 768)  # ƴ�Ӻ�ά�ȷ���
        
        # ����ͶӰ�㣨�������ά�ȣ�
        self.final_proj = nn.Linear(768, 512)  # ��ȷά������

    def forward(self, image, semantic=None, weights=None):
        # �Ӿ�������ȡ
        visual_feat = self.visual_backbone(image.flatten(1))  # [B, 768]
        
        # ��֧����
        if semantic is not None:
            semantic_feat = self.semantic_proj(semantic)  # [B, 768]
            # ��ģ̬�ںϣ�ƴ�ӣ�
            fused_feat = torch.cat([visual_feat, semantic_feat], dim=1)  # [B, 1536]
            visual_feat = self.fusion_layer(fused_feat)  # [B, 768]
        
        # ��Ȩ�ںϣ�ʾ����
        if weights is not None:
            # ����Ȩ��Ӧ���ڶ�߶������������ʵ��ʵ�֣�
            pass  
        
        # ����ͶӰ
        output = self.final_proj(visual_feat)  # [B, 512]
        return output

# ��֤����
if __name__ == "__main__":
    image_encoder = ImageEncoder()
    dummy_image = torch.randn(2, 3, 224, 224)  # ������״ [2, 3, 224, 224]
    dummy_semantic = torch.randn(2, 768)
    
    # ���ֵ��÷�ʽ��֤
    output1 = image_encoder(dummy_image)  # ���Ӿ�
    output2 = image_encoder(dummy_image, dummy_semantic)  # �Ӿ�+����
    output3 = image_encoder(dummy_image, dummy_semantic, weights=[0.8, 0.2])  # ��Ȩ
    
    print(f"Output shapes: {output1.shape}, {output2.shape}, {output3.shape}")