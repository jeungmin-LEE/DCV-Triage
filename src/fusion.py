"""Cross-modal attention fusion and MLP classifiers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossModalAttention(nn.Module):
    """
    두 모달 간 상호 attention (샘플별 스칼라 게이트 방식)
    각 샘플이 독립적으로 처리되어 배치 간 누수 없음
    + 투영 레이어로 상대 모달 정보 직접 주입
    """
    def __init__(self, modal1_dim, modal2_dim, attention_dim=64):
        super(CrossModalAttention, self).__init__()
        
        # Modal 1 → Modal 2 게이트 (샘플별 스칼라)
        self.gate_m1_to_m2 = nn.Sequential(
            nn.Linear(modal1_dim + modal2_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()  # 0~1 게이트 값
        )
        
        # Modal 2 → Modal 1 게이트 (샘플별 스칼라)
        self.gate_m2_to_m1 = nn.Sequential(
            nn.Linear(modal1_dim + modal2_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, 1),
            nn.Sigmoid()  # 0~1 게이트 값
        )
        
        # 차원 맞추기 위한 투영 레이어
        self.proj_m2_to_m1 = nn.Linear(modal2_dim, modal1_dim)  # 16D → 20D
        self.proj_m1_to_m2 = nn.Linear(modal1_dim, modal2_dim)  # 20D → 16D
        
    def forward(self, modal1_features, modal2_features):
        """
        modal1_features: (batch_size, modal1_dim) = 20D
        modal2_features: (batch_size, modal2_dim) = 16D
        
        Returns:
            attended_m1: gate * m1 + (1-gate) * Proj(m2)
            attended_m2: gate * m2 + (1-gate) * Proj(m1)
        """
        # 두 모달 결합
        combined = torch.cat([modal1_features, modal2_features], dim=1)
        
        # 샘플별 게이트 값 계산 (배치 간 독립)
        gate_12 = self.gate_m1_to_m2(combined)  # (B, 1) - Modal1이 Modal2 수용도
        gate_21 = self.gate_m2_to_m1(combined)  # (B, 1) - Modal2가 Modal1 수용도
        
        # 상대 모달 정보 투영
        m2_projected = self.proj_m2_to_m1(modal2_features)  # (B, 20) - Modal2를 Modal1 공간으로
        m1_projected = self.proj_m1_to_m2(modal1_features)  # (B, 16) - Modal1을 Modal2 공간으로
        
        # 게이트를 이용한 정보 융합
        # gate=1: 자기 자신만, gate=0: 상대방 정보만
        attended_m1 = gate_21 * modal1_features + (1 - gate_21) * m2_projected
        attended_m2 = gate_12 * modal2_features + (1 - gate_12) * m1_projected
        
        return attended_m1, attended_m2


# ==================== Cross-Modal Fusion MLP ====================
class CrossModalFusionMLP(nn.Module):
    """
    Cross-modal attention을 적용한 후 융합하여 최종 랭킹 스코어 출력
    """
    def __init__(self, modal1_dim, modal2_dim, attention_dim=64, hidden_dims=[128, 64, 32], dropout=0.3):
        super(CrossModalFusionMLP, self).__init__()
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(modal1_dim, modal2_dim, attention_dim)
        
        # Fusion network
        # 원본 + attended 특성 모두 사용
        input_dim = (modal1_dim * 2) + (modal2_dim * 2)  # 원본 + attended
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 마지막 레이어 제외
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        # 마지막 Sigmoid 제거 (BCEWithLogitsLoss 사용을 위해)
        # layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, modal1_features, modal2_features):
        """
        modal1_features: (batch_size, modal1_dim) = 20D
        modal2_features: (batch_size, modal2_dim) = 16D
        
        Returns: logits (not probabilities!)
        """
        # Cross-modal attention 적용
        attended_m1, attended_m2 = self.cross_attention(modal1_features, modal2_features)
        
        # 원본 특성 + attended 특성을 모두 결합
        fused = torch.cat([
            modal1_features, attended_m1,  # Modal 1: 20D + 20D
            modal2_features, attended_m2   # Modal 2: 16D + 16D
        ], dim=1)  # Total: 72D
        
        score = self.network(fused)
        return score.squeeze(-1)


# ==================== Late Fusion MLP (기존 버전 유지) ====================
class LateFusionMLP(nn.Module):
    """
    두 모달의 특성을 결합하여 최종 랭킹 스코어 출력 (단순 concatenation)
    """
    def __init__(self, modal1_dim, modal2_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super(LateFusionMLP, self).__init__()
        
        input_dim = modal1_dim + modal2_dim
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # 마지막 레이어 제외
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        
        # 마지막 Sigmoid 제거 (BCEWithLogitsLoss 사용)
        # layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, modal1_features, modal2_features):
        """
        modal1_features: (batch_size, modal1_dim)
        modal2_features: (batch_size, modal2_dim)
        
        Returns: logits (not probabilities!)
        """
        # Late fusion: 단순 concatenation
        fused = torch.cat([modal1_features, modal2_features], dim=1)
        score = self.network(fused)
        return score.squeeze(-1)


# ==================== 통합 멀티모달 모델 ====================