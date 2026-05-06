"""MultimodalRankingModel: end-to-end pipeline orchestration."""

import numpy as np
import torch
import torch.nn as nn

from .modal1 import Modal1_Features_Classifier
from .modal2 import Modal2_GCN
from .fusion import CrossModalFusionMLP, CrossModalAttention


class MultimodalRankingModel:
    """
    전체 파이프라인을 통합한 멀티모달 랭킹 모델
    """
    def __init__(self, 
                 # Modal 1 파라미터
                 n_input_features=19,  # 입력 특성 개수
                 classifier_type='logistic',
                 use_calibration=True,  # Classifier 확률 보정 사용 여부
                 # Modal 2 파라미터 (GCN)
                 node_feature_dim=None,
                 gcn_hidden_dims=[64, 32],
                 gcn_output_dim=16,
                 # Fusion 파라미터
                 use_cross_attention=True,  # Cross-modal attention 사용 여부
                 attention_dim=64,
                 mlp_hidden_dims=[128, 64, 32],
                 dropout=0.3,
                 # 🔥 Score saturation 방지
                 temperature=1.0,  # Temperature scaling (>1이면 확률이 부드러워짐)
                 logits_clip=10.0,  # Logits clipping 범위
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.device = device
        self.use_cross_attention = use_cross_attention
        self.n_input_features = n_input_features
        self.temperature = temperature  # 🔥 Temperature scaling
        self.logits_clip = logits_clip  # 🔥 Logits clipping
        
        # Modal 1: 원본 특성 + Calibrated Classifier
        self.modal1 = Modal1_Features_Classifier(
            classifier_type=classifier_type,
            use_calibration=use_calibration
        )
        
        # Modal 2: GCN
        if node_feature_dim is not None:
            self.modal2_gcn = Modal2_GCN(
                input_dim=node_feature_dim,
                hidden_dims=gcn_hidden_dims,
                output_dim=gcn_output_dim,
                dropout=dropout
            ).to(device)
        else:
            self.modal2_gcn = None
        
        # Fusion MLP (Cross-attention 사용 여부에 따라 선택)
        modal1_dim = n_input_features + 1  # 19D + P(Pathogenic) = 20D
        modal2_dim = gcn_output_dim
        
        if use_cross_attention:
            self.fusion_mlp = CrossModalFusionMLP(
                modal1_dim=modal1_dim,
                modal2_dim=modal2_dim,
                attention_dim=attention_dim,
                hidden_dims=mlp_hidden_dims,
                dropout=dropout
            ).to(device)
        else:
            self.fusion_mlp = LateFusionMLP(
                modal1_dim=modal1_dim,
                modal2_dim=modal2_dim,
                hidden_dims=mlp_hidden_dims,
                dropout=dropout
            ).to(device)
        
        print(f"\n{'='*70}")
        print(f"Multimodal Ranking Model Initialized")
        print(f"{'='*70}")
        print(f"Modal 1: {n_input_features}D features → Classifier (21D: 19D+UMAP) → {modal1_dim}D")
        print(f"  → Classifier learns with: 19D original + 2D UMAP = 21D")
        print(f"  → Fusion receives: 19D original + 1D Probability = 20D")
        print(f"  → UMAP info is embedded in Probability!")
        print(f"Modal 2: GCN ({node_feature_dim}D → {gcn_output_dim}D)")
        print(f"Fusion: Cross-attention={'ON' if use_cross_attention else 'OFF'} (Fixed: No batch leakage)")
        print(f"Total fusion input: {modal1_dim}D + {modal2_dim}D = {modal1_dim + modal2_dim}D")
        print(f"Device: {device}")
        print(f"{'='*70}\n")
        
    def train_modal1(self, X_all, y_all, labeled_mask=None):
        """
        Modal 1 학습 (VUS 포함 전체 데이터 사용)
        
        Parameters:
        -----------
        X_all: (n_all, n_features) - VUS 포함 전체 변이 (19D)
        y_all: (n_all,) - 라벨 (VUS는 -1)
        labeled_mask: (n_all,) - 라벨 있는 샘플만 True (None이면 자동 생성)
        """
        print(f"\n{'='*70}")
        print(f"Training Modal 1 (Features + Classifier)")
        print(f"{'='*70}")
        self.modal1.fit(X_all, y_all, labeled_mask)
        print(f"{'='*70}\n")
        
    def train_fusion(self, 
                     X_tabular, 
                     node_features, 
                     adjacency_matrix, 
                     node_indices,
                     y_binary, 
                     epochs=100, 
                     lr=0.001, 
                     batch_size=32):
        """
        Fusion MLP 학습 (이진 분류)
        
        X_tabular: (n_samples, 19) - Modal 1 입력 (원본 특성)
        node_features: (n_nodes, feature_dim) - 그래프 노드 특성
        adjacency_matrix: (n_nodes, n_nodes) - 인접 행렬
        node_indices: (n_samples,) - 각 샘플에 해당하는 노드 인덱스
        y_binary: (n_samples,) - 이진 라벨 (0 or 1)
        """
        print(f"\n{'='*70}")
        print(f"Training Fusion Model")
        print(f"{'='*70}")
        
        # Modal 1 특성 추출 (19D + Probability)
        modal1_features = self.modal1.transform(X_tabular)  # (n, 20)
        modal1_features = torch.FloatTensor(modal1_features).to(self.device)
        
        print(f"Modal 1 features: {modal1_features.shape}")
        
        # Modal 2 특성 추출 (GCN)
        node_features_tensor = torch.FloatTensor(node_features).to(self.device)
        adj_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)
        adj_normalized = Modal2_GCN.normalize_adjacency(adj_tensor)
        
        with torch.no_grad():
            all_node_embeddings = self.modal2_gcn(node_features_tensor, adj_normalized)
        
        modal2_features = all_node_embeddings[node_indices]
        
        print(f"Modal 2 features: {modal2_features.shape}")
        
        # 타겟 준비 (이진 라벨)
        y_tensor = torch.FloatTensor(y_binary).to(self.device)
        
        # 학습 (Weight decay 추가로 regularization 강화)
        optimizer = torch.optim.Adam(
            self.fusion_mlp.parameters(), 
            lr=lr, 
            weight_decay=1e-4  # 🔥 L2 regularization 추가
        )
        criterion = nn.BCEWithLogitsLoss()  # Logits 직접 사용 (수치 안정성 개선)
        
        dataset_size = len(y_binary)
        num_batches = (dataset_size + batch_size - 1) // batch_size  # ceil
        
        print(f"\nTraining for {epochs} epochs (batch_size={batch_size}, {num_batches} batches)...")
        
        for epoch in range(epochs):
            self.fusion_mlp.train()
            epoch_loss = 0
            
            # 미니배치 학습
            indices = torch.randperm(dataset_size)
            for i in range(0, dataset_size, batch_size):
                batch_indices = indices[i:i+batch_size]
                
                batch_modal1 = modal1_features[batch_indices]
                batch_modal2 = modal2_features[batch_indices]
                batch_y = y_tensor[batch_indices]
                
                optimizer.zero_grad()
                logits = self.fusion_mlp(batch_modal1, batch_modal2)  # logits (not proba)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches  # 배치당 평균 손실
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")
        
        print(f"{'='*70}\n")
    
    def predict(self, X_tabular, node_features, adjacency_matrix, node_indices):
        """
        최종 랭킹 스코어 예측 (Raw Logits)
        
        🎯 핵심 개념:
        - Logits = 모델의 "생 시그널" (raw confidence)
        - 높을수록 positive class에 가까움
        - Ranking에는 logits를 직접 사용! (상대적 순서만 중요)
        - 확률은 해석용으로만 사용 (predict_proba 메서드 참조)
        
        Returns:
            numpy.ndarray: Raw logits (ranking score)
        """
        self.fusion_mlp.eval()
        
        # Modal 1 특성 (19D + Probability)
        modal1_features = self.modal1.transform(X_tabular)
        modal1_features = torch.FloatTensor(modal1_features).to(self.device)
        
        # Modal 2 특성
        node_features_tensor = torch.FloatTensor(node_features).to(self.device)
        adj_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)
        adj_normalized = Modal2_GCN.normalize_adjacency(adj_tensor)
        
        with torch.no_grad():
            all_node_embeddings = self.modal2_gcn(node_features_tensor, adj_normalized)
            modal2_features = all_node_embeddings[node_indices]
            
            # 🔥 Logits 예측 (이게 ranking score!)
            logits = self.fusion_mlp(modal1_features, modal2_features)
        
        # Logits를 그대로 반환 (ranking에 최적)
        return logits.cpu().numpy().flatten()
    
    def predict_proba(self, X_tabular, node_features, adjacency_matrix, node_indices):
        """
        확률 예측 (해석용)
        
        이 메서드는 사람이 이해하기 쉬운 0~1 확률로 변환합니다.
        실제 ranking에는 predict()의 logits를 사용하세요!
        
        Returns:
            numpy.ndarray: Sigmoid 확률 (0~1)
        """
        self.fusion_mlp.eval()
        
        # Modal 1 특성
        modal1_features = self.modal1.transform(X_tabular)
        modal1_features = torch.FloatTensor(modal1_features).to(self.device)
        
        # Modal 2 특성
        node_features_tensor = torch.FloatTensor(node_features).to(self.device)
        adj_tensor = torch.FloatTensor(adjacency_matrix).to(self.device)
        adj_normalized = Modal2_GCN.normalize_adjacency(adj_tensor)
        
        with torch.no_grad():
            all_node_embeddings = self.modal2_gcn(node_features_tensor, adj_normalized)
            modal2_features = all_node_embeddings[node_indices]
            
            # Logits 예측
            logits = self.fusion_mlp(modal1_features, modal2_features)
            
            # 🔥 Temperature scaling (확률을 부드럽게)
            scaled_logits = logits / self.temperature
            
            # Sigmoid로 확률 변환
            probabilities = torch.sigmoid(scaled_logits)
        
        return probabilities.cpu().numpy().flatten()