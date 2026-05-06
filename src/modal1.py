"""Modal 1: Variant-level feature extraction and calibrated classification."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
try:
    from sklearn.calibration import CalibratedClassifierCV
    CALIBRATION_AVAILABLE = True
except ImportError:
    CALIBRATION_AVAILABLE = False
from umap import UMAP


class Modal1_Features_Classifier:
    """
    19개 원본 피처 + Classifier Probability
    UMAP은 시각화/분석용으로만 사용
    """
    def __init__(self, classifier_type='logistic', use_calibration=True, random_state=42):
        self.classifier_type = classifier_type
        self.use_calibration = use_calibration
        self.random_state = random_state
        
        # 분류기 선택
        if classifier_type == 'logistic':
            base_clf = LogisticRegression(random_state=random_state, max_iter=1000)
        elif classifier_type == 'svm':
            base_clf = SVC(kernel='rbf', probability=True, random_state=random_state)
        
        # Calibration 래퍼 (사용 가능한 경우만)
        if use_calibration and CALIBRATION_AVAILABLE:
            self.classifier = CalibratedClassifierCV(
                base_clf, 
                method='sigmoid',  # 🔥 isotonic → sigmoid (더 부드러운 보정)
                cv=3  # 3-fold CV로 보정
            )
            print(f"✓ Using calibrated {classifier_type} classifier (sigmoid method)")
        else:
            self.classifier = base_clf
            if use_calibration and not CALIBRATION_AVAILABLE:
                print(f"⚠ Calibration requested but not available (sklearn version issue)")
                print(f"  Using uncalibrated {classifier_type} classifier")
        
        self.scaler = StandardScaler()
        
        # UMAP은 시각화용 (Fusion에는 사용 안 함!)
        self.umap = UMAP(
            n_components=2,
            random_state=random_state,
            metric='euclidean',
            n_neighbors=30,
            min_dist=0.0,
            target_weight=0.5
        )
        self.umap_embedding = None  # 시각화용으로만 저장
        
    def fit(self, X_all, y_all, labeled_mask=None):
        """
        전체 데이터로 학습
        
        Parameters:
        -----------
        X_all: (n_all_samples, n_features) - VUS 포함 전체 입력 피처
        y_all: (n_all_samples,) - 라벨 (Pathogenic=1, Benign=0, VUS=-1)
        labeled_mask: (n_all_samples,) - True/False, 라벨이 있는 샘플만 True
        """
        # labeled_mask가 없으면 y != -1인 것들만 사용
        if labeled_mask is None:
            labeled_mask = (y_all != -1)
        
        labeled_count = labeled_mask.sum() if hasattr(labeled_mask, 'sum') else sum(labeled_mask)
        
        # 정규화: labeled 데이터로만 fit! (데이터 누수 방지)
        if labeled_count > 0:
            X_labeled = X_all[labeled_mask]
            self.scaler.fit(X_labeled)
            print(f"✓ Scaler fitted on {labeled_count} labeled samples only (no data leakage)")
        else:
            print("⚠ Warning: No labeled data. Fitting scaler on all data.")
            self.scaler.fit(X_all)
        
        # 전체 데이터 변환
        X_scaled = self.scaler.transform(X_all)
        
        # 1. UMAP 학습 (시각화용 - P/B만으로)
        if labeled_count > 0:
            X_labeled = X_scaled[labeled_mask]
            y_labeled = y_all[labeled_mask]
            
            print(f"[UMAP] Training on {labeled_count} labeled samples (for visualization)...")
            self.umap.fit(X_labeled, y=y_labeled)
            
            # 전체 데이터 변환 (시각화용)
            self.umap_embedding = self.umap.transform(X_scaled)
            print(f"✓ UMAP embedding created: {self.umap_embedding.shape}")
        else:
            print("Warning: No labeled data. Using unsupervised UMAP.")
            self.umap_embedding = self.umap.fit_transform(X_scaled)
        
        # 2. Classifier 학습 (19D + UMAP 2D = 21D 특성으로!)
        if labeled_count > 0:
            X_train = X_scaled[labeled_mask]
            y_train = y_all[labeled_mask]
            
            if len(np.unique(y_train)) >= 2:
                # UMAP 좌표도 포함 (transform과 동일하게!)
                umap_train = self.umap_embedding[labeled_mask]  # (n, 2)
                X_train_with_umap = np.hstack([X_train, umap_train])  # (n, 21)
                
                print(f"\n[Classifier] Training on {labeled_count} labeled samples (21D features: 19D + UMAP 2D)...")
                self.classifier.fit(X_train_with_umap, y_train)
                
                # 학습 정확도 확인
                train_acc = self.classifier.score(X_train_with_umap, y_train)
                print(f"✓ Classifier trained: P={sum(y_train==1)}, B={sum(y_train==0)}")
                print(f"  Training accuracy: {train_acc:.3f}")
            else:
                print(f"⚠ Warning: Only {len(np.unique(y_train))} class found. Cannot train classifier.")
        else:
            print("⚠ Warning: No labeled data for classifier training.")
        
        return self
    
    def transform(self, X):
        """
        Fusion용 특성 추출: 원본 19D + Probability
        
        핵심 로직:
        1. Classifier는 21D(19D + UMAP 2D)로 학습됨
        2. 따라서 predict_proba도 21D를 입력받아야 함
        3. 하지만 Fusion에는 20D(19D + Prob)만 전달
           → UMAP 정보는 Probability에 이미 반영됨
        
        Parameters:
        -----------
        X: (n_samples, 19) - 원본 특성
        
        Returns:
        --------
        summary_features: (n_samples, 20) = [19D features + P(Pathogenic)]
        """
        X_scaled = self.scaler.transform(X)
        
        # 1. UMAP 좌표 계산 (Classifier 입력용)
        try:
            umap_coords = self.umap.transform(X_scaled)  # (n, 2)
        except Exception as e:
            print(f"Warning: UMAP transform failed: {e}")
            print("Using zero coordinates")
            umap_coords = np.zeros((len(X), 2))
        
        # 2. Classifier 입력: 19D + UMAP 2D = 21D (학습 시와 동일!)
        X_for_clf = np.hstack([X_scaled, umap_coords])
        
        # 3. Probability 계산 (21D 입력!)
        try:
            proba = self.classifier.predict_proba(X_for_clf)[:, 1]  # P(Pathogenic)
            
            # 확률 clipping (포화 방지)
            proba = np.clip(proba, 0.01, 0.99)
            proba_values = proba.reshape(-1, 1)
        except (AttributeError, ValueError) as e:
            # 분류기 학습 안 된 경우 0.5로
            print(f"Warning: Classifier prediction failed: {e}")
            print("Using 0.5 as default probability.")
            proba_values = np.full((len(X), 1), 0.5)
        
        # 4. Fusion 입력: 원본 19D + Probability = 20D
        #    (UMAP은 Classifier 예측에만 사용, Fusion에는 안 들어감)
        summary_features = np.hstack([X_scaled, proba_values])
        # (n, 20) = [19D normalized features + P(Pathogenic)]
        
        return summary_features
    
    def fit_transform(self, X_all, y_all, labeled_mask=None):
        """
        fit과 transform을 한 번에
        """
        self.fit(X_all, y_all, labeled_mask)
        return self.transform(X_all)


# ==================== Modal 2: GCN for PPI Network ====================