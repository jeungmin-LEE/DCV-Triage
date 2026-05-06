"""Data preparation: variant annotation loading, PPI graph construction, and feature engineering."""

import numpy as np
import pandas as pd
import networkx as nx
from scipy import sparse


# ==================== 사용 예시 ====================
def prepare_variant_data(annovar_file, string_file, mm_genes, 
                        label_column='CLNSIG', gene_column='Gene.refGene',
                        alias_file=None,
                        rwr_restart_prob=0.15, ppr_alpha=0.15,
                        use_prediction_labels=False,
                        exonic_only=True, remove_super_hubs=True):
    """
    전체 데이터 준비 파이프라인
    
    Parameters:
    -----------
    annovar_file: str
        Annovar 주석 파일 경로
    string_file: str
        STRING PPI 파일 경로
    mm_genes: list
        MM 관련 허브 유전자 리스트 (RWR/PPR seed, PPI에 강제 추가됨)
    label_column: str
        병원성 라벨 컬럼명 (기본: 'CLNSIG')
    gene_column: str
        유전자 컬럼명
    alias_file: str
        STRING alias 파일 경로 (None이면 자동 추론)
    rwr_restart_prob: float
        RWR 재시작 확률 (기본 0.15, 직접 구현)
    ppr_alpha: float
        PPR 텔레포트 확률 (기본 0.15, NetworkX 구현)
    remove_super_hubs: bool
        슈퍼 허브(UBC, Histone 등) 점수 제거 여부 (기본: True)
    use_prediction_labels: bool
        ClinVar 없을 때 예측 스코어로 라벨 생성 여부
    exonic_only: bool
        Exonic variant만 사용 여부 (기본 True)
        
    Returns:
    --------
    X_train, y_train, X_all, y_all, node_features, adjacency_matrix,
    node_indices_train, node_indices_all, df_all, loader
    """
    
    loader = VariantDataLoader()
    
    print("="*70)
    print("MM Pathogenic Variant Prediction - Data Preparation")
    print("="*70)
    
    # 1. Annovar 데이터 로딩
    df_all = loader.load_annovar_data(annovar_file, label_column)
    
    # 1.5. Exonic+Splicing 필터링, Synonymous SNV 제외 (옵션)
    if exonic_only and 'Func.refGene' in df_all.columns:
        before = len(df_all)
        
        # Exonic OR Splicing 변이 포함
        functional_mask = (
            df_all['Func.refGene'].str.contains('exonic', case=False, na=False) |
            df_all['Func.refGene'].str.contains('splicing', case=False, na=False)
        )
        df_all = df_all[functional_mask].reset_index(drop=True)
        
        after_func = len(df_all)
        print(f"\n✓ Filtered to exonic/splicing variants: {after_func} (from {before:,})")
        
        # Synonymous SNV 제외 (ExonicFunc.refGene 컬럼 사용)
        if 'ExonicFunc.refGene' in df_all.columns:
            before_syn = len(df_all)
            syn_mask = df_all['ExonicFunc.refGene'].str.contains('synonymous SNV', case=False, na=False)
            df_all = df_all[~syn_mask].reset_index(drop=True)
            
            print(f"✓ Excluded synonymous SNV: {len(df_all)} (removed {before_syn - len(df_all)})")
        
        print(f"  Total filtering rate: {len(df_all)/before*100:.1f}%")
    
    # 2. 특성 추출 (전체)
    X_all, feature_names = loader.preprocess_features(df_all)
    
    # 3. 라벨 추출
    if use_prediction_labels:
        print("\n⚠ Using prediction scores as labels (no ClinVar data)")
        # CADD score 기반 라벨 생성
        y_all_temp, labeled_idx = loader.create_prediction_labels(df_all)
    else:
        # ClinVar 라벨 사용
        y_all_temp, labeled_idx = loader.extract_labels(df_all, label_column, include_vus=False)
    
    # y_all 생성 (전체 데이터에 대해 -1 채우기)
    y_all = np.full(len(df_all), -1)
    if labeled_idx is not None and len(labeled_idx) > 0:
        y_all[labeled_idx] = y_all_temp
        
        # 학습용 데이터
        y_train = y_all_temp
        X_train = X_all[labeled_idx]
        df_train = df_all.iloc[labeled_idx].reset_index(drop=True)
    else:
        print("\n❌ ERROR: No training data available!")
        y_train = np.array([])
        X_train = np.array([]).reshape(0, X_all.shape[1])
        df_train = df_all.iloc[:0]
    
    # 4. Gene 매핑 (전체 & 학습용)
    gene_list_all = loader.extract_gene_mapping(df_all, gene_column)
    gene_list_train = loader.extract_gene_mapping(df_train, gene_column) if len(df_train) > 0 else []
    
    # 고유 유전자 리스트 (UNKNOWN 제외)
    # gene_list_all은 이제 list of lists
    all_genes_flat = [g for genes in gene_list_all for g in genes]
    unique_genes = set(all_genes_flat) - {'UNKNOWN'}
    
    print(f"\nUnique genes in data: {len(unique_genes)}")
    print(f"MM seed genes: {len(mm_genes)}")
    
    # 5. PPI 네트워크 로딩 (데이터 유전자 + MM seed 강제 추가)
    if alias_file:
        ppi_graph = loader.load_string_with_gene_filter(
            string_file=string_file,
            alias_file=alias_file,
            my_genes=list(unique_genes),
            mm_seed_genes=mm_genes,  # 👈 MM 유전자를 PPI에 강제 추가!
            score_threshold=400
        )
    else:
        print("\n⚠ WARNING: No alias file provided. Using unfiltered PPI (may be slow).")
        ppi_graph = loader.load_string_ppi(string_file, alias_file=None, score_threshold=400)
        # MM 유전자 수동 추가
        for mm_gene in mm_genes:
            if mm_gene not in ppi_graph.nodes():
                ppi_graph.add_node(mm_gene)
        print(f"✓ Added {len(mm_genes)} MM genes to PPI")
    
    # 7. RWR + PPR 계산 (MM 허브 유전자로부터, 슈퍼 허브 제거)
    print("\n" + "="*70)
    print("Computing Network Propagation Scores (RWR + PPR)")
    print("="*70)
    
    # 7-1. RWR 계산 (직접 구현)
    rwr_scores = loader.compute_rwr_from_seeds(
        ppi_graph, mm_genes, 
        restart_prob=rwr_restart_prob,
        remove_super_hubs=remove_super_hubs
    )
    
    # 7-2. PPR 계산 (NetworkX 구현)
    ppr_scores = loader.compute_ppr_from_seeds(
        ppi_graph, mm_genes, 
        alpha=ppr_alpha,
        remove_super_hubs=remove_super_hubs
    )
    
    # 7-3. Seed 유전자의 RWR/PPR 점수를 0으로 설정 (bias 방지)
    # 목적: seed "주변" 유전자 발견, seed 자체는 이미 알려짐
    # Circular reasoning 방지 - 이미 아는 유전자가 높은 점수 받는 건 의미 없음
    print("\n[Seed Gene Exclusion from Ranking]")
    excluded_rwr = 0
    excluded_ppr = 0
    
    for seed in mm_genes:
        if seed in rwr_scores and rwr_scores[seed] > 0:
            rwr_scores[seed] = 0.0
            excluded_rwr += 1
        if seed in ppr_scores and ppr_scores[seed] > 0:
            ppr_scores[seed] = 0.0
            excluded_ppr += 1
    
    if excluded_rwr > 0:
        print(f"✓ RWR: Set {excluded_rwr} seed genes to 0 (avoid circular reasoning)")
    if excluded_ppr > 0:
        print(f"✓ PPR: Set {excluded_ppr} seed genes to 0 (avoid circular reasoning)")
    
    print(f"  → Seed genes excluded from ranking to prevent bias")
    print(f"  → Goal: Discover NOVEL genes around seeds, not re-discover seeds themselves")
    
    # Top 10 genes by RWR (seed 제외 후)
    print(f"\n[Top 10 genes by RWR score (seeds excluded)]")
    sorted_rwr = sorted(rwr_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for gene, score in sorted_rwr:
        print(f"  {gene}: {score:.6f}")
    
    # Top 10 genes by PPR (seed 제외 후)
    print(f"\n[Top 10 genes by PPR score (seeds excluded)]")
    sorted_ppr = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    for gene, score in sorted_ppr:
        print(f"  {gene}: {score:.6f}")
    
    # RWR과 PPR 점수 비교
    print(f"\n[Score Correlation Analysis]")
    common_genes = set(rwr_scores.keys()) & set(ppr_scores.keys())
    if len(common_genes) > 0:
        rwr_vals = np.array([rwr_scores[g] for g in common_genes])
        ppr_vals = np.array([ppr_scores[g] for g in common_genes])
        correlation = np.corrcoef(rwr_vals, ppr_vals)[0, 1]
        print(f"✓ RWR-PPR correlation: {correlation:.4f}")
        if correlation > 0.9:
            print(f"  → High agreement (>0.9): Both methods identify similar candidates")
        elif correlation > 0.7:
            print(f"  → Good agreement (0.7-0.9): Methods generally consistent")
        else:
            print(f"  → Moderate agreement (<0.7): Methods provide complementary information")
    
    # 8. 노드 인덱스 매핑
    node_indices_all = loader.create_gene_to_node_mapping(gene_list_all, ppi_graph)
    node_indices_train = loader.create_gene_to_node_mapping(gene_list_train, ppi_graph)
    
    # 9. 노드 특성 생성 (RWR + PPR 둘 다 포함!)
    node_features = loader.create_node_features_with_rwr_and_ppr(ppi_graph, rwr_scores, ppr_scores)
    
    # 10. 인접 행렬 생성
    adjacency_matrix = loader.create_adjacency_matrix(ppi_graph)
    
    # 11. Seed 유전자 여부 추가 (결과 분석용)
    # Note: Seed genes는 RWR/PPR 시작점이자 알려진 MM 연관 유전자
    # 이 표시는 결과 분석 단계에서 seed vs novel을 구분하기 위함
    genes_all = [genes[0] if isinstance(genes, list) else genes for genes in gene_list_all]
    df_all['is_seed_gene'] = [g in mm_genes for g in genes_all]
    
    print("\n" + "="*70)
    print("Data preparation completed!")
    print("="*70)
    print(f"[Training Set - Labeled only]")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape} (P={(y_train==1).sum()}, B={(y_train==0).sum()})")
    print(f"  node_indices_train: {node_indices_train.shape}")
    print(f"\n[Full Dataset - Including VUS]")
    print(f"  X_all: {X_all.shape}")
    print(f"  y_all: {y_all.shape} (P={(y_all==1).sum()}, B={(y_all==0).sum()}, VUS={(y_all==-1).sum()})")
    print(f"  node_indices_all: {node_indices_all.shape}")
    print(f"  Seed genes (RWR/PPR starts): {df_all['is_seed_gene'].sum()} variants")
    print(f"\n[Graph Data]")
    print(f"  node_features: {node_features.shape} (8D: centrality + RWR + PPR)")
    print(f"  adjacency_matrix: {adjacency_matrix.shape}")
    print(f"  Total nodes in PPI: {ppi_graph.number_of_nodes()}")
    print(f"  MM seed genes in PPI: {sum([g in ppi_graph.nodes() for g in mm_genes])}/{len(mm_genes)}")
    print("="*70)
    
    return X_train, y_train, X_all, y_all, node_features, adjacency_matrix, \
           node_indices_train, node_indices_all, df_all, loader