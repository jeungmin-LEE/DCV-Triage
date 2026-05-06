#!/usr/bin/env python3
"""
DCV-TRIAGE: Disease-Contextual VUS Triage
==========================================

Usage:
    python run_pipeline.py --annovar <file> --string <file> --aliases <file> --seeds "GENE1,GENE2,..."
    python run_pipeline.py --config config.yaml

Examples:
    # Multiple Myeloma (default seeds)
    python run_pipeline.py \\
        --annovar data/variants.csv \\
        --string data/9606.protein.links.full.v12.0.txt \\
        --aliases data/9606.protein.aliases.v12.0.txt \\
        --output results/mm_analysis/

    # Custom disease with specific seeds
    python run_pipeline.py \\
        --annovar data/variants.csv \\
        --string data/9606.protein.links.full.v12.0.txt \\
        --aliases data/9606.protein.aliases.v12.0.txt \\
        --seeds "INS,GCK,HNF1A,HNF4A" \\
        --output results/diabetes/

    # Using config file
    python run_pipeline.py --config config.yaml
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_prep import prepare_variant_data
from modal1 import Modal1_Features_Classifier
from modal2 import Modal2_GCN
from fusion import CrossModalFusionMLP, CrossModalAttention
from pipeline import MultimodalRankingModel


# Default MM seed genes
DEFAULT_SEEDS = [
    'CRBN', 'FGFR3', 'TP53', 'KRAS', 'TNFRSF17', 'NRAS', 'LIG4', 'BRAF',
    'CXCR4', 'XPO1', 'CD38', 'RBX1', 'CUL4A', 'PSMB5', 'DDB1', 'BCL2',
    'FDPS', 'TUBB4A'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='DCV-TRIAGE: Disease-Contextual VUS Triage',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input files
    parser.add_argument('--annovar', type=str,
                        help='Path to ANNOVAR-annotated variant CSV')
    parser.add_argument('--string', type=str,
                        help='Path to STRING PPI file')
    parser.add_argument('--aliases', type=str,
                        help='Path to STRING alias file')
    
    # Disease context
    parser.add_argument('--seeds', type=str, default=None,
                        help='Comma-separated seed gene list (default: MM seeds)')
    parser.add_argument('--disease', type=str, default='custom',
                        help='Disease name for output labelling')
    
    # Configuration
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file (overrides other args)')
    
    # Output
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory')
    parser.add_argument('--top-k', type=int, nargs='+', default=[20, 50, 100],
                        help='Top-K cutoffs to report (default: 20 50 100)')
    
    # Model parameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Flags
    parser.add_argument('--no-figures', action='store_true',
                        help='Skip figure generation')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal output')
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seeds(seed):
    """Fix all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def run(args):
    """Main pipeline execution."""
    
    # ── Setup ──────────────────────────────────────────────
    set_seeds(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output, exist_ok=True)
    
    # Parse seed genes
    if args.seeds:
        seed_genes = [g.strip() for g in args.seeds.split(',')]
    else:
        seed_genes = DEFAULT_SEEDS
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*60}")
    print(f"  DCV-TRIAGE: Disease-Contextual VUS Triage")
    print(f"  Disease context: {args.disease}")
    print(f"  Seed genes: {len(seed_genes)}")
    print(f"  Device: {device}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")
    
    # ── Step 1: Data preparation ──────────────────────────
    print("[1/4] Preparing variant data...")
    X_train, y_train, X_all, y_all, node_feat, adj, \
        node_idx_train, node_idx_all, df_all, loader = prepare_variant_data(
        annovar_file=args.annovar,
        string_file=args.string,
        mm_genes=seed_genes,
        alias_file=args.aliases,
        label_column='CLNSIG',
        gene_column='Gene.refGene',
        rwr_restart_prob=0.15,
        ppr_alpha=0.15,
        exonic_only=True,
        remove_super_hubs=True
    )
    
    n_total = len(df_all)
    n_labeled = (y_all != -1).sum()
    n_vus = (y_all == -1).sum()
    print(f"  Total variants: {n_total:,}")
    print(f"  Labelled: {n_labeled:,} | VUS: {n_vus:,}")
    print(f"  Seed genes in PPI: {len(seed_genes)}")
    
    # ── Step 2: Model initialisation ──────────────────────
    print("\n[2/4] Initialising model...")
    model = MultimodalRankingModel(
        n_input_features=X_all.shape[1],
        classifier_type='logistic',
        node_feature_dim=node_feat.shape[1],
        gcn_hidden_dims=[64, 32],
        gcn_output_dim=16,
        use_cross_attention=True,
        attention_dim=64,
        mlp_hidden_dims=[128, 64, 32],
        dropout=0.3
    )
    
    # ── Step 3: Training ──────────────────────────────────
    print("\n[3/4] Training...")
    model.fit(
        X_all=X_all,
        y_all=y_all,
        node_feat=node_feat,
        adj=adj,
        node_idx_train=node_idx_train,
        node_idx_all=node_idx_all,
        epochs=args.epochs,
        lr=args.lr,
        device=device
    )
    
    # ── Step 4: Ranking ───────────────────────────────────
    print("\n[4/4] Generating rankings...")
    df_all['ranking_score'] = model.predict(
        X_all=X_all,
        node_feat=node_feat,
        adj=adj,
        node_idx_all=node_idx_all,
        device=device
    )
    
    # VUS only, exclude seed genes
    vus_mask = (y_all == -1)
    seed_mask = df_all['Gene.refGene'].isin(seed_genes)
    df_vus = df_all[vus_mask & ~seed_mask].copy()
    df_vus = df_vus.sort_values('ranking_score', ascending=False)
    
    # Gene-level ranking
    gene_ranking = df_vus.groupby('Gene.refGene')['ranking_score'].max() \
                        .sort_values(ascending=False).reset_index()
    gene_ranking.columns = ['Gene', 'Score']
    gene_ranking['Rank'] = range(1, len(gene_ranking) + 1)
    
    # ── Save results ──────────────────────────────────────
    print(f"\nSaving results to {args.output}/")
    
    # Full rankings
    df_all.to_csv(os.path.join(args.output, 'full_variant_ranking.csv'), index=False)
    gene_ranking.to_csv(os.path.join(args.output, 'full_gene_ranking.csv'), index=False)
    
    # Top-K reports
    for k in args.top_k:
        topk = gene_ranking.head(k)
        topk.to_csv(os.path.join(args.output, f'top{k}_genes.csv'), index=False)
    
    # Summary report
    with open(os.path.join(args.output, 'summary.txt'), 'w') as f:
        f.write(f"DCV-TRIAGE Analysis Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Date: {timestamp}\n")
        f.write(f"Disease: {args.disease}\n")
        f.write(f"Seed genes: {', '.join(seed_genes)}\n")
        f.write(f"Total variants: {n_total:,}\n")
        f.write(f"Labelled: {n_labeled:,}\n")
        f.write(f"VUS pool: {n_vus:,}\n")
        f.write(f"Unique genes ranked: {len(gene_ranking):,}\n")
        f.write(f"\n")
        for k in args.top_k:
            topk = gene_ranking.head(k)
            f.write(f"Top-{k} genes:\n")
            for _, row in topk.iterrows():
                f.write(f"  {int(row['Rank']):3d}. {row['Gene']:<15s} {row['Score']:.2f}\n")
            f.write(f"\n")
    
    # Print Top-20
    print(f"\n{'='*60}")
    print(f"  Top-20 Genes")
    print(f"{'='*60}")
    top20 = gene_ranking.head(20)
    for _, row in top20.iterrows():
        print(f"  {int(row['Rank']):3d}. {row['Gene']:<15s} {row['Score']:.2f}")
    
    print(f"\n✓ Results saved to {args.output}/")
    print(f"  - full_variant_ranking.csv ({n_total:,} variants)")
    print(f"  - full_gene_ranking.csv ({len(gene_ranking):,} genes)")
    for k in args.top_k:
        print(f"  - top{k}_genes.csv")
    print(f"  - summary.txt")
    
    return {
        'df_all': df_all,
        'gene_ranking': gene_ranking,
        'model': model,
        'seed_genes': seed_genes,
    }


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Override args with config values
        if 'data' in config:
            args.annovar = args.annovar or config['data'].get('annovar_file')
            args.string = args.string or config['data'].get('string_file')
            args.aliases = args.aliases or config['data'].get('alias_file')
        if 'seed_genes' in config:
            args.seeds = ','.join(config['seed_genes'])
        if 'training' in config:
            args.epochs = config['training'].get('epochs', args.epochs)
            args.lr = config['training'].get('learning_rate', args.lr)
            args.seed = config['training'].get('random_seed', args.seed)
        if 'output' in config:
            args.output = config['output'].get('directory', args.output)
    
    # Validate inputs
    if not args.annovar:
        print("Error: --annovar is required. Use --help for usage.")
        sys.exit(1)
    if not args.string:
        print("Error: --string is required. Use --help for usage.")
        sys.exit(1)
    
    run(args)


if __name__ == '__main__':
    main()
