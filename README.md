# DCV-TRIAGE

**D**isease-**C**ontextual **V**US **TRIAGE**: A multimodal cross-attention framework for disease-contextual prioritisation of rare variants of uncertain significance.

## Overview

DCV-TRIAGE integrates two complementary modalities to re-rank rare VUS in a disease-contextual manner:

- **Modal 1**: Variant-level pathogenicity features (19 ANNOVAR annotations → calibrated logistic classifier → 20D vector)
- **Modal 2**: Seed-anchored PPI network embeddings (STRING v12 → RWR/PPR → frozen GCN → 16D vector)
- **Fusion**: Cross-attention gating combines both modalities to produce a continuous disease-contextual ranking score

The framework is disease-agnostic in architecture but disease-specific through seed gene specification. Changing the seed set reconfigures the ranking for a different disease context without retraining the network components.

## Quick Start

### 1. Installation

```bash
git clone https://github.com/[username]/DCV-TRIAGE.git
cd DCV-TRIAGE
pip install -r requirements.txt
```

### 2. Prepare Input Data

You need three files:

| File | Description | Source |
|------|-------------|--------|
| Variant annotation CSV | ANNOVAR-annotated variants with CLNSIG labels | Your data |
| STRING PPI | Protein-protein interaction network | [STRING v12](https://string-db.org/) |
| STRING aliases | Gene name mapping | [STRING v12](https://string-db.org/) |

### 3. Run

**Option A: Command line (recommended)**

```bash
python run_pipeline.py \
  --annovar data/your_variants.csv \
  --string data/9606.protein.links.full.v12.0.txt \
  --aliases data/9606.protein.aliases.v12.0.txt \
  --seeds "CRBN,FGFR3,TP53,KRAS,TNFRSF17,NRAS,LIG4,BRAF,CXCR4,XPO1,CD38,RBX1,CUL4A,PSMB5,DDB1,BCL2,FDPS,TUBB4A" \
  --output results/my_analysis/
```

**Option B: Python API**

```python
from src.pipeline import run_full_pipeline

results = run_full_pipeline(
    annovar_file="data/your_variants.csv",
    string_file="data/9606.protein.links.full.v12.0.txt",
    alias_file="data/9606.protein.aliases.v12.0.txt",
    seed_genes=["CRBN", "FGFR3", "TP53", "KRAS", ...],
    output_dir="results/my_analysis/"
)

# Access ranked VUS
print(results['top20_genes'])
print(results['df_all'].head())
```

**Option C: Jupyter notebook**

See `notebooks/demo_MM.ipynb` for an interactive walkthrough.

### 4. Output

The pipeline produces:

```
results/my_analysis/
├── full_variant_ranking.csv     # All variants with ranking scores
├── full_gene_ranking.csv        # Gene-level max-score ranking
├── top20_report.txt             # Top-20 summary
└── figures/
    ├── score_distribution.png
    └── seed_distance.png
```

## Applying to a Different Disease

Change the seed genes to reconfigure for any disease:

```bash
# Example: Type 2 Diabetes
python run_pipeline.py \
  --annovar data/your_variants.csv \
  --string data/9606.protein.links.full.v12.0.txt \
  --aliases data/9606.protein.aliases.v12.0.txt \
  --seeds "INS,GCK,HNF1A,HNF4A,ABCC8,KCNJ11" \
  --output results/diabetes/
```

Seed genes can be derived from disease-gene databases such as [Open Targets](https://www.opentargets.org/).

## Configuration

All hyperparameters can be set via `config.yaml` or command-line arguments:

```yaml
# config.yaml
model:
  classifier_type: logistic
  gcn_hidden_dims: [64, 32]
  gcn_output_dim: 16
  attention_dim: 64
  mlp_hidden_dims: [128, 64, 32]
  dropout: 0.3
  learning_rate: 0.0001
  epochs: 100

data:
  rwr_restart_prob: 0.15
  ppr_alpha: 0.15
  exonic_only: true
  remove_super_hubs: true

output:
  top_k: [20, 50, 100]
  save_full_ranking: true
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- scikit-learn ≥ 1.0
- pandas, numpy, networkx
- umap-learn
- matplotlib (optional, for figures)

See `requirements.txt` for exact versions.

## Reproducibility

To reproduce the results from the manuscript, the input data must be obtained independently:
- **UK Biobank**: Available upon approved application (Application No. 100359)
- **ClinVar**: Freely available from [NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)
- **STRING v12**: Freely available from [STRING](https://string-db.org/)

Configuration files used for the manuscript analyses are provided in `configs/`.

## Citation

If you use DCV-TRIAGE in your research, please cite:

```
[Citation to be added upon publication]
```

## Licence

[MIT / Apache 2.0 — choose one]
