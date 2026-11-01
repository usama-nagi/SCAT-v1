# SCAT: Structure-Conditioned Antibody Transformer for CDR Design

Code for **SCAT**, an antigen-conditioned generator for antibody CDR sequences, accompanying the paper:

> **SCAT: Structure-Conditioned Antibody Transformer for CDR Design**
> M. Usama Bin Abad, Nicola Toschi, Elisa Bianconi, Franco Raimondi
> *2nd SIMBIOCHEM Workshop at NeurIPS 2026* (non-archival)

<!-- TODO: add paper / OpenReview / arXiv link once available -->

**TL;DR.** SCAT generates CDR-H3 sequences conditioned on antigen 3D structure. An SE(3)-invariant geometry encoder, a repertoire-pretrained sequence encoder, and per-residue antigen cross-attention in a causal decoder are trained with a three-phase curriculum. We test whether the model actually uses the antigen with controlled interventions (antigen swap, geometry degradation, no antigen), and screen generated candidates *post hoc* with Boltz-2 and FoldX. These tools are never used inside generation.

---

## Overview

SCAT is a causal masked-infilling model for antibody CDR design (~12M parameters).

- **Geometry encoder.** SE(3)-invariant message passing (e3nn) over a C-alpha k-NN graph (k ≤ 12, 10 Å cutoff), with per-residue chemistry features. Node features are scalar-only, so the encoder is invariant but has no access to relative orientation. The paper states this explicitly.
- **Sequence encoder.** A 4-layer bidirectional Transformer, trained jointly on antibody repertoires (OAS).
- **Conditioning.** Antigen geometry, antibody geometry (optional) and pooled sequence embedding are fused into a conditioning vector **c**, which is prepended to the decoder as a memory token.
- **Decoder.** A 6-layer causal Transformer with per-residue antigen cross-attention after layers 2 and 4 and once after the full stack, controlled by a bounded learned gate.
- **Training.** Six loss terms (AR, MLM, geometry, interface, NLL ranking, antigen-embedding InfoNCE) across a three-phase curriculum on OAS (~350M sequences) and SAbDab (15,511 complexes).

Evaluation is on **CHIMERA-BENCH** (292 held-out antibody-antigen complexes) and on property-based generation for HER2 (trastuzumab scaffold), TROP2 (sacituzumab) and PD-L1 (atezolizumab).

## Main results

All numbers below are from the paper (checkpoint at step 950,000). Please read the caveats at the end of this section before comparing against other methods.

**CHIMERA-BENCH, CDR-H3 recovery (292 complexes)**

| Method | AAR-H3 | CAAR | Unique % | Enrich-r |
|---|---|---|---|---|
| RAAD | 0.370 | 0.210 | 76.4 | 0.68 |
| MEAN | 0.370 | 0.240 | 66.8 | 0.46 |
| DiffAb | 0.230 | 0.140 | 100.0 | -0.29 |
| RefineGNN | 0.210 | 0.100 | 99.7 | 0.30 |
| AGForce | 0.400 | 0.210 | 95.5 | 0.78 |
| **SCAT** | **0.449** | **0.245** | **100.0** | **0.85** |

Baseline numbers are taken from the shared evaluation reported by AGForce. SCAT was evaluated independently.

**Antigen dependence.** The real-vs-shuffled antigen gap in AAR is positive in all 16 decoding configurations tested (+0.042 to +0.068). Teacher-forced CDR-H3 perplexity is 12.31 with the correct antigen, versus 17.57 (shuffled) and 14.10 (random). Antigen swap and geometry degradation change generation at a small, reproducible set of CDR-H3 positions, and removing the antigen entirely produces the largest drift.

**Cross-attention depth** (separately trained variants, one run each)

| Injection layers | AAR | CAAR | Ag-gap |
|---|---|---|---|
| {2} | 0.379 | 0.171 | 0.003 |
| {2, 4} | 0.414 | 0.209 | 0.022 |
| {2, 4, post-stack} (reported model) | 0.449 | 0.245 | 0.047 |

**Property-based generation with post hoc screening** (10,000 generated sequences per target)

| Target (scaffold) | 3-property success | 4-property success (incl. FoldX interaction energy) |
|---|---|---|
| HER2 (trastuzumab) | 0.447 | 0.250 |
| TROP2 (sacituzumab) | 0.232 | 0.101 |
| PD-L1 (atezolizumab) | 0.044 | 0.010 |

The three properties are FvNetCharge, FvCSP and HISum. The four-property rate adds a Boltz-2 + FoldX interaction-energy criterion measured against the reference antibody, on subsamples of about 100 candidates per target. PD-L1 is low largely because the fixed atezolizumab framework supplies almost none of the net charge the FvCSP threshold requires. SCAT is never trained toward these thresholds.

**Caveats (also stated in the paper).**
- The baseline comparison shares a benchmark, not a protocol. Inputs, training data and task formulation differ. SCAT is sequence-only and trained on a much larger corpus, while the baselines co-design structure and sequence on SAbDab-scale data.
- AAR and CAAR measure recovery of the native CDR-H3. They do not establish binding, affinity or specificity.
- No generated candidate has been tested experimentally. Boltz-2 and FoldX scores are computational prioritization proxies.
- "Held out" means passing the antigen-cluster and antibody-sequence filters described in the appendix. It does not rule out all remote homologues.

## Repository layout

<!-- TODO: adjust paths and descriptions to match the actual repository tree -->

```
.
├── scat_model.py              # SCAT architecture (encoders, conditioning, decoder, heads)
├── training_losses.py         # AR, MLM, geometry, interface, rank, InfoNCE losses
├── generate_candidates.py     # CDR-H3 candidate generation
├── evaluation_new.py          # CHIMERA-BENCH evaluation
├── sample_evaluation_new.py   # Sampling-based evaluation (antigen gap, decoding sweep)
├── configs/                   # Training / model configs
├── data/                      # Dataset loading and collation
└── plots/                     # Figures used in the paper
```

## Installation

<!-- TODO: confirm versions and add a requirements.txt / environment.yml -->

```bash
git clone https://github.com/usama-nagi/SCAT-v1.git
cd uSCAT-v1
conda create -n scat python=3.10
conda activate scat
pip install -r requirements.txt
```

Core dependencies: PyTorch, e3nn, BioPython, NumPy. For post hoc screening you also need [Boltz-2](https://github.com/jwohlwend/boltz) and [FoldX](https://foldxsuite.crg.eu/) (FoldX requires its own license). MMseqs2 is used for antigen clustering.

## Data

SCAT trains on two sources. Neither is redistributed here.

- **OAS** (Observed Antibody Space): paired and unpaired repertoire sequences, filtered and split by clonotype (V-gene, J-gene, exact CDR3).
- **SAbDab**: antibody-antigen complexes, split by antigen-sequence cluster (MMseqs2, 70% identity / 80% coverage).
- **CHIMERA-BENCH** and the three target complexes (HER2 1N8Z, TROP2 9PI9, PD-L1 5X8L) are excluded from training on two axes: antigen-cluster membership and antibody CDR-H3 similarity (≥ 80% positional match). Appendix of the paper gives the full procedure and counts.

<!-- TODO: add download / preprocessing commands -->

## Usage

<!-- TODO: fill in with the real entry points and arguments -->

**Generate CDR-H3 candidates** for an antigen and antibody scaffold:

```bash
python generate_candidates.py --help
```

**Evaluate on CHIMERA-BENCH:**

```bash
python evaluation_new.py --help
python sample_evaluation_new.py --help
```

**Checkpoint.** The paper results use the checkpoint availaible at.
<!-- TODO: add checkpoint link (Zenodo / Hugging Face / OSF) -->

## Citation

```bibtex
@inproceedings{abad2026scat,
  title     = {{SCAT}: Structure-Conditioned Antibody Transformer for {CDR} Design},
  author    = {Bin Abad, M. Usama and Toschi, Nicola and Bianconi, Elisa and Raimondi, Franco},
  booktitle = {2nd SIMBIOCHEM Workshop at NeurIPS 2026},
  year      = {2026}
}
```

## License

<!-- TODO: choose a license (e.g. MIT / Apache-2.0) and add a LICENSE file -->

## Contact

M. Usama Bin Abad, Gran Sasso Science Institute (GSSI), L'Aquila, Italy.
