# METHOD: Deep generative model of genetic variation data improves imputation accuracy in private populations

Official repository for artificial genome generation and imputation using METHOD.

This repository is still being modified. In general, the Impute5 pipeline here should not be used. It is made specifically for the single SNP imputation experiments and does not work well on clusters since a new VCF file needs to be created each time a SNP is dropped.

All code in the `plots/structure/` directory is adopted and modified from this paper: [Deep convolutional and conditional neural networks for large-scale genomic data generation](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011584#sec002).

---

## Installation

### Install PyJuice from source

```bash
git clone https://github.com/Tractables/pyjuice.git
cd pyjuice
pip install -e .
```
### Clone this repository
```bash
git clone https://github.com/prateekanand2/genetic_pc.git
```

### Train a small model on the 805 SNP data and generate samples
```bash
cd genetic_pc/pc
python3 train.py
python3 generate.py
```

### Visualize PCA
```bash
pc/demo/plot.ipynb
```
