# CRank - Consistency based Ranking
### Ranking pre-trained segmentation models for zero-shot transferability 
([link to paper](https://arxiv.org/abs/2503.00450))

![Fig1](./figures/MICCAI_intro_figure.svg)


## Installation
#### Clone repository

Clone and navigate to this repository

```
git clone https://github.com/kreshuklab/model_ranking.git
cd model_ranking
```

#### Install model_ranking conda environment

```
conda env create -f environment.yaml
conda activate model_ranking
```
#### Use Weights and Biases logging
- Login/Sign up at https://wandb.ai/login
- Get your api token at https://wandb.ai/authorize (you'll be ask to provide this token on the first hylfm run, specifically on import wandb)
