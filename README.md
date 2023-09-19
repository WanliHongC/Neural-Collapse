# Neural-Collapse
Code for reproducing experiments for paper [Neural Collapse for Unconstrained Feature Model under Cross-entropy Loss with Imbalanced Data](https://arxiv.org/abs/2309.09725). For citation, please use, 

```
@article{WS23,
  title={Neural Collapse for Unconstrained Feature Model under Cross-entropy Loss with Imbalanced Data},
  author={Hong, Wanli and Ling, Shuyang},
  journal={arXiv preprint arXiv:2309.09725},
  year={2023}
}
```

Files Description:\
utils.py, train_utils.py, ana_utils.py: utility functions \
block_main.py: reproducing $\mathcal{NC_1}$ and block structure of networks \
minority_l_main.py: reproducing minority collapse for networks via changing regularization parameters \
minority_n_main.py: reproducing minority collapse for networks via changing number of samples \
limit_main.py: reproducing change of angles of networks \ 
opt_main.py: solving convex optimization from UFM \
models: directory to save weight and outputs of models \
plots: directory to save generated plots \
data: directory to save generated datasets \
opt_results: directory to save optimization results for UFM 

Testing Environments:\
Use virtual environment tools (e.g miniconda) to install packages and run experiments
python==3.8.17
pip install -r requirements.txt

Example Code for Reproduction:\
To reproduce experiment for $\mathcal{NC_1}$ and block structure (Figure 1 \& 2):\
python block_main.py --dataset_t cifar10 --structure 1 --network ResNet18 

To reproduce experiment for minority collapse against regularization parameters (Figure 5 \& 6):\
python minority_l_main.py --ka 5 --reg_z_l 0.0027 0.0030 0.0033 0.0039 0.0045 0.0051 0.0057 0.0063 0.0069 0.0075 \
python minority_l_main.py --ka 3 --reg_z_l 0.0037 0.0041 0.0045 0.0053 0.0062 0.0070 0.0078 0.0086 0.0094 0.0102 


To reproduce experiment for minority collapse against number of samples (Figure 7 \& 8):\
python minority_n_main.py --ka 5 --nalist 100 200 300 400 600 800 1000 1100 1200 1400 \
python minority_n_main.py --ka 3 --nalist 100 400 450 500 700 1000 1600 2200 2500 2800 

To reproduce experiment for angles plot against number of samples(Figure 9):\
limit_main.py --nalist 500 1000 1500 2000 2500 3000 3500 4000 4500 5000 5500\

To solve optimization problem from UFM:\
limit_main.py --cluster_s 3 3 4 --5000 4000 3000 --reg_z 0.005 --reg_b 0.01






