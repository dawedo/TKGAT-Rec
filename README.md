
# TKGAT-Rec: Temporal Knowledge-Infused Graph Attention Network for Recommendation
 
This is our implementation for the following paper:

>Dr. Dawed Omer Ahmed, Dr. Venkateswara Rao Kagita,Dr. Vikas Kumar


Corresponding Author: Dr. Venkateswara Rao Kagita (venkat.kagita@nitw.ac.in)
Dr. Dawed Omer Ahmed: do24csr1r07@student.nitw.ac.in


## Introduction
Capturing user preference patterns from multiple aspects and effectively fusing them is crucial for designing a recommender system capable of accurately
predicting a user’s future interests. However, many existing approaches fail to simultaneously capture collaborative signals from user–item interaction graphs,
the dynamic evolution of user preferences over time, and the complex semantic relationships encoded in knowledge graphs. In this paper, we propose a
novel Temporal Knowledge-Infused Graph Attention Network for Recommendation (TKGAT-Rec) that integrates Graph Neural Network based representation
learning with a time-aware knowledge graph attention mechanism. The proposed model adapts cross-attention to jointly model collaborative filtering representations and temporal sequences, along with learnable path-aware prediction
strategies to exploit multi-hop relational information. This enables the model to exchange information across multiple pattern recognizers, thereby allowing
them to complement each other effectively. We further investigate four attention mechanisms—Product, Bi-Interaction, Bi-Perceptron, and Concat—to capture
fine-grained semantic signals from knowledge graphs. Temporal and positional embeddings are incorporated to model evolving user preferences. Experiments
on benchmark datasets show that the proposed framework outperforms strong baselines, demonstrating the effectiveness of combining temporal modeling with
knowledge-aware attention to improve recommendation quality.
## Environment Requirement
The code has been tested running under Python 3.9.12. The required packages are as follows:
* python == 3.9.12
* tensorflow == 2.9.1
* numpy == 1.21.5
* scipy == 1.7.3
* sklearn == 1.0.2

## Examples to Run the code
The instruction of commands has been clearly stated in the code (see src/main.py).

* Movie
```
python main.py  --dataset movie --aggregator concat --n_epochs 10 --neighbor_sample_size 4 --dim 32 --n_iter 2 --batch_size 65536 --l2_weight 5e-6 --lr 2e-2 --layer_size [32] --adj_type plain --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0 --smoothing_steps 1 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0
```


* Music
```
python main.py --dataset music --aggregator concat --n_epochs 10 --neighbor_sample_size 8 --dim 32 --n_iter 1 --batch_size 128 --l2_weight 1e-4 --lr 0.005 --layer_size [32] --adj_type norm --alg_type lightgcn --model_type KGCN_LightGCN --node_dropout [0.1] --mess_dropout [0.1] --node_dropout_flag 1 --alpha 0.5 --smoothing_steps 8 --pretrain 0 --att h_ur_t --runs 3 --gpu_id 0
```


## About implementation

We build our model based on the implementations of Personalized knowledge-aware recommendation with collaborative and attentive graph convolutional networks (https://github.com/rasoul119560/LGKAT).

## About Datasets
The datasets available from the corresponding author


## Citation 
If you would like to use our code, please cite:
```

```
