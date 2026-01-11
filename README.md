# When GNNs Meet MoE: From Structural Design to Representation-level Analysis
<img width="2814" height="1635" alt="image" src="https://github.com/user-attachments/assets/1375a5cb-e15c-454c-b54d-1cb0b213a01c" />

## Mixture-of-Experts for Graph Neural Networks

This repository provides the official implementation of our systematic study on integrating Mixture-of-Experts (MoE) into Graph Neural Networks (GNNs).
While MoE is a pivotal paradigm for scaling LLMs, its potential in the graph domain has remained largely underexplored. We extend the design space of MoE-GNNs beyond simple message-passing by integrating MoE into diverse trainable components of GCN, GIN, and GAT, including underexplored elements like GIN self-loop weights and GAT attention mechanisms.

## Installation

We used the following core packages under **Python 3.10+**.

```text
torch 2.3.1
torch-geometric 2.6.1
rdkit 2025.3.6
numpy 1.26.4
scikit-learn 1.7.2
```

## Example

For training baseline model or MoE integrated model, `train.ipynb`
