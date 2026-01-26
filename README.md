# Learnable Obfuscation on CIFAR-10

This project replicates the CIFAR-10 experiment setup from the Learnable Obfuscation paper using:
- ResNet-50 pretrained embeddings (2048-d) with L2 normalization
- Class-k mixing to create soft labels and mixed samples
- Obfuscation via random masking projection to d=500, Gaussian noise, and label permutation
- A 3-layer MLP trained on the obfuscated dataset
- Evaluation by decoding the permuted labels back to the original CIFAR-10 class space

The main script is `test.py`.

## What the code does

Pipeline overview:
1. Embed CIFAR-10 images using pretrained ResNet-50 (ImageNet) to get 2048-d feature vectors, then L2-normalize.
2. Sample a balanced private subset of size `n` (equal per class).
3. Build a mixed dataset of size `m` using (i, j)-class-k mixing:
   - each mixed sample averages k examples from class i and k examples from class j
   - labels become soft targets: one-hot if i=j, else 50/50 across i and j
4. Obfuscate the mixed dataset:
   - project with a random masking matrix W into `d=500`
   - add Gaussian noise with std `sigma`
   - permute sample order Π1
   - permute label space Π2
5. Train a 3-layer MLP on obfuscated inputs and permuted soft labels.
6. Evaluate by projecting test embeddings with the same W and inverting Π2 to recover original class predictions.

## Results (example runs)

These are example outputs from Apple Silicon MPS:
- `--n 1000 --m 4000 --k 5 --sigma 0.03 --epochs 5`  → ~79% test accuracy
- `--n 2000 --m 4000 --k 10 --sigma 0.04 --epochs 10` → ~81% test accuracy

Your exact numbers may vary slightly due to randomness and hardware.

## Requirements

- macOS recommended (Apple Silicon supported via MPS)
- Python 3.12+ recommended (PyTorch support on very new Python versions can be inconsistent)
- Packages:
  - torch
  - torchvision
  - numpy

## Setup

From the project folder:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision numpy
