# Reinforcement Learning from Human Feedback (RLHF)

### Using Proximal Policy Optimization (PPO) to Fine-Tune GPT-2 for Sentiment Alignment

This project provides a step-by-step implementation of the **Reinforcement Learning from Human Feedback (RLHF)** pipeline. While RLHF is traditionally used to make conversational agents more helpful or safe, this project demonstrates its core mechanics by aligning a pre-trained **GPT-2** model to generate sentences with a specific human preference: **positive sentiment**.

The implementation is divided into three main notebooks corresponding to the standard RLHF stages:

## 🚀 Project Pipeline

### 1. Supervised Fine-Tuning (SFT)

The first stage involves adapting the base GPT-2 model to the target domain.

**Objective**: Teach the model the basics of the task through supervised learning.

**Dataset**: Fine-tuned on the **SST-2 (Stanford Sentiment Treebank)** dataset.

**Notebook**: `1-SFT.ipynb`

### 2. Reward Model (RM) Training

In this stage, a separate model is trained to act as a proxy for human judgment.

**Objective**: Learn to score the "positivity" of a generated sentence.

**Methodology**: A reward head is added to GPT-2 to output a scalar value representing the sentiment score.

**Notebook**: `2-RM Training.ipynb`

### 3. RLHF via Proximal Policy Optimization (PPO)

The final stage uses the Reward Model to guide the fine-tuning of the SFT model through Reinforcement Learning.

**Objective**: Iteratively optimize the model policy to maximize the reward (positive sentiment).

**Key Technique**: Uses **PPO** and **KL-divergence regularization** to ensure the model remains coherent and doesn't drift too far from the original language distribution.

**Notebook**: `3-RLHF.ipynb`

<img width="815" height="489" alt="Screenshot 2026-03-11 at 14 30 12" src="https://github.com/user-attachments/assets/5058c532-6a64-4cb5-9ec1-2f7208441132" />


## 📊 Results

The methodology successfully demonstrates how a policy model can be refined against a learned reward function. The final model is heavily biased toward generating text that aligns with the defined positive sentiment preference while maintaining linguistic coherence.

## 🛠️ Requirements

* `Python 3.x`
* `PyTorch`
* `Hugging Face Transformers`
* `Datasets`
* `trl` (Transformer Reinforcement Learning library)
---
* Developed as part of the *Knowledge Processing in Intelligent Systems* practical seminar at the **University of Hamburg**.
