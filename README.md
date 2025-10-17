# SchemaToken

This is the pytorch implementation of our manuscript:

> SchemaToken: Tokenizing Database Schemas for Efficient and Robust Text-to-SQL Linking

## Overview
Schema linking is a crucial step in Text-to-SQL systems, connecting natural language mentions to relevant database tables and columns. Despite recent advances in LLM-based Text-to-SQL models, current approaches face two fundamental limitations: (1) providing detailed column descriptions inflates the input context, often exceeding the model’s effective sequence length, and (2) predicting multi-token column names increases output complexity, making the task fragile to generation errors. To address these challenges, we propose SchemaToken, which represents each database column as a dedicated token in the LLM’s vocabulary. This design compresses column semantics into single embeddings, reducing context length and simplifying prediction. We introduce a two-stage training paradigm: (i) Semantic Alignment, which enables the LLM to understand the newly added schema tokens via structured embeddings, and (ii) Pseudo-task Alignment, which trains the LLM to generate these tokens as an unordered set using reinforcement learning with a Jaccard-based reward. Experiments on multiple benchmark datasets demonstrate that SchemaToken significantly improves schema linking performance, particularly for large and complex schemas, while maintaining computational efficiency. 

<p align="center">
  <img src="./framework.png" width=900>
</p>

# Data Download
Download datasets from the official repository of [Spider](https://yale-lily.github.io/spider) and [BIRD](https://bird-bench.github.io/)

## Data Preparation
```bash
python preprocess.py
```

## Initialize embedding
```bash
python init_embed.py
```

## Model Training
```bash
python train.py
```

# Model Inference
```bash
python eval.py
```
