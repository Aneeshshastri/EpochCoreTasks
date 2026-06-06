# Subtask 1 - Encoder-Only Transformers for Relational Reasoning

## Repository Structure

```text
Task1/Subtask1/
├── Models/                     # Contains saved model checkpoints (Transformer, MLP, LSTM)
├── Output/                     # Attention heatmaps and visualizations
├── Demo.py                     # Streamlit app for Transformer inference visualization
├── Transformer.py              # Implementation of Transformer components (RoPE, MHSA, FFN)
├── Subtask1.ipynb              # Jupyter notebook with data loading, training loops, and evaluation
├── ranking_dataset.csv         # Synthetic dataset for the Array Element Ranking task
└── README.md                   # This file
```

## Usage Instructions

### Training and Evaluation
The full training pipeline and baseline comparisons are implemented in `Subtask1.ipynb`. 
Run the notebook sequentially to:
1. Load the synthetic dataset (`ranking_dataset.csv`).
2. Train the baseline models (MLP, LSTM).
3. Train the Transformer model (`DigitRanker`).
4. Evaluate the models and generate attention heatmaps.

### Inference & Visualization Demo
To launch the interactive Streamlit demo for the Transformer model:
```bash
streamlit run Demo.py
```
The web interface allows you to input an arbitrary sequence of 10 integers. It will output the predicted ranks and display a heatmap of the multi-head attention weights, letting you observe the model's internal routing logic.

## Analysis

### Transformer vs. Baselines
The goal of this task is to predict the relative sorted rank of each integer in a sequence. This is fundamentally a global relational reasoning task.

**Performance Results:**
| Model | Test Token Accuracy | Test Exact Sequence Accuracy |
| --- | --- | --- |
| Transformer (DigitRanker) | 0.9843 | 0.8730 |
| MLP Baseline (MLPRanker) | 0.5843 | 0.0090 |
| LSTM Baseline (LSTMRanker) | 0.6800 | 0.0210 |

**Comparison:**
- **MLP Baseline:** The MLP model achieves poor accuracy. It processes inputs through fixed feed-forward layers, which prevents it from dynamically comparing varying input values against each other across different sequence permutations.
- **LSTM Baseline:** The LSTM performs slightly better by iteratively processing the sequence and updating a hidden state. However, its exact sequence accuracy is near zero. LSTMs struggle with global relational reasoning because they must compress all previous context into a single hidden vector, making precise all-to-all comparisons difficult.
- **Transformer:** The Transformer dramatically outperforms both baselines. Its bidirectional multi-head self-attention mechanism enables every token to simultaneously attend to and compare itself against every other token in the sequence. This unbottlenecked, global context view perfectly aligns with the mathematical requirements of sequence ranking.

### Multi-Headed Attention Matrices
By extracting and visualizing the attention weights from the `DigitRanker` architecture, we can see how the model implicitly learns to sort:
- **Pairwise Comparisons:** To determine its rank, an element essentially needs to count how many other elements are smaller than it. The attention heatmaps reveal that tokens learn to selectively attend to elements that satisfy these relative magnitude checks (e.g., attending strictly to larger or smaller numbers).
- **Head Specialization:** The multi-head mechanism allows different heads to specialize. For example, some heads may act as "max finders" by strongly attending to the largest numbers in the sequence, while other heads may focus on identifying the smallest values or making local magnitude comparisons. 
- **Global Context Representation:** Because attention computes direct dot-products between all token pairs, the relational distances between numbers are captured in a single step. The representation of the numerical values (via learned embeddings) allows the $Q$ and $K$ projections to generate high attention scores when the numerical relationship is relevant to the rank calculation.
