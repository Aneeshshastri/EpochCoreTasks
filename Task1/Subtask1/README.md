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
The goal of this task is to predict the relative sorted rank of each integer in a sequence.

**Performance Results:**
| Model | Test Token Accuracy | Test Exact Sequence Accuracy |
| --- | --- | --- |
| Transformer (DigitRanker) | 0.9843 | 0.8730 |
| MLP Baseline (MLPRanker) | 0.5843 | 0.0090 |
| LSTM Baseline (LSTMRanker) | 0.6800 | 0.0210 |

**Comparison:**
- **MLP Baseline:** The MLP model achieves poor accuracy. It converts the input sequence into a float vector and then passes it through dense neural network layers. It lacks the built-in pairwise comparison mechanism of the Transformer.
- **LSTM Baseline:** The LSTM performs slightly better than the MLP but still has very low accuracy. LSTMs struggle with global relational reasoning because they must compress all previous context into a single hidden vector, making precise all-to-all comparisons difficult.
- **Transformer:** The Transformer dramatically outperforms both baselines. Its bidirectional multi-head self-attention mechanism enables every token to simultaneously attend to and compare itself against every other token in the sequence. This unbottlenecked, global context view perfectly aligns with the mathematical requirements of sequence ranking.

### Multi-Headed Attention Matrices
By extracting and visualizing the attention weights from the `DigitRanker` architecture, we can see how the model implicitly learns to sort:
- **Pairwise Comparisons:** To determine its rank, an element essentially needs to count how many other elements are smaller than it. The attention heatmaps reveal that tokens learn to selectively attend to elements that satisfy these relative magnitude checks (e.g., attending strictly to larger or smaller numbers). 

- **Head Specialization:** The multi-head mechanism allows different heads to specialize. As per the visualisation, you can see that some heads act as min-max finders strongly attending to either the largest or the smallest element in the sequence, while other heads may focus on finding the second largest or second smallest elements and so on. (Although visualisation suggests head specialisation isn't absolute or to this degree)


- **Global Context Representation:** Because attention computes direct dot-products between all token pairs, the relational distances between numbers are captured in a single step. The representation of the numerical values (via learned embeddings) allows the $Q$ and $K$ projections to generate high attention scores when the numerical relationship is relevant to the rank calculation.


### Experimentation Results

- **Postional Embeddings**:  Experimentation showed that Positional Embedding for this tasks were unnecessary, and learned absolute positional encodings actually resulted in worse accuracy. RoPE didn't yield significantly better results, but a slight increase in accuracy was observed.

- **Optimal Parameters**: The optimal parameters were found using a short fine-tuning study done using optuna. The parameters were  
```
{"num_layers": 2, "num_heads": 16, "d_k": 4, "learning_rate": 0.0036534068309100557}
```

- **Optimizer Details**: An AdamW optimizer with weight decay 1e-4 and a cosine decay rate schedule was used for training. The model was trained for 200 epochs with a patience of 10 (for EarlyStopping).

