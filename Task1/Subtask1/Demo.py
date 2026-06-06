import Transformer as Model
import streamlit as st
from flax import nnx
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH=os.path.join(BASE_DIR,"Models/The_Final_Transformer")
@st.cache_resource
def load_model():
    #Initializes the model architecture and loads weights via Orbax.
    best_params={'num_layers': 2, 'num_heads': 16, 'd_k': 4, 'learning_rate': 0.0036534068309100557}
    d_model = best_params['d_k']*best_params['num_heads']
    n_heads = best_params['num_heads']
    num_layers = best_params['num_layers']
    max_seq_len = 10
    num_classes = 10
    rngs  = nnx.Rngs(params=0, dropout=1)
    model = Model.DigitRanker(d_model, n_heads, num_layers, max_seq_len, num_classes,use_lape=False, rngs=rngs)
    template_state=nnx.state(model)
    checkpointer=ocp.PyTreeCheckpointer()
    restored_state = checkpointer.restore(
        MODEL_PATH, 
        args=ocp.args.PyTreeRestore(item=template_state, partial_restore=True)
    )
    nnx.update(model, restored_state)
    return model

@nnx.jit
def run_inference(model, input_tensor):
    return model(input_tensor)


# --- 2. User Interface ---
st.title("Transformer Sequence Sorter")
st.markdown("Analyzing the routing logic of a 2-Layer, 16-Head architecture.")

#load model
model = load_model()

# User Input
st.subheader("Input Sequence")
st.write("Enter exactly 10 integers, separated by spaces or commas.")
user_input = st.text_input("Sequence:", "317, 469, 685, 72, 142, 661, 287, 980, 885, 152")
if st.button("Sort & Analyze"):
    try:
        # Parse input
        raw_strings = user_input.replace(',', ' ').split()
        input_seq = [int(x) for x in raw_strings]
        
        if len(input_seq) != 10:
            st.error(f"Expected exactly 10 integers. Got {len(input_seq)}.")
            st.stop()
            
        input_tensor = jnp.array([input_seq])
        
        st.success("Input valid. Proceeding to inference...")
        
        logits = run_inference(model, input_tensor) 
        predictions = jnp.argmax(logits, axis=-1)
        
        st.subheader("Results")
        st.write(f"Input Sequence: {input_seq}")
        st.write(f"Predicted Ranks: {predictions[0].tolist()}")
        
        # Visualization 
        num_layers = len(model.transformer_blocks)
        n_heads = model.transformer_blocks[0].mha.n_heads
        labels = [str(int(val)) for val in input_seq]
        #split into 2 parts for better visualization
        for part in range(2):
            start_head = part * 8
            end_head = start_head + 8
            
            st.subheader(f"Attention Heatmaps (Heads {start_head+1}-{end_head})")
            fig, axes = plt.subplots(num_layers, 8, figsize=(48, 12))
            if num_layers == 1:
                axes = np.expand_dims(axes, axis=0)
                
            for l_idx, block in enumerate(model.transformer_blocks):
                attn_weights = block.mha.sown_attn[...]
                attn_weights = np.array(attn_weights[0])  # (n_heads, 10, 10)
                
                for h_idx in range(start_head, end_head):
                    ax = axes[l_idx, h_idx - start_head]
                    sns.heatmap(
                        attn_weights[h_idx], 
                        annot=True, 
                        fmt=".2f", 
                        cmap="viridis", 
                        xticklabels=labels, 
                        yticklabels=labels, 
                        ax=ax,
                        cbar=False,
                        annot_kws={"size": 14}
                    )
                    ax.set_title(f"Layer {l_idx+1} Head {h_idx+1}", fontsize=18)
                    ax.set_xlabel("Key", fontsize=16)
                    ax.set_ylabel("Query", fontsize=16)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    
            plt.tight_layout()
            st.pyplot(fig)
            
    except ValueError:
        st.error("Invalid input. Please ensure all values are integers.")