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
    """Initializes the model architecture and loads weights via Orbax."""
    # Initialize abstract model with the optimized parameters (3L, 8H, 64d)
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

@jax.jit
def run_inference(model, input_tensor):
    return model(input_tensor)


# --- 2. User Interface ---
st.title("Transformer Sequence Sorter")
st.markdown("Analyzing the routing logic of a 3-Layer, 8-Head architecture.")

# Initialize model
model = load_model()

# Input area
st.subheader("Input Sequence")
st.write("Enter exactly 10 integers, separated by spaces or commas.")
user_input = st.text_input("Sequence:", "317, 469, 685, 72, 142, 661, 287, 980, 885, 152")

# --- 3. Execution Trigger ---
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
    # Get rank (argmax of logits)
        predictions = jnp.argmax(logits, axis=-1)
        
        st.subheader("Results")
        st.write(f"Input Sequence: {input_seq}")
        st.write(f"Predicted Ranks: {predictions[0].tolist()}")
        
        # --- Visualization ---
        # Extract attention from the last layer (example)
        # Note: You may need to adjust the access path based on how nnx stores Intermediates
        attn_weights = model.transformer_blocks[-1].mha.sown_attn.value
        
        st.subheader("Attention Heatmap (Last Layer)")
        fig, ax = plt.subplots()
        sns.heatmap(attn_weights[0, 0, :, :], annot=True, ax=ax, cmap="viridis")
        st.pyplot(fig)
            
    except ValueError:
        st.error("Invalid input. Please ensure all values are integers.")