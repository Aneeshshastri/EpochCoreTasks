import streamlit as st
import os
import jax
from flax import nnx
import orbax.checkpoint as ocp
import bpe_tokenizer
from models import picoGPT
from Inference import generate_poem
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH=os.path.join(BASE_DIR,"Saved_Models/Tokenizer.bin")
MODEL_PATH=os.path.join(BASE_DIR,"Saved_Models/picoGPT")



#========================1) Load Everything============================================
@st.cache_resource
def load_pipeline():
    """
    Loads the tokenizer and compiles the model into VRAM exactly once.
    Streamlit will skip this function on subsequent UI interactions.
    """
    
    #load tokenizer==================================
    tokenizer = bpe_tokenizer.BPETokenizer()
    tokenizer.load(TOKENIZER_PATH)
    
    #create model instance============================
    d_k = 32
    num_heads = 4
    num_layers = 4
    max_seq_len = 256
    TOTAL_VOCAB_SIZE = tokenizer.get_vocab_size() + 2
    rngs = nnx.Rngs(params=0, dropout=1)
    model = picoGPT(
        vocab_size=TOTAL_VOCAB_SIZE,
        d_model=d_k * num_heads,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        rngs=rngs,
        dropout_rate=0.3
    )

    #load model======================================
    checkpointer = ocp.StandardCheckpointer()
    template_state = nnx.state(model)
    
    restored_state = checkpointer.restore(MODEL_PATH, template_state)
    nnx.update(model, restored_state)
    model.eval()

    return model, tokenizer


model, tokenizer = load_pipeline()

st.title("picoGPT: Autoregressive Poetry Generation")

#=======================2) Decoding Parameters============================
st.sidebar.header("Decoding Parameters")
temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.5, value=0.85, step=0.05)
rep_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.2, step=0.05)
top_k = st.sidebar.slider("Top-K", min_value=1, max_value=100, value=40, step=1)
max_new_tokens = st.sidebar.slider("Max Sequence Length", min_value=50, max_value=1000, value=500, step=50)

#=======================3) Metadata Input Interface============================
st.subheader("Conditioning Metadata")
col1, col2 = st.columns(2)

with col1:
    author = st.text_input("Author", value="WILLIAM SHAKESPEARE")
    era = st.text_input("Era", value="Renaissance")

with col2:
    poem_type = st.text_input("Type", value="Mythology & Folklore")
    title = st.text_input("Title", value="The ancient songs")

#=======================4. Content Input Interface==========================
st.subheader("Starting Content (Optional)")
starting_content = st.text_area("Provide the opening lines...", value="")

#=======================5. Generation Trigger==============================
if st.button("Generate Poem"):
    metadata = {
        "author": author,
        "era": era,
        "type": poem_type,
        "title": title
    }
    
    with st.spinner("picoGPT is GPTing this poem"):
        output = generate_poem(
            model=model,
            tokenizer=tokenizer,
            metadata=metadata,
            content=starting_content,
            max_len=256,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            rep_penalty=rep_penalty,
            seed=42
        )
        
    st.subheader("Generated Output")
    # Reconstruct the full sequence for the UI
    full_sequence = starting_content + output
    st.text_area("Sequence", value=full_sequence, height=400)