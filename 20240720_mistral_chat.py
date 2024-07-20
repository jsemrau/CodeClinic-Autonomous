from transformers import pipeline
import os, torch, time
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import BitsAndBytesConfig
import streamlit as st
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"]="expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:64"
os.environ['CUDA_VISIBLE_DEVICES']='4,3,0,1,2'
os.environ["PYTORCH_USE_CUDA_DSA"] = "1"

st.title("💬 Hi Jan, I am a pirate")
checkpoint="mistralai/Mistral-Nemo-Instruct-2407"
st.caption(f"🚀 An autonomous agent powered by {checkpoint}")

quantization_config = BitsAndBytesConfig(load_in_8bit=True,llm_int8_threshold=5.0,llm_int8_enable_fp32_cpu_offload=True)
model_kwargs = {
    "max_length": 64,
    "device_map": "auto",
    "offload_folder": "offload",
    "max_memory": {4: "12GB",3: "11GB", 1: "8GB", 2: "8GB", 0: "8GB"},
    "quantization_config": quantization_config,
    #"attn_implementation":"flash_attention_2",

}

system = '''
           The assistant anwers questions the way a highly informed individualwould if they were talking to an expoert
           and can let the human know this when relevant.
            
           It should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions.
           
           Matt is happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks.
           It uses markdown for coding.
            
           Matt does not mention this information about itself unless the information is directly pertinent to the human's query.
            
        '''

def initialize_agent(checkpoint):

    #mistral_models_path = Path.home().joinpath('mistral_models', 'Nemo-Instruct')
    #tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tekken.json")  # change to extracted tokenizer file

    chatbot = pipeline("text-generation",
                           model=checkpoint,
                           device=None,
                           batch_size=5,
                           temperature= 0.3,
                           max_new_tokens=768,
                           do_sample=True,
                           torch_dtype=torch.float32,
                           use_fast=True,
                           model_kwargs=model_kwargs,
                           #eos_id = tokenizer.instruct_tokenizer.tokenizer.eos_id
                       )

    return chatbot

if "messages" not in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": system}]
    
if "agent" not in st.session_state:
    st.session_state["agent"] = initialize_agent(checkpoint)


for msg in st.session_state['messages']:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    result =st.session_state["agent"](prompt)

    rDict = dict(result[0])
    msg = rDict['generated_text']

    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.chat_message("assistant").write(msg)
