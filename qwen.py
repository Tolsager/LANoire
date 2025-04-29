# QWEN
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True).eval()

# Specify hyperparameters for generation (No need to do this if you are using transformers>4.32.0)
# model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-Audio-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'audio': r'data\raw\1_buyers_beware\clovis_galletta\q1\clovis_galletta_lie_0.mp3'}, # Either a local path or an url
    {'text': 'Is the person lying?'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)

