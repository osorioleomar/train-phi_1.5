import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("osorioleomar/phi-1_5-finetuned-gsm8k", trust_remote_code=True, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
inputs = tokenizer('''question: I have 1098 apples. 2 of them fell into the abyss. I ate 6. How many remaining? answer: ''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=512)
text = tokenizer.batch_decode(outputs)[0]
print(text)