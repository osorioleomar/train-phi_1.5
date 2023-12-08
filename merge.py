from peft import PeftModel
from transformers import AutoModelForCausalLM
from huggingface_hub import notebook_login
import torch
notebook_login()
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True, torch_dtype=torch.float32)
peft_model = PeftModel.from_pretrained(model, "osorioleomar/phi-1_5-finetuned-gsm8k", from_transformers=True)
model = peft_model.merge_and_unload()
print(model)
model.push_to_hub("osorioleomar/phi-1_5-finetuned-gsm8k")