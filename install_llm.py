import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.1-8B"
print("Downloading and saving model...")
save_directory = r"..." # save path, remove r if linux os
# Load from Hub
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cpu", # Save using CPU to avoid VRAM overhead during the copy
    trust_remote_code=True 
)

# Save to your local directory
tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)

print(f"Model saved to {save_directory}")