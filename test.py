import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Set seed for reproducibility
torch.manual_seed(42)

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained('custom-fine-tuned-model')
tokenizer = GPT2Tokenizer.from_pretrained('custom-fine-tuned-model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Test a sentence from the text file
test_sentence = "gibuhat sa Dios ang mga mananap sa yuta ingon"

# Tokenize the input sentence
input_ids = tokenizer.encode(test_sentence, add_special_tokens=True, return_tensors='pt').to(device)

# Generate a response
generated = model.generate(input_ids, max_length=64, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
decoded_output = tokenizer.decode(generated[0], skip_special_tokens=True)

print(f"Generated output: \033[92m{decoded_output}\033[0m")