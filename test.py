import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set seed for reproducibility
torch.manual_seed(42)

# Encode input text
input_text = "Is there a god?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text
attention_mask = torch.ones_like(input_ids)
output = model.generate(
	input_ids,
	attention_mask=attention_mask,
	max_length=100,
	num_return_sequences=1,
	no_repeat_ngram_size=3
)

# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)