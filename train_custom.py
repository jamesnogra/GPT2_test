import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config

#ensures reproducibility of the results when using random operations or when working with models that involve randomness, such as neural networks
torch.manual_seed(42)

def generate_text(model, tokenizer, prompt_text, max_length=32, temperature=0.7):
    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')
    input_ids = input_ids.to(model.device)

    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=temperature,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Set the path to your training data directory
training_data_dir = 'texts'

# Initialize the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define your own GPT2LMHeadModel configuration
model_config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,  # Adjust the maximum position length if necessary (gpt2 has 1024)
    n_embd=256,  # Adjust the embedding dimension (gpt2 has 768)
    n_layer=8,  # Adjust the number of layers (gpt2 has 12)
    n_head=8,  # Adjust the number of attention heads (gpt2 has 12)
    intermediate_size=2048,  # Adjust the intermediate size (gpt2 has 3072)
)

# Create your own GPT2LMHeadModel with the custom configuration
model = GPT2LMHeadModel(config=model_config)

# Set the device to GPU if available, otherwise use CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using \033[92m{device}\033[0m for training...')
model.to(device)

# Load the training data from files in the directory
training_data = ''
for filename in os.listdir(training_data_dir):
    file_path = os.path.join(training_data_dir, filename)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        training_data += file.read()

# Fine-tune the model on the training data
window_size = 1024
stride = 256
max_length = 32

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
total_iterations = len(training_data) // stride

for i in range(0, len(training_data) - window_size, stride):
    segment = training_data[i:i + window_size]
    inputs = tokenizer.encode(segment, return_tensors='pt', truncation=True, max_length=max_length)
    inputs = inputs.to(device)
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    model.zero_grad()
    if (i // stride) % 100 == 0:
        # Print progress and loss during training
        print(f"Iteration {i // stride}/{total_iterations}")
        print(f"Loss: {loss.item()}")
        # Example usage: Generate text based on a prompt
        prompt = 'nahitabo sa human niining mga butanga, nga ang magbalantay'
        generated_text = generate_text(model, tokenizer, prompt, max_length=max_length)
        print(generated_text)

# Save the fine-tuned model
model.save_pretrained('custom-fine-tuned-model')
tokenizer.save_pretrained('custom-fine-tuned-model')