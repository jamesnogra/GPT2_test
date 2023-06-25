import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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
model = GPT2LMHeadModel.from_pretrained('gpt2')

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

# Sliding window parameters
window_size = 1024  # Adjust the window size as per your requirements
stride = 1024  # Adjust the stride as per your requirements

# Split the training data into overlapping segments
segments = []
start = 0
while start < len(training_data):
    end = min(start + window_size, len(training_data))
    segments.append(training_data[start:end])
    start += stride

# Fine-tune the model on the segments
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
total_segments = len(segments)

# Fine-tune the model on the segments
for i, segment in enumerate(segments, start=1):
    inputs = tokenizer.encode(segment, return_tensors='pt')
    inputs = inputs.to(device)
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    model.zero_grad()
    if i % 10 == 0:
        # Print progress and loss during training
        print(f"Segment {i}/{total_segments}")
        print(f"Loss: {loss.item()}")
        # Example usage: Generate text based on a prompt
        prompt = "Si jehova kay"
        generated_text = generate_text(model, tokenizer, prompt)
        print(generated_text)

# Save the fine-tuned model
model.save_pretrained('fine-tuned-model')
tokenizer.save_pretrained('fine-tuned-model')