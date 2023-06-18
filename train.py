import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW

torch.manual_seed(42)

# Define your dataset class
class MyDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.texts = texts
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

# Initialize the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Fine-tuning parameters
batch_size = 1
learning_rate = 1e-4
epochs = 25

# Prepare your training dataset
# Directory path containing the text files
directory_path = 'texts'
# Read text files from the directory
train_texts = []
for filename in os.listdir(directory_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            train_texts.append(text)
train_dataset = MyDataset(train_texts, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the optimizer and the loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

# Fine-tuning loop
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using \033[92m{device}\033[0m for training...')
model.to(device)
model.train()

for epoch in range(epochs):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Manually pad sequences to the same length
        max_length = input_ids.shape[1]
        input_ids = torch.nn.functional.pad(input_ids, pad=(0, max_length - input_ids.shape[1]), value=0)
        attention_mask = torch.nn.functional.pad(attention_mask, pad=(0, max_length - attention_mask.shape[1]), value=0)
        labels = torch.nn.functional.pad(labels, pad=(0, max_length - labels.shape[1]), value=-100)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

    if (epoch+1) % 5 == 0:
        # Test a sentence after each epoch
        test_sentence = 'kahayag sa hawan sa langit aron sa pagbulag sa adlaw'
        input_ids = tokenizer.encode(test_sentence, add_special_tokens=True, return_tensors='pt').to(device)
        generated = model.generate(input_ids, max_length=32, num_return_sequences=1)
        decoded_output = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Generated output: \033[92m{decoded_output}\033[0m")

# Save the fine-tuned model
model.save_pretrained('fine-tuned-model')
tokenizer.save_pretrained('fine-tuned-model')
