import torch
# pip install flask
from flask import Flask, request, Response, render_template
# pip install -U flask-cors
from flask_cors import CORS
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize flask
app = Flask(__name__)
CORS(app)

# Load pre-trained model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained('fine-tuned-model')
tokenizer = GPT2Tokenizer.from_pretrained('fine-tuned-model')
# Set seed for reproducibility
torch.manual_seed(42)

@app.route('/input-text')
def input_text():
	return render_template('input-text.html')

@app.route('/model-api', methods=['POST'])
def model_api():
	input_text = request.form['input_text']
	# Encode input text
	input_ids = tokenizer.encode(input_text, return_tensors='pt')
	# Generate text
	attention_mask = torch.ones_like(input_ids)
	output = model.generate(
		input_ids,
		attention_mask=attention_mask,
		max_length=128,
		num_return_sequences=1,
		no_repeat_ngram_size=3
	)
	# Decode and print generated text
	generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
	return generated_text

if __name__ == '__main__':
	app.run(debug=True, port='8085', host='0.0.0.0', use_reloader=False)