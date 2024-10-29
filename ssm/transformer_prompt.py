from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained('./trained_transformer')
tokenizer = GPT2Tokenizer.from_pretrained('./trained_transformer')

# Function to generate a response
def generate_response(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=0.95, temperature=1.0)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example prompt to start a conversation
prompt = "What is the meaning of life?"

response = generate_response(prompt, model, tokenizer)
print("Model:", response)

# Continue the conversation
while True:
    prompt = input("You: ")
    response = generate_response(prompt, model, tokenizer)
    print("Model:", response)
