import torch
from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Step 1: Define the model configuration (small GPT-like model)
config = GPT2Config(
    vocab_size=50257,      # Vocabulary size (customized for Sanskrit later)
    n_positions=1024,      # Number of tokens the model can see at once
    n_ctx=1024,            # Context window
    n_embd=768,            # Embedding size (smaller than GPT-3)
    n_layer=12,            # Number of transformer layers
    n_head=12,             # Number of attention heads
)

# Step 2: Initialize the model and tokenizer
model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Step 3: Tokenize your custom Sanskrit corpus
# For demonstration, we load a Sanskrit dataset
dataset = load_dataset("bclay/sanskrit_texts")  # Example, replace with your own

def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Define TrainingArguments
training_args = TrainingArguments(
    output_dir="./gpt_sanskrit",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    logging_dir='./logs',
)

# Step 5: Trainer class for handling training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],  # Assuming train split
)

# Step 6: Train the model
trainer.train()

'''
Explanation of Components:
    1. GPT2Config: Defines the architecture of the model (layers, heads, etc.). This is where you can adjust the size of the model (making it smaller than typical LLMs).
    2. GPT2LMHeadModel: The GPT-like model used for causal language modeling (predicting the next word in a sequence).
    3. Trainer: Hugging Face's convenient wrapper for model training.
'''