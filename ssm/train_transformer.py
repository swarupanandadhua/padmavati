import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Load the GPT-2 tokenizer and model (untrained version)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# If using an untrained model, reset the weights
model.init_weights()

# Load the text file and create a dataset
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size
    )
    return dataset

# Data collator for masking tokens (to fill gaps)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # We are doing casual language modeling (not masked)
)

# Load your book
train_dataset = load_dataset('book.txt', tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',            # Directory to save model outputs
    overwrite_output_dir=True,         # Overwrite existing outputs
    num_train_epochs=3,                # Train for 3 epochs
    per_device_train_batch_size=2,     # Batch size
    save_steps=10_000,                 # Save every 10k steps
    save_total_limit=2,                # Limit to 2 saved models
    prediction_loss_only=True,         # Only calculate loss
)

# Trainer for training the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the trained model
trainer.save_model('./trained_transformer')
tokenizer.save_pretrained('./trained_transformer')
