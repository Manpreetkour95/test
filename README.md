# test
from transformers import ElectraForSequenceClassification, ElectraTokenizer, TrainingArguments, Trainer
import torch

# Define a dummy dataset
train_texts = ["This is the first example.", "Another example for training.", "One more training example."]
train_labels = [1, 0, 1]  # Example labels (binary classification)

# Load tokenizer
tokenizer = ElectraTokenizer.from_pretrained("google/electra-base-discriminator")

# Tokenize the training texts
tokenized_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")

# Create a torch Dataset
class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

dataset = DummyDataset(tokenized_inputs, train_labels)

# Load the pre-trained Electra model
model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator")

# Set the number of labels
model.config.num_labels = 2  # Number of classes (binary classification)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="output_directory",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    save_steps=500,
    save_total_limit=2,
    overwrite_output_dir=True,
)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Start training
trainer.train()

