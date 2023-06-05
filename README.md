# test
# Prepare training data
training_data_lines = []
for text, label in data_list:
    line = f"{text} __label__{label}\n"
    training_data_lines.append(line)

# Write training data to a file
train_data_file = "train.txt"  # Path to the training data file
with open(train_data_file, "w", encoding="utf-8") as file:
    file.writelines(training_data_lines)

# Training the model
model = fasttext.train_supervised(input=train_data_file, lr=0.1, epoch=25, wordNgrams=2)

# Testing the model
test_data = "test.txt"  # Path to the testing data file
result = model.test(test_data)
print("Precision:", result[1])
print("Recall:", result[2])
