from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, TrainingArguments, Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, load_metric
import numpy as np


# Define metric
metric = load_metric("seqeval")

# Initialize Tokenizer and Model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForTokenClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# Define the Label Map
label_map = {"First name": 0, "Last name": 1}

# Define Your Training, Validation, and Test Data Here
# Define Your Training, Validation, and Test Data Here
training_examples = [
    ("Chris Stansbury", [("Chris", "First name"), ("Stansbury", "Last name")]),
    ("America Ferrera", [("America", "First name"), ("Ferrera", "Last name")]),
    ("Jean-Louis Dumas", [("Jean-Louis", "First name"), ("Dumas", "Last name")]),
    ("Serena Van der Woodsen", [("Serena", "First name"), ("Van der Woodsen", "Last name")]),
    ("Elon Musk", [("Elon", "First name"), ("Musk", "Last name")]),
    ("Marie Curie", [("Marie", "First name"), ("Curie", "Last name")]),
    ("Martin Luther King", [("Martin Luther", "First name"), ("King", "Last name")]),
    ("Mary-Kate Olsen", [("Mary-Kate", "First name"), ("Olsen", "Last name")]),
    ("Robert Downey Jr.", [("Robert", "First name"), ("Downey Jr.", "Last name")]),
    ("Julia Louis-Dreyfus", [("Julia", "First name"), ("Louis-Dreyfus", "Last name")]),
]

validation_examples = [
    ("Steve Jobs", [("Steve", "First name"), ("Jobs", "Last name")]),
    ("Meryl Streep", [("Meryl", "First name"), ("Streep", "Last name")]),
    ("Bill Gates", [("Bill", "First name"), ("Gates", "Last name")]),
    ("Oprah Winfrey", [("Oprah", "First name"), ("Winfrey", "Last name")]),
    ("Mark Zuckerberg", [("Mark", "First name"), ("Zuckerberg", "Last name")]),
    ("Taylor Swift", [("Taylor", "First name"), ("Swift", "Last name")]),
    ("Tom Hanks", [("Tom", "First name"), ("Hanks", "Last name")]),
    ("Beyonce Knowles", [("Beyonce", "First name"), ("Knowles", "Last name")]),
    ("Morgan Freeman", [("Morgan", "First name"), ("Freeman", "Last name")]),
    ("Angelina Jolie", [("Angelina", "First name"), ("Jolie", "Last name")]),
]

test_examples = [
    ("Johnny Depp", [("Johnny", "First name"), ("Depp", "Last name")]),
    ("Emma Watson", [("Emma", "First name"), ("Watson", "Last name")]),
    ("George Clooney", [("George", "First name"), ("Clooney", "Last name")]),
    ("Jennifer Aniston", [("Jennifer", "First name"), ("Aniston", "Last name")]),
    ("Leonardo DiCaprio", [("Leonardo", "First name"), ("DiCaprio", "Last name")]),
    ("Brad Pitt", [("Brad", "First name"), ("Pitt", "Last name")]),
    ("Scarlett Johansson", [("Scarlett", "First name"), ("Johansson", "Last name")]),
    ("Will Smith", [("Will", "First name"), ("Smith", "Last name")]),
    ("Matt Damon", [("Matt", "First name"), ("Damon", "Last name")]),
    ("Cate Blanchett", [("Cate", "First name"), ("Blanchett", "Last name")]),
]

# ... Rest of the code remains the same ...

# Tokenize Data
def tokenize_data(data):
    texts = [item[0] for item in data]
    label_lists = [[label_map[label[1]] for label in item[1]] for item in data]

    tokens = tokenizer(texts, truncation=True, padding='max_length', return_offsets_mapping=True, is_split_into_words=False)

    labels = []
    for offset_mapping, label_list in zip(tokens['offset_mapping'], label_lists):
        label_seq = []
        label_idx = 0
        for offset in offset_mapping:
            if offset[0] == offset[1]:
                label_seq.append(-100)
            else:
                if label_idx < len(label_list):
                    label_seq.append(label_list[label_idx])
                    label_idx += 1
                else:
                    label_seq.append(-100)
        labels.append(label_seq)

    tokens['labels'] = labels

    return tokens

# Tokenize train and validation datasets
train_data = tokenize_data(training_examples)
val_data = tokenize_data(validation_examples)

# Create Dataset instances
train_dataset = Dataset.from_dict(train_data)
val_dataset = Dataset.from_dict(val_data)

test_data = tokenize_data(test_examples)
test_dataset = Dataset.from_dict(test_data)

# Compute Metrics
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_map[id] for (id, label) in zip(prediction, label) if label != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_map[id] for id in label if id != -100]
        for label in labels
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "accuracy": results["overall_accuracy"],
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
    }

# Training Arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch",
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer) # Pass the tokenizer used in your script


# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset, # Pass the validation dataset here
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Start Training
trainer.train()

# Evaluate on Test Data
test_results = trainer.evaluate(test_dataset)
print(test_results)
