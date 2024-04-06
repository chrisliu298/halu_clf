from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_truthfulqa_question_answers():
    truthful_qa = load_dataset("truthful_qa", "generation")["validation"]
    all_qa_pairs = []
    for x in tqdm(truthful_qa, desc="Processing truthful_qa"):
        all_qa_pairs.extend(
            [
                {"question": q, "answer": a, "label": l}
                for q, a, l in product([x["question"]], x["correct_answers"], [1])
            ]
        )
        all_qa_pairs.extend(
            [
                {"question": q, "answer": a, "label": l}
                for q, a, l in product([x["question"]], x["incorrect_answers"], [0])
            ]
        )
    print(f"Number of QA pairs: {len(all_qa_pairs)}")
    questions = [x["question"] for x in all_qa_pairs]
    answers = [" " + x["answer"] for x in all_qa_pairs]
    labels = [x["label"] for x in all_qa_pairs]
    question_answers = [q + a for q, a in zip(questions, answers)]
    return question_answers, labels


def make_mmlu_question_answers():
    mmlu = load_dataset("cais/mmlu", "all")["validation"]

    def make_labels(answer):
        return [1 if i == answer else 0 for i in range(4)]

    question_answers = []
    labels = []
    for sample in mmlu:
        question_answers.extend(
            [sample["question"] + " " + a for a in sample["choices"]]
        )
        labels.extend(make_labels(sample["answer"]))
    return question_answers, labels


question_answers, labels = make_truthfulqa_question_answers()
dataset = Dataset.from_dict({"text": question_answers, "label": labels})
dataset = dataset.class_encode_column("label")
dataset = dataset.train_test_split(test_size=0.5, seed=42, stratify_by_column="label")
train_dataset = dataset["train"]
test_dataset = dataset["test"]


def f1_score(references, predictions):
    true_positives = np.sum((predictions == 1) & (references == 1))
    false_positives = np.sum((predictions == 1) & (references == 0))
    false_negatives = np.sum((predictions == 0) & (references == 1))
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )
    f1_score = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return f1_score


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = F.cross_entropy(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits, labels = torch.tensor(logits).to(device), torch.tensor(labels).to(device)
    loss = F.cross_entropy(logits, labels)
    pred = torch.argmax(logits, dim=1).cpu().numpy()
    acc = np.mean(pred == labels.cpu().numpy())
    f1 = f1_score(labels.cpu().numpy(), pred)
    return {"acc": acc, "loss": loss.item(), "f1": f1}


print(f"Number of train examples: {len(dataset['train'])}")
print(f"Number of test examples: {len(dataset['test'])}")

num_class_0 = dataset["train"]["label"].count(0)
num_class_1 = dataset["train"]["label"].count(1)
class_weights = [
    (num_class_0 + num_class_1) / num_class_0,
    (num_class_0 + num_class_1) / num_class_1,
]  # Example of inverse frequency calculation
class_weights = torch.tensor(class_weights).float().to("cuda")
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
print(f"Class weights: {class_weights}")

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large")
# Here we truncate from the left because we want to preserve as many answer tokens as possible.
tokenizer.truncation_side = "left"
model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-v3-large")
model.config.pooler_dropout = 0.1

# Tokenize the dataset
# We don't do padding here because we use dynamic padding in the data collator.
tokenized_dataset = dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, max_length=512),
    batched=True,
)
train_max_len = max([len(x) for x in tokenized_dataset["train"]["input_ids"]])
test_max_len = max([len(x) for x in tokenized_dataset["test"]["input_ids"]])
print(f"Maximum train length: {train_max_len}")
print(f"Maximum test length: {test_max_len}")
# We use this data collator to pad the input_ids and attention_mask
# dynamically to the longest sequence in the batch (as opposed to
# padding to the longest sequence in the dataset, which might not be
# efficient).
data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer, padding="longest", return_tensors="pt"
)

# Hyperaprameters are based on the following papers:
# [On the Stability of Fine-tuning BERT: Misconceptions, Explanations, and Strong Baselines](https://arxiv.org/abs/2006.04884)
# [DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing](https://arxiv.org/abs/2111.09543)
num_train_epochs = 20
training_args = TrainingArguments(
    overwrite_output_dir=True,
    output_dir="classifier",
    learning_rate=1e-5,
    adam_epsilon=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.999,
    weight_decay=0.01,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_train_epochs,
    logging_strategy="steps",
    logging_steps=0.1,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="test_acc",
    greater_is_better=True,
    report_to="none",
)

# Define the trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset={
        "train": tokenized_dataset["train"],
        "test": tokenized_dataset["test"],
    },
    compute_metrics=compute_metrics,
    data_collator=data_collator,
)
trainer.train()
test_metrics = trainer.evaluate(eval_dataset=tokenized_dataset["test"])
print(f"Test metrics: {test_metrics}")

# Save the model
trainer.save_model("classifier")
