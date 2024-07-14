"""
Module for training NER model
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from datasets import load_from_disk

from transformers import PreTrainedModel, TrainingArguments, Trainer
import numpy as np
from torch import nn
import torch
from torchcrf import CRF
from transformers import AutoTokenizer
from transformers import AutoModelForTokenClassification, AutoConfig
from transformers import DataCollatorForTokenClassification
import json

model_checkpoint = "google/electra-small-discriminator"  # model_checkpoint = "microsoft/deberta-v3-small"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

dataset = load_from_disk('ner_dataset_24_feb/')
print(dataset)

dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
test_dataset = dataset['test']

label_all_tokens = True


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, max_length=512, padding=True,
                                 is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


tokenized_train_datasets = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_test_datasets = test_dataset.map(tokenize_and_align_labels, batched=True)

# model_checkpoint = "microsoft/deberta-v3-small"
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=7)
model.to("cuda")


class BertCRF(PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)

        self.bert = AutoModelForTokenClassification.from_config(config)

        self.num_labels = config.num_labels
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.bilstm = nn.LSTM(config.hidden_size, (config.hidden_size) // 2, dropout=0.05, batch_first=True,
                              bidirectional=True)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss. Indices should be in ``[0, ..., config.num_labels -
            1]``.
        """
        # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self = self.to(device)

        # input_ids = input_ids.to(device)
        # attention_mask = attention_mask.to(device)
        # token_type_ids = token_type_ids.to(device)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                            position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states, return_dict=return_dict, )

        sequence_output = outputs.hidden_states[-1]
        sequence_output = self.dropout(sequence_output)
        lstm_output, hc = self.bilstm(sequence_output)
        logits = self.classifier(lstm_output)

        loss = None
        if labels is not None:
            mask = torch.not_equal(labels, torch.tensor(-100))
            # mask = torch.not_equal(labels, torch.tensor(-100).to(labels.device))  # Adjust device here
            new_logits = logits[mask]
            new_logits = new_logits.unsqueeze(0)
            new_labels = labels[mask]
            new_labels = new_labels.unsqueeze(0)
            log_likelihood, tags = self.crf(new_logits, new_labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, tags


label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
id2label = {str(i): label for i, label in enumerate(label_list)}
label2id = {label: str(i) for i, label in enumerate(label_list)}

num_labels = 7
config = AutoConfig.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
    num_labels=num_labels,
    ignore_mismatched_sizes=True,
    output_hidden_states=True
)

crfNer = BertCRF(config)
crfNer.bert.load_state_dict(model.state_dict())


def calculate_total_no_steps(num_epochs, train_dataset_len, batch_size):
    total_samples = train_dataset_len
    steps_per_epoch = total_samples // batch_size  
    total_steps = steps_per_epoch * num_epochs  
    return total_steps


args = TrainingArguments("test-ner-2",
                         learning_rate=4e-5,
                         per_device_train_batch_size=32,
                         per_device_eval_batch_size=32,
                         num_train_epochs=7,
                         save_total_limit=7,
                         weight_decay=0.01,
                         report_to="wandb",
                         logging_strategy="steps",
                         logging_steps=500,
                         evaluation_strategy="steps",
                         eval_steps=10000,
                         save_steps=10000,
                         save_strategy="steps",
                         load_best_model_at_end=True,
                         fp16=False, bf16=False, )

data_collator = DataCollatorForTokenClassification(tokenizer)

from datasets import load_metric

metric = load_metric("seqeval")

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']

"""
Be careful while modifying the below code. because 👇
1. Transformers model produces outputs in 3 dimesnsions (batch_size * seq_len * num_labels) so we need to argmax the axis=2.
2. CRF will produce outputs in 2 dimensions (batch size * seq_len) which is already correct so we just need to compute the true predictions.
"""


def compute_metrics(p):
    predictions, labels = p
    # Check if predictions are already in label index form (CRF output)
    if predictions.ndim == 2:
        # The output is already in the correct form, no need for argmax
        true_predictions = [[label_list[int(p)] for (p, l) in zip(prediction, label) if l != -100] for prediction, label
                            in zip(predictions, labels)]
    else:
        # Use argmax for models that output probabilities (non-CRF)
        predictions = np.argmax(predictions, axis=2)
        true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                            zip(predictions, labels)]

    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"], }


model.to("cuda")
trainer = Trainer(model=model,
                  args=args,
                  train_dataset=tokenized_train_datasets,
                  eval_dataset=tokenized_test_datasets,
                  data_collator=data_collator,
                  tokenizer=tokenizer,
                  compute_metrics=compute_metrics)

trainer.train()

model_path = 'electra_small_ner_finetuned_v1_24_feb'
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

id2label = {str(i): label for i, label in enumerate(label_list)}
label2id = {label: str(i) for i, label in enumerate(label_list)}

config = json.load(open("electra_small_ner_finetuned_v1_24_feb/config.json"))
config["id2label"] = id2label
config["label2id"] = label2id
json.dump(config, open("electra_small_ner_finetuned_v1_24_feb/config.json", "w"))
