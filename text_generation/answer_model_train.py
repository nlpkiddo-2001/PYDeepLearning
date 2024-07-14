import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# !nvidia-smi

# !pip3 install -U pandas
# !pip3 install -U scikit-learn
# !pip3 install -U numpy
# !pip3 install -U torch
# !pip3 install -U transformers
# !pip3 install -U datasets
# !pip3 install -U nltk
# !pip3 install -U accelerate
# !pip3 install -U rouge_score
# !pip3 install -U sentencepiece
# !pip3 install -U git+https://github.com/huggingface/transformers.git
# !pip3 install -U git+https://github.com/huggingface/accelerate.git
# !pip3 install torch torchvision
# !pip3 install -U tensorboard

from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup
from datasets import load_metric
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import nltk
nltk.download('punkt')
import numpy as np
from transformers import EarlyStoppingCallback
from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW


def get_dataset():
    """
    This function will return the dataset
    """
    train_dataframe = pd.read_csv('sep7_train_dataset.csv')
    train_dataframe['source_text'] = train_dataframe['source_text'].str.replace("zoho", "[COMPANY_NAME]", case=False)
    train_dataframe['target_text'] = train_dataframe['target_text'].str.replace("zoho", "[COMPANY_NAME]", case=False)
    train_dataframe = train_dataframe.drop_duplicates(subset='source_text')

    test_dataframe = pd.read_csv('sep7_test_dataset.csv')
    test_dataframe['source_text'] = test_dataframe['source_text'].str.replace("zoho", "[COMPANY_NAME]", case=False)
    test_dataframe['target_text'] = test_dataframe['target_text'].str.replace("zoho", "[COMPANY_NAME]", case=False)
    test_dataframe = test_dataframe.drop_duplicates(subset='source_text')

    return train_dataframe, test_dataframe


def get_model_tokenizer():
    """
    Function will return the model and tokenizer
    """
    model_name = "t5-base"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    print(model.to(device).device)
    return model, tokenizer


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_length, target_length):
        assert 'source_text' in dataframe.columns, "Missing 'context' column in the dataframe."
        assert 'target_text' in dataframe.columns, "Missing 'label' column in the dataframe."

        self.tokenizer = tokenizer
        self.source_length = source_length
        self.target_length = target_length

        self.data = self._tokenize_data(dataframe)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        source_ids = self.data[index]['source_ids']
        source_mask = self.data[index]['source_mask']
        target_ids = self.data[index]['target_ids']
        target_mask = self.data[index]['target_mask']

        return {
            'input_ids': source_ids.to(dtype=torch.long),
            'attention_mask': source_mask.to(dtype=torch.long),
            'labels': target_ids.to(dtype=torch.long),
        }

    def _tokenize_data(self, dataframe):
        tokenized_data = []

        for _, row in dataframe.iterrows():
            source_text = row['source_text']
            target_text = row['target_text']

            source = self.tokenizer.batch_encode_plus(
                [source_text],
                max_length=self.source_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            target = self.tokenizer.batch_encode_plus(
                [target_text],
                max_length=self.target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            source_ids = source['input_ids'].squeeze()
            source_mask = source['attention_mask'].squeeze()
            target_ids = target['input_ids'].squeeze()
            target_mask = target['attention_mask'].squeeze()

            tokenized_data.append({
                'source_ids': source_ids,
                'source_mask': source_mask,
                'target_ids': target_ids,
                'target_mask': target_mask,
            })

        return tokenized_data


def seed_everything(seed_value):
    torch.manual_seed(seed_value)
    # np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True


def load_datasets(tokenizer, max_len, qa_len):
    train_dataset, validation_dataset = get_dataset()
    training_set = CustomDataset(train_dataset, tokenizer, max_len, qa_len)
    val_set = CustomDataset(validation_dataset, tokenizer, max_len, qa_len)
    return training_set, val_set


metric = load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip()))
                     for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip()))
                      for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels,
                            use_stemmer=True)
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id)
                       for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def get_optimizer_and_scheduler(model, num_training_steps, lr=3e-4, warmup_ratio=0.2):
    """
    This function returns the AdamW optimizer and linear learning rate scheduler.

    Args:
        model: The PyTorch model for which the optimizer and scheduler are created.
        num_training_steps: The total number of training steps.
        lr: The learning rate for the optimizer (default is 3e-4).
        warmup_ratio: The warmup ratio for the learning rate scheduler (default is 0.2).

    Returns:
        optimizer: AdamW optimizer with specified parameters.
        scheduler: Linear learning rate scheduler with warmup.
    """
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08)
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    return optimizer, scheduler


def get_training_val_set():
    """
    Function will train the Model.
    """
    training_set, val_set = load_datasets(tokenizer, 1024, 256)
    return training_set, val_set


def train_model(model, tokenizer):
    """
    Function will train model
    """
    BATCH_SIZE = 4

    SEED_VALUE = 42
    seed_everything(SEED_VALUE)
    print('SEEDING COMPLETED')

    data_collator = DataCollatorForSeq2Seq(tokenizer)
    metric = load_metric("rouge")
    print('METRIC LOADED')

    MODEL_DIR = 'flan-t5-base-answer-model-dir-sep7'
    writer = SummaryWriter(log_dir=MODEL_DIR)

    #     args = Seq2SeqTrainingArguments(
    #         output_dir=MODEL_DIR,
    #         evaluation_strategy="epoch",
    #         logging_strategy="steps",
    #         logging_steps=500,
    #         save_strategy="epoch",
    #         learning_rate=4e-5,
    #         gradient_accumulation_steps=4,
    #         per_device_train_batch_size=2,
    #         per_device_eval_batch_size=8,
    #         bf16=True,
    #         weight_decay=0.01,
    #         save_total_limit=5,
    #         num_train_epochs=7,
    #         predict_with_generate=True,
    #         load_best_model_at_end=True,
    #         metric_for_best_model="rouge1",
    #         report_to='tensorboard',
    #     )

    args = Seq2SeqTrainingArguments(
        output_dir=MODEL_DIR,
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=250,
        save_strategy="epoch",
        learning_rate=4e-5,
        gradient_accumulation_steps=4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        bf16=True,
        save_total_limit=5,
        num_train_epochs=7,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        report_to='tensorboard',
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-08,
        lr_scheduler_type="linear",
        warmup_ratio=0.2,
    )

    training_set, val_set = get_training_val_set()

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=training_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            TensorBoardCallback(tb_writer=writer)
        ]
    )
    trainer.train()
    trainer.save_model("sep7_answer_model")
    print("Model Saved")
    print("Model trained")


model, tokenizer = get_model_tokenizer()
train_model(model, tokenizer)
