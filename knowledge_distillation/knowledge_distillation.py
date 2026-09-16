import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

class TextDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_length=512):
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        prompt = "respond with phrases, contains key element: "
        source_text = prompt + self.data[idx]['source_text']
        target_text = self.data[idx]['target_text']
        
        inputs = self.tokenizer(source_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        labels = self.tokenizer(target_text, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        
        labels['input_ids'][labels['input_ids'] == self.tokenizer.pad_token_id] = -100
        
        input_dict = {key: val.squeeze(0) for key, val in inputs.items()}
        input_dict['labels'] = labels['input_ids'].squeeze(0)
        
        return input_dict


def distillation_loss(student_logits, teacher_logits, labels):
    student_loss = torch.nn.CrossEntropyLoss()(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
    student_probs = torch.nn.functional.softmax(student_logits, dim=-1)
    kd_loss = torch.nn.KLDivLoss(reduction="batchmean")(torch.log(student_probs), teacher_probs)
    return student_loss + 0.5 * kd_loss # using 0.5 as alpha value is recommended (But you can experiment)


tokenizer = T5Tokenizer.from_pretrained('t5-large-keyphrase-generator-may-19-checkpoints/checkpoint-2904/')
teacher_model = T5ForConditionalGeneration.from_pretrained('t5-large-keyphrase-generator-may-19-checkpoints/checkpoint-2904/').eval() 
student_model = T5ForConditionalGeneration.from_pretrained('t5-small')


dataset = TextDataset('gpt4_24k_data_18may.json', tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

num_epochs = 3
optimizer = AdamW(student_model.parameters(), lr=6e-5, weight_decay=0.01)
num_training_steps = len(dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
teacher_model.to(device)
teacher_model.eval()
student_model.to(device)

for epoch in range(num_epochs):
    student_model.train()
    for batch in tqdm(dataloader, desc='!Training!'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        with torch.no_grad():
            teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        student_output = student_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        
        loss = distillation_loss(student_output.logits, teacher_output.logits, labels)
        
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

print("Training completed.")
