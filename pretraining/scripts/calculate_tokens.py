"""
Module to calculate the number of tokens in the pretrain, midtrain and fine tune dataset
"""
import os
import json
from tokenizers import Tokenizer


class CalculateTokens:
    def __init__(self, stage: str, dataset_dir: str, tokenizer_path: str):
        """
        Constructor for calculating the tokens
        """
        self.stage = stage
        self.dataset_dir = dataset_dir
        self.tokenizer_path = tokenizer_path

        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)

    def calculate_tokens(self):
        """Calculate the number of tokens"""
        total_token_count = 0
        if self.stage == "pretrain" or self.stage == "midtrain":
            data_jsonl_files = list(os.path.join(self.dataset_dir))
            for jsonl_data in data_jsonl_files:
                datas = json.loads(jsonl_data)
                for data in datas:
                    if "text" in data:
                        text_content = data.get("text")
                        token_len = self.tokenizer.encode(text_content)
                        total_token_count += token_len

        elif self.stage == "finetune":
            finetune_data_jsonl_files = list(os.path.join(self.dataset_dir))
            for jsonl_data in finetune_data_jsonl_files:
                all_content = json.loads(jsonl_data)
                for messages in all_content:
                    contents = ""
                    for _, content in messages.items():
                        contents += content
                    
                    token_len = self.tokenizer.encode(contents)
                    total_token_count += token_len

        return total_token_count
    

if __name__ == "__main__":
    token_calculator = CalculateTokens(
        stage="midtrain",
        dataset_dir="/Users/vipin-16319/PycharmProjects/PYDeepLearning/pretraining/data/sft",
        tokenizer_path="/Users/vipin-16319/PycharmProjects/PYDeepLearning/pretraining/data/tokenizer.json"
    )
    print(token_calculator.calculate_tokens())



