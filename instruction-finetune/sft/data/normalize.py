"""
This module is used to normalize any kind of input data into one canonical format. 
That format taken from HF -> https://huggingface.co/docs/transformers/en/chat_templating

chat = [
    {"role": "user", "content": "Can you format the answer in JSON?"},
    {"role": "assistant", "content": '{"name": "'},
]
"""

