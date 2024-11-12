# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

with open('/Users/vipin-16319/PycharmProjects/PYDeepLearning/input.txt','r', encoding='utf-8') as file:
    text = file.read()



characters = sorted(list(set(text)))
# print(f"characters is {characters}")
# print(len(characters))
# print(''.join(characters))
vocab_size = len(''.join(characters))


# creating a mapping from character to integer
# i.e character tokenizer

character_to_integer = {ch : i for i, ch in enumerate(characters)}
integer_to_character = {i : ch for i, ch in enumerate(characters)}

# print(character_to_integer)r
# print(integer_to_character)

encode = lambda s : [character_to_integer[ch] for ch in s]
decode = lambda l : ''.join([integer_to_character[i] for i in l])

# print(encode('hii'))
# print(decode(encode('hii')))

data = torch.tensor(encode(text), dtype=torch.long)
# print(data.shape, data.dtype)
# print(data[:1000])

n = int(0.9 * len(data))
# print(n)
train_data = data[:n]
test_data = data[n:]

block_size = 8
x = train_data[:block_size]
y = train_data[1:block_size + 1]

for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    # print(f"when input is {context} output is {target}")

# let's say in english if a sentence is like `I love` mostly next word is `you`

torch.manual_seed(42)
batch_size = 4
block_size = 8

def get_batch(split):
    data  = train_data if split == 'train' else test_data
    idx = torch.randint(len(data) - block_size, (batch_size,))
    # print(idx)
    x = torch.stack([data[i : i+block_size] for i in idx])
    y = torch.stack([data[i+1 : i+block_size+1] for i in idx])
    return x, y

xb, yb = get_batch('train')


# print(xb.shape) # [4, 8]
# print(xb)


class BiGramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BiGramLanguageModel(vocab_size)
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))


optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32


for steps in range(20000):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()


print(f"## LOSS IS ##")
print(loss.item())

print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist())) # somehow the generation is meaningful



