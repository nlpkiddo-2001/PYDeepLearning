#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install -q git+https://github.com/google-deepmind/gemma.git')
# it will install jax optax fax
get_ipython().system('pip install -U tensorflow')


# In[7]:


get_ipython().system('pip install -U tensorflow_datasets')


# In[2]:


import os
import enum
import re
import string

import chex
import jax
import jax.numpy as jnp
import optax

import tensorflow as tf
import tensorflow_datasets as tfds

from gemma import params as params_lib
from gemma import sampler as sampler_lib
from gemma import transformer as transformer_lib
import sentencepiece as spm


# In[22]:


get_ipython().system('pip install kaggle')


# In[3]:


import kaggle


# In[4]:


from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()


# In[6]:


import kagglehub

GEMMA_VARIANT = '2b'
GEMMA_PATH = kagglehub.model_download(f'google/gemma/flax/{GEMMA_VARIANT}')


# In[7]:


CKPT_PATH = os.path.join(GEMMA_PATH, GEMMA_VARIANT)
TOKENIZER_PATH = os.path.join(GEMMA_PATH, 'tokenizer.model')
print('CKPT_PATH:', CKPT_PATH)
print('TOKENIZER_PATH:', TOKENIZER_PATH)


# ### **LODAING DATASETS**

# In[18]:


ds = tfds.load("mtnt/en-fr", split="train")

ds = ds.take(2)
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()


# In[8]:


vocab = spm.SentencePieceProcessor()
vocab.Load(TOKENIZER_PATH)


# In[27]:


class GemmaTokenizer:

    def __init__(self,spm_processer: spm.SentencePieceProcessor):
        self._spm_processer = spm_processer

    @property
    def pad_id(self) -> int:
        """Fast access to the pad ID."""
        return self._spm_processer.pad_id()

    def tokenize(self,
                example: str | bytes,
                prefix: str = '',
                suffix: str = '',
                add_eos: bool = True) -> jax.Array:
        """
        The tokenization function

        Args:
            example: Input string to tokenize.
            prefix:  Prefix to add to the input string.
            suffix:  Suffix to add to the input string.
            add_eos: If True, add an "end of sentence" token at the end of the output
               sequence.
        Returns:
            Tokens corresponding to the input string.
        """
        int_list = [self._spm_processer.bos_id()]
        int_list.extend(self._spm_processer.EncodeAsIds(prefix + example + suffix))
        if add_eos:
            int_list.append(self._spm_processer.eos_id())

        return jnp.array(int_list, dtype=jnp.int32)

    def tokenize_tf_op(self,
                      str_tensor: tf.Tensor,
                      prefix: str = '',
                      suffix: str = '',
                      add_eos: bool = True) -> tf.Tensor:
        """A tensorflow operator for the tokenize function"""
        encoded = tf.numpy_function(
            self.tokenize,
            [str_tensor, prefix, suffix, add_eos],
            tf.int32
        )
        encoded.set_shape([None])
        return encoded

    def to_string(self, tokens: jax.Array) -> str:
        """Convert an array of tokens to string"""
        return self._spm_processer.EncodeAsIds(tokens.tolist())


# In[28]:


tokenizer = GemmaTokenizer(vocab)


# In[29]:


vocab.encode('fuck')


# In[30]:


vocab.decode([34024])


# In[31]:


def tokenize_source(tokenizer, example: tf.Tensor):
    return tokenizer.tokenize_tf_op(
        example,
        prefix = 'Translate this into french:\n',
        suffix = '\n',
        add_eos=False 
    )

def tokenize_destination(tokenizer, example: tf.Tensor):
    return tokenizer.tokenize_tf_op(example,
                                  add_eos=True)


# In[32]:


ds = tfds.load("mtnt/en-fr",split="train")
ds = ds.take(2)
ds = ds.map(lambda x: {'src': tokenize_source(tokenizer, x['src']),
                       'dst': tokenize_destination(tokenizer, x['dst'])})
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()


# In[36]:


@chex.dataclass(frozen=True)
class TrainingInput:
    input_tokens: jax.Array
    target_mask: jax.Array


class DatasetSplit(enum.Enum):
    TRAIN = 'train'
    VALIDATION = 'valid'

class MTNTDatasetBuilder:
    """The dataset builder for the MTNT dataset."""
    N_ITEMS = {DatasetSplit.TRAIN: 35_692,
             DatasetSplit.VALIDATION: 811}
    
    BUFFER_SIZE_SHUFFLE = 10_000
    TRANSLATION_PREFIX = 'Translate this into French:\n'
    TRANSLATION_SUFFIX = '\n'


    def __init__(self,
                tokenizer: GemmaTokenizer,
                max_seq_len: int):
        """Constructor.
    
        Args:
          tokenizer: Gemma tokenizer to use.
          max_seq_len: size of each sequence in a given batch.
        """

        self._tokenizer = tokenizer
        self._base_data = {
            DatasetSplit.TRAIN: tfds.load('mtnt/en-fr',split='train'),
            DatasetSplit.VALIDATION: tfds.load("mtnt/en-fr",split="valid")
        }
        self._max_seq_len = max_seq_len

    def _tokenize_source(self, example: tf.Tensor):
        """Tokenization function for the source."""
        return self._tokenizer.tokenize_tf_op(example,
                                              prefix=self.TRANSLATION_PREFIX,
                                              suffix=self.TRANSLATION_SUFFIX,
                                              add_eos=False)

    def _tokenize_destination(self, example: tf.Tensor):
        """Tokenization function for the French translation."""
        return self._tokenizer.tokenize_tf_op(example,
                                              add_eos=True)


    def _pad_up_to_max_len(self,
                          input_tensor: tf.Tensor,
                          pad_value: int | bool):
        seq_len = tf.shape(input_tensor)[0]
        to_pad = tf.maximum(self._max_seq_len - seq_len, 0)
        return tf.pad(input_tensor,
                     [[0, to_pad]],
                     mode='CONSTANT',
                     constant_values=pad_value)

    def _to_training_input(self, 
                          src_tokens: jax.Array,
                          dst_tokens: jax.Array,
                          ) -> TrainingInput:
        """Building a training input from a tuple of source and destination tokens."""
        # concatenation of source and target
        tokens = tf.concat([src_tokens, dst_tokens], axis=0)

        q_mask = tf.zeros_like(src_tokens, dtype=tf.bool)
        a_mask = tf.ones_like(dst_tokens, dtype=tf.bool)
        mask = tf.concat([q_mask, a_mask], axis=0)
        print(mask)

        tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)

        mask = self._pad_up_to_max_len(mask, False)
        return TrainingInput(input_tokens=tokens, target_mask=mask)

    def get_train_dataset(self, batch_size: int, num_epochs: int):
        """Build the training dataset."""
    
        # Tokenize each sample.
        ds = self._base_data[DatasetSplit.TRAIN].map(lambda x : (self._tokenize_source(x['src']),
                                                                 self._tokenize_destination(x['dst'])))
    
        # Convert the samples to training inputs.
        ds = ds.map(lambda x, y: self._to_training_input(x, y))
    
        # Remove the samples that are too long.
        ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    
        # Shuffle the dataset.
        ds = ds.shuffle(buffer_size=self.BUFFER_SIZE_SHUFFLE)
    
        # Repeat if necessary.
        ds = ds.repeat(num_epochs)
    
        # Build batches.
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds

    def get_validation_dataset(self, batch_size: int):
        """Build the validation dataset."""
    
        # Same steps as in `get_train_dataset`, but without shuffling and no repetition.
        ds = self._base_data[DatasetSplit.VALIDATION].map(lambda x : (self._tokenize_source(x['src']),
                                                                      self._tokenize_destination(x['dst'])))
        ds = ds.map(lambda x, y: self._to_training_input(x, y))
        ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
        ds = ds.batch(batch_size, drop_remainder=True)
        return ds




# In[37]:


tokenizer = GemmaTokenizer(vocab)

dataset_builder = MTNTDatasetBuilder(tokenizer, max_seq_len=20)
ds = dataset_builder.get_train_dataset(3, 1)
ds = ds.take(2)
ds = ds.as_numpy_iterator()

for idx, example in enumerate(ds):
  print(f'Example {idx}:')
  for key, val in example.items():
    print(f'{key}: {val}')
  print()


# ### **MODEL CONFIG**

# In[38]:


params = params_lib.load_and_format_params(CKPT_PATH)


# In[39]:


params


# In[40]:


config_2b = transformer_lib.TransformerConfig.from_params(
    params,
    cache_size=30
)

model_2b = transformer_lib.Transformer(config=config_2b)


# In[41]:


def forward_and_loss_fn(params,
                       *,
                       model,
                       input_tokens,
                       input_mask,
                       positions,
                       attention_mask):
    logits, _  = model.apply(
        params, 
        input_tokens, 
        positions, 
        None, 
        attention_mask
    )

    logits = logits[0, :-1]
    
    target_tokens = input_tokens[0, 1:]
    target_mask = input_mask[0, 1:]

    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

    one_hot = one_hot * target_mask.astype(one_hot.dtype)[...,None]

    norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

    return -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor


# In[42]:


def get_attention_mask_and_positions(example, pad_id):
    pad_mask = example != pad_id
    current_token_position = transformer_lib.build_positions_from_mask(pad_mask)
    attention_mask = transformers_lib.make_causal_attn_mask(pad_mask)
    return current_token_position, attention_mask


# In[43]:


def train_step(
    model,
    params,
    optmizer,
    opt_state,
    pad_id,
    example
):
    positions, attention_mask = get_attention_mask_and_positions(
        example.input_tokens,
        pad_id
    )

    train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(params,
                                                               model=model,
                                                               input_tokens=example.input_tokens,
                                                               input_mask = example.target_mask,
                                                               positions=positions,
                                                               attention_mask=attention_mask)
    updates, opt_state = optmizer.update(grads, opt_state)
    params = optax.apply_updates(param, updates)

    return train_loss, params, opt_state


# In[44]:


def validation_step(model: transformer_lib.Transformer,
                    params,
                    pad_id: int,
                    example: TrainingInput,
                    ):
  positions, attention_mask = get_attention_mask_and_positions(example.input_tokens, pad_id)
  val_loss = forward_and_loss_fn(params,
                                 model=model,
                                 input_tokens=example.input_tokens,
                                 input_mask=example.target_mask,
                                 positions=positions,
                                 attention_mask=attention_mask)
  return val_loss


# In[47]:


@chex.dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    num_epochs: int
    eval_every_n: int
    batch_size: int
    max_steps: int | None = None

def train_loop(
    model,
    params,
    dataset_builder,
    training_cfg
):
    compiled_train_step = jax.jit(train_step, static_argnames=['model','optimizer'])

    compiled_validation_step = jax.jit(validation_step, static_argnames=['model'])

    optimzer = optax.sgd(training_cfg.learning_rate)
    opt_stae = optimzer.init(params)

    train_ds = dataset_builder.get_train_dataset(batch_size=training_cfg.batch_size,
                                               num_epochs=training_cfg.num_epochs)
    train_ds = train_ds.as_numpy_iterator()

    validation_ds = dataset_builder.get_validation_dataset(batch_size=training_cfg.batch_size)
    validation_ds = validation_ds.take(50)

    n_steps = 0
    avg_loss = 0

    n_steps_eval = 0
    eval_loss = 0
    val_iterator = validation_ds.as_numpy_iterator()
    for val_example in val_iterator:
        eval_loss += compiled_validation_step(
            model,
            params,
            dataset_builder._tokenizer.pad_ids,
            val_example
        )
        n_steps_eval += 1

    print(f"Start, Validation loss :: {eval_loss/n_steps_eval}")

    for train_example in train_ds:
        train_loss, params, opt_state = compiled_train_step(model=model,
                                                           params=params,
                                                           optimizer=optimizer,
                                                           opt_state=opt_state,
                                                           pad_id=dataset_builder._tokenizer.pad_id,
                                                           example=train_example)
        n_steps += 1
        avg_loss += train_loss
        if n_steps % training_cfg.eval_every_n == 0:
            eval_loss = 0

            n_steps_eval = 0
            val_iterator = validation_ds.as_numpy_iterator()
            for val_example in val_iterator:
                eval_loss += compiled_validation_step(model,
                                                     params,
                                                     dataset_builder._tokenizer.pad_id,
                                                      val_example)
                n_steps_eval += 1
            avg_loss /= training_cfg.eval_every_n
            eval_loss /= n_steps_eval
            print(f"STEP {n_steps} training loss: {avg_loss} - eval loss: {eval_loss}")
            avg_loss = 0
        if training_cfg.max_steps is not None and n_steps>training_cfg.max_steps:
            break
    return params


# In[ ]:


SEQ_SIZE = 25
tokenizer = GemmaTokenizer(vocab)
dataset_builder= MTNTDatasetBuilder(tokenizer, SEQ_SIZE)
training_cfg = TrainingConfig(learning_rate=1e-4,
                              num_epochs=1,
                              eval_every_n=20,
                              batch_size=1,
                              max_steps=100)

params = train_loop(model=model_2b,
                    params={'params': params['transformer']},
                    dataset_builder=dataset_builder,
                    training_cfg=training_cfg)

