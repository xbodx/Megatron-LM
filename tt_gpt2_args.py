from datetime import datetime
import socket

import os

now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(f'now {now}')

host = socket.gethostname()
print(f'host {host}')

RAW_PATH = '/data/dataset/preprocessing'
DATA_PATH = '/data/dataset/data'
VOCAB_PATH = '/data/dataset/vocab'
CHECKPOINT_PATH = '/data/dataset/checkpoints'

def get_checkpoint_path():
    return f'{CHECKPOINT_PATH}/checkpoints_{now}_{host}'

class TtPretrainGpt2Args:
    num_layers = 24
    hidden_size = 1024
    num_attention_heads = 16
    batch_size = 8
    seq_length = 1024
    max_position_embeddings = 1024
    train_iters = 500000
    lr_decay_iters = 320000
    save = get_checkpoint_path()
    load = get_checkpoint_path()
    data_path = os.path.join(DATA_PATH, 'tt-gpt2_text_documents')
    vocab_file = os.path.join(VOCAB_PATH, 'vocab.json')
    merge_file = os.path.join(VOCAB_PATH, 'merges.txt')
    data_impl = 'mmap' # 'infer', 'lazy', 'cached'
    split = 949, 50, 1
    distributed_backend = 'nccl' # 'gloo', 'mpi'
    lr = 0.00015
    min_lr = 0.00001
    lr_decay_style = 'cosine' # 'linear', 'exponential'
    weight_decay = 1
    clip_grad = .0
    warmup = .01
    tokenizer_type = 'GPT2BPETokenizer'
    checkpoint_activations = True
    log_interval = 100
    save_interval = 10000
    eval_interval = 1000
    eval_iters = 10
    fp16 = True

class TtPreprocessingGpt3Args:
    input = RAW_PATH
    output_prefix = os.path.join(DATA_PATH, 'tt-gpt2')
    vocab = os.path.join(VOCAB_PATH, 'vocab.json')
    merge_file = os.path.join(VOCAB_PATH, 'merges.txt')
    dataset_impl = 'mmap'
    tokenizer_type = 'GPT2BPETokenizer'
    append_eod = False
    workers = 24