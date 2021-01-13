"""Pretrain GPT3"""
import sys

import torch

from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu
from megatron.data.gpt2_dataset import build_train_valid_test_datasets
from megatron.model import GPT2Model
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids
from megatron.utils import reduce_losses
from tt_gpt3_args import TtPretrainGpt3Args


def model_provider():
    """Build the model."""

    print_rank_0('building GPT2 model ...')
    model = GPT2Model(num_tokentypes=0, parallel_output=True)

    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    eod_token = None # todo: tokenizer.eod

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        eod_token,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()

    # Get the batch.
    timers('batch generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch generator').stop()
    # Forward model.
    losses = model(tokens, position_ids, attention_mask, labels=labels)
    loss_mask = loss_mask.view(-1)
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    reduced_loss = reduce_losses([loss])

    return loss, {'lm loss': reduced_loss[0]}


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT3 ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT2 datasets ...")

    return train_ds, valid_ds, test_ds

def get_Torch_info() -> str:
    result = []
    result.append(f"torch.cuda.is_available = {'True' if torch.cuda.is_available() else 'False'}")
    result.append(f"Torch version = {torch.__version__}")
    if hasattr(torch, 'cuda_version'):
        result.append(f"Torch cuda_version = {torch.cuda_version}")
    return result

def opts_to_argv(opts):
    sys.argv.append(f'--num-layers={opts.num_layers}')
    sys.argv.append(f'--hidden-size={opts.hidden_size}')
    sys.argv.append(f'--num-attention-heads={opts.num_attention_heads}')
    sys.argv.append(f'--batch-size={opts.batch_size}')
    sys.argv.append(f'--seq-length={opts.seq_length}')
    sys.argv.append(f'--max-position-embeddings={opts.max_position_embeddings}')
    sys.argv.append(f'--train-iters={opts.train_iters}')
    sys.argv.append(f'--lr-decay-iters={opts.lr_decay_iters}')
    sys.argv.append(f'--save={opts.save}')
    sys.argv.append(f'--load={opts.load}')
    sys.argv.append(f'--data-path={opts.data_path}')
    sys.argv.append(f'--vocab-file={opts.vocab_file}')
    sys.argv.append(f'--merge-file={opts.merge_file}')
    sys.argv.append(f'--data-impl={opts.data_impl}')
    sys.argv.append(f'--split={opts.split}')
    sys.argv.append(f'--distributed-backend={opts.distributed_backend}')
    sys.argv.append(f'--lr={opts.lr}')
    sys.argv.append(f'--min-lr={opts.min_lr}')
    sys.argv.append(f'--lr-decay-style={opts.lr_decay_style}')
    sys.argv.append(f'--weight-decay={opts.weight_decay}')
    sys.argv.append(f'--clip-grad={opts.clip_grad}')
    sys.argv.append(f'--warmup={opts.warmup}')
    sys.argv.append(f'--tokenizer-type={opts.tokenizer_type}')
    sys.argv.append(f'--checkpoint-activations')
    if opts.checkpoint_activations:
        sys.argv.append(f'--checkpoint-activations')
    sys.argv.append(f'--log-interval={opts.log_interval}')
    sys.argv.append(f'--save-interval={opts.save_interval}')
    sys.argv.append(f'--eval-interval={opts.eval_interval}')
    sys.argv.append(f'--eval-iters={opts.eval_iters}')
    if opts.fp16:
        sys.argv.append(f'--fp16')
