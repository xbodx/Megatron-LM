from tt_gpt2_args import TtPretrainGpt2Args
from tt_pretrain import train_valid_test_datasets_provider, model_provider, forward_step, get_Torch_info, opts_to_argv
from megatron.training import pretrain

if __name__ == "__main__":
    opts = TtPretrainGpt2Args()
    opts_to_argv(opts)

    print(get_Torch_info())

    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': opts.tokenizer_type})
