import sys

from tt_gpt3_args import TtPreprocessingGpt3Args
from tools.preprocess_data import data_info

if __name__ == "__main__":
    opts = TtPreprocessingGpt3Args()
    sys.argv.append(f'--input={opts.input}')
    sys.argv.append(f'--output-prefix={opts.output_prefix}')
    sys.argv.append(f'--vocab={opts.vocab}')
    sys.argv.append(f'--dataset-impl={opts.dataset_impl}')
    sys.argv.append(f'--tokenizer-type={opts.tokenizer_type}')
    sys.argv.append(f'--merge-file={opts.merge_file}')
    if opts.append_eod:
        sys.argv.append(f'--append-eod')
    sys.argv.append(f'--workers={opts.workers}')

    data_info()