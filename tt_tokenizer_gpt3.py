import os

from tokenizers import CharBPETokenizer, models, pre_tokenizers, decoders, trainers, processors, normalizers
from tokenizers import Tokenizer

from tt_gpt3_args import VOCAB_PATH, RAW_PATH

def collect_files(path: str, files: []):
    for pt in os.listdir(path):
        fn = os.path.join(path, pt)
        if os.path.isfile(fn):
            files.append(fn)
        else:
            collect_files(fn, files)

def train_char_BPE_tokenizer(vocabName: str):
    files = []
    collect_files(RAW_PATH(), files)
    print(f'START! found {len(files)} files')

    tokenizer = CharBPETokenizer()
    tokenizer.train(files)
    tokenizer.save(os.path.join(VOCAB_PATH(), vocabName))

    encoded = tokenizer.encode("Привет, как дела?")
    print(f'FINISH! {encoded}')

def train_byte_level_BPE_tokenizer(vocabName: str = None):
    files = []
    collect_files(RAW_PATH(), files)
    print(f'START! found {len(files)} files')

    tokenizer = Tokenizer(models.BPE())
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Lowercase()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=20000, min_frequency=2)
    tokenizer.train(trainer, files)
    if vocabName:
        tokenizer.save(os.path.join(VOCAB_PATH(), vocabName), pretty=False)
    else:
        tokenizer.model.save(VOCAB_PATH())

    encoded = tokenizer.encode("Test: Привет, как дела?")
    print(f'FINISH! {encoded}')

if __name__ == "__main__":
    # train_char_BPE_tokenizer('bpe-vocab.json')
    # train_byte_level_BPE_tokenizer('byte-level-bpe-vocab.json')
    train_byte_level_BPE_tokenizer()