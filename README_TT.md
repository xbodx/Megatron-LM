# pyenv

```
pyenv install --list
pyenv install 3.6.8
pyenv versions
pyenv local 3.6.8
```

# venv

in windows
```
python -m venv .nlp_pretrain_env
.\.nlp_pretrain_env\Scripts\activate.bat 
python -m pip install --upgrade pip

pip install pybind11 six regex numpy nltk tokenizers
pip install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/xbodx/apex.git
pip install git+https://github.com/xbodx/Megatron-LM-Helpers.git
# or
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ../../libs/apex
pip install ../../libs/MegatronHelpers
```

in linux
```
python -m venv .nlp_pretrain_env
source ./.nlp_pretrain_env/bin/activate
python -m pip install --upgrade pip

pip install pybind11 six regex numpy nltk tokenizers
pip install torch==1.5.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" git+https://github.com/NVIDIA/apex.git
pip install git+https://github.com/xbodx/Megatron-LM-Helpers.git
```

# run dev tokenizer, preprocessing, train

tokenizer
```
screen python tt_tokenizer_gpt3.py
```

peprocessing
```
screen python tt_preprocessing_gpt3.py
```

train
```
screen python tt_pretrain_gpt3.py
```
