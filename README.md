# 🔥mojo backend for pytorch

### steps
1. `python tiny_model.py` and/or `python model_bert_test.py`
2. if changes made to mojo_dir/, `mojo package mojo_dir -o kernels.mojopkg`

### infra todo
- [ ] set up mojo backend

### kernels todo
- [x] torch.abs
- [x] torch.max
- [x] torch.allclose
- [x] torch.nn.functional.cosine_similarity
- [x] torch.nn.Linear
- [x] torch.nn.ReLU*
- [x] torch.nn.Softmax*
- [ ] BertModel kernels

\**note: do not print properly when printing the model*

### good to know
this project uses relative imports. 

if you see any errors like
```
ValueError: Path provided as custom extension to Graph must be a Mojo source or binary package: kernels.mojopkg
```
that means `kernels.mojopkg` cannot be found by `examples/*.py`. please run these files from the root directory like
```
python examples/*.py
```

if you see any errors like
```
ModuleNotFoundError: No module named 'mojo_custom_class'
```
run this in the root directory:
```
export PYTHONPATH=$(pwd)
```