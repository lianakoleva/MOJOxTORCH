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
- [x] torch.nn.ReLUg
- [ ] BertModel kernels