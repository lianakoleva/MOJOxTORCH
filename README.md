# 🔥mojo backend for pytorch

### steps
1. `pytest model_bert_test.py`
2. if changes made to mojo_dir/, `mojo package mojo_dir -o kernels.mojopkg`

### infra todo
- [ ] set up mojo backend

### kernels todo
- [x] torch.abs
- [x] torch.max
- [x] torch.allclose
- [x] torch.nn.functional.cosine_similarity
- [ ] BertModel kernels