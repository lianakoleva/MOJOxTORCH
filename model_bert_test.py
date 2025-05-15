import copy

import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertModel
from mojo_custom_class import CustomOpLibrary, register_custom_op
from pathlib import Path


device = "cpu"

@pytest.mark.parametrize(
    "prompt",
    ["How are you today?"],
)
@pytest.mark.parametrize("dtype", [torch.float32]) # mojo does not support float16 for CPU
def test_accuracy_bert(prompt, dtype):
    op_library = CustomOpLibrary(Path("./kernels.mojopkg"))
    _abs = register_custom_op(op_library.abs)
    @_abs.register_fake
    def _(x):
        y = torch.empty_like(x)
        return y
    torch.abs = _abs
    _max = register_custom_op(op_library.max)
    @_max.register_fake
    def _(x):
        return torch.tensor(-1, dtype=x.dtype)
    torch.max = _max
    _allclose = register_custom_op(op_library.allclose)
    @_allclose.register_fake
    def _(x, y):
        return x == y
    torch.allclose = _allclose
    _cosine_similarity = register_custom_op(op_library.cosine_similarity)
    @_cosine_similarity.register_fake
    def _(x, y, dim, eps):
        return 0.0
    torch.cosine_similarity = _cosine_similarity

    config = BertConfig()
    model = BertModel(config)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    ref_model = copy.deepcopy(model)
    ref_model.to(torch.float64).to(device).eval()
    ref_inputs = copy.deepcopy(inputs).to(torch.float64)
    with torch.no_grad():
        ref_outputs = ref_model(**ref_inputs).last_hidden_state.to(dtype)

    res_model = copy.deepcopy(model)
    res_model.to(dtype).to(device).eval()
    res_inputs = copy.deepcopy(inputs).to(dtype)
    with torch.no_grad():
        res_outputs = res_model(**res_inputs).last_hidden_state

    maxdiff = torch.max(torch.abs(ref_outputs - res_outputs))
    succeed = True
    if (
        torch.allclose(
            ref_outputs,
            res_outputs,
            # TODO: figure out how to pass overloaded (e.g., torch.allclose.{overload} = _allclose{full})
            # atol=1e-3,
            # rtol=1e-3,
        )
        is False
    ):
        score = torch.nn.functional.cosine_similarity(
            ref_outputs.flatten(),
            res_outputs.flatten(),
            dim=0,
            eps=1e-6,
        )
        succeed = score >= 0.99
        print("score: ", score, "succeed: ", succeed, "maxdiff: ", maxdiff)
    else: # with float32, the if statement is not entered - want to test the operators
        score = torch.nn.functional.cosine_similarity(
            ref_outputs.flatten(),
            res_outputs.flatten(),
            dim=0,
            eps=1e-6,
        )
        succeed = score >= 0.99
        print("score: ", score, "succeed: ", succeed, "maxdiff: ", maxdiff)
    assert (
        succeed
    ), f"BERT_{dtype} FAIL with maxdiff {maxdiff} and score {score}\nREF: \
        {ref_outputs}\nRES: {res_outputs}"

def main():
    test_accuracy_bert("How are you today?", torch.float32)

if __name__ == "__main__":
    main()