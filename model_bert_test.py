import copy

import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertModel
from mojo_custom_class import CustomOpLibrary, register_custom_op
from pathlib import Path

device = "cuda:0"


'''@testop.register_fake
def _(pic):
    print("testop2")
    return pic.new_empty(pic.shape[:-1])'''


@pytest.mark.parametrize(
    "prompt",
    ["How are you today?"], # "What is your name?", "Who are you?", "Where are you from?"],
)
@pytest.mark.parametrize("dtype", [torch.float16]) #, torch.float32, torch.bfloat16])
def test_accuracy_bert(prompt, dtype):
    op_library = CustomOpLibrary(Path("./kernels.mojopkg"))
    #testop = register_custom_op(op_library.testop)
    abs = register_custom_op(op_library.abs)
    @abs.register_fake
    def _(x):
        print("absop")
        return x

    config = BertConfig()
    model = BertModel(config)
    tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    ref_model = copy.deepcopy(model)
    ref_model.to(torch.float32).to(device).eval()
    ref_inputs = copy.deepcopy(inputs).to(torch.float32)
    with torch.no_grad():
        ref_outputs = ref_model(**ref_inputs).last_hidden_state.to(dtype)

    res_model = copy.deepcopy(model)
    res_model.to(dtype).to(device).eval()
    res_inputs = copy.deepcopy(inputs).to(dtype)
    with torch.no_grad():
        res_outputs = res_model(**res_inputs).last_hidden_state

    print("ref_outputs.shape: ", ref_outputs.shape)
    print("res_outputs.shape: ", res_outputs.shape)
    maxdiff = torch.max(torch.abs(ref_outputs - res_outputs))
    succeed = True
    if (
        torch.allclose(
            ref_outputs,
            res_outputs,
            atol=1e-3,
            rtol=1e-3,
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
    assert (
        succeed
    ), f"BERT_{dtype} FAIL with maxdiff {maxdiff} and score {score}\nREF: \
        {ref_outputs}\nRES: {res_outputs}"
    
def main():
    test_accuracy_bert("How are you today?", torch.float16)

if __name__ == "__main__":
    main()