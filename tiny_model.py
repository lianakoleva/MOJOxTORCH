import copy

import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertModel
from mojo_custom_class import CustomOpLibrary, register_custom_op
from pathlib import Path
import max.nn.linear
from activation import ReLU as _relu
from activation import Softmax as _softmax


class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


def original():
    torch.manual_seed(14) # used for testing to guarantee same results
    tinymodel = TinyModel()
    print('original model:')
    print(tinymodel)

    return tinymodel


def mojo():
    torch.manual_seed(14) # used for testing to guarantee same results
    tinymodel = TinyModel()
    print('mojo model:')
    print(tinymodel)

    return tinymodel


def main():
    original_model = original()
    torch.nn.linear = max.nn.linear.Linear
    torch.nn.ReLU = _relu
    torch.nn.Softmax = _softmax
    mojo_model = mojo()
    for o_param, m_param in zip(original_model.parameters(), mojo_model.parameters()):
        assert torch.allclose(o_param, m_param)
    for o_param, m_param in zip(original_model.linear1.parameters(), mojo_model.linear1.parameters()):
        assert torch.allclose(o_param, m_param)
    for o_param, m_param in zip(original_model.linear2.parameters(), mojo_model.linear2.parameters()):
        assert torch.allclose(o_param, m_param)

    print("mojo model passed 🔥")


if __name__ == "__main__":
    main()
