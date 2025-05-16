import copy

import pytest
import torch
from transformers import AutoTokenizer, BertConfig, BertModel
from mojo_custom_class import CustomOpLibrary, register_custom_op
from pathlib import Path
import max.nn.linear
# import max.graph.ops

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

def main():
    op_library = CustomOpLibrary(Path("./kernels.mojopkg"))
    torch.nn.linear = max.nn.linear.Linear
    # torch.nn.ReLU = max.graph.ops.relu

    tinymodel = TinyModel()
    print('The model:')
    print(tinymodel)

    print('\n\nJust one layer:')
    print(tinymodel.linear2)

    print('\n\nModel params:')
    for param in tinymodel.parameters():
        print(param)

    print('\n\nLayer params:')
    for param in tinymodel.linear2.parameters():
        print(param)

if __name__ == "__main__":
    main()
