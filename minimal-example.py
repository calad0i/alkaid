import numpy as np
import torch
from alkaid.trace import trace
from alkaid.trace import FVArrayInput
from alkaid import converter
    

def my_torch_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    return torch.maximum(x1, x2)


def my_np_function(x):
    x1, x2 = x[:, 0], x[:, 1]
    return np.maximum(x1, x2)
    

if __name__ == "__main__":

    inp = FVArrayInput((1, 2)).quantize(k=0, i=1, f=0)

    # functional API works w/ numpy functions
    out = my_np_function(inp)
    print(trace(inp, out))

    # ... but not torch
    try:
        out = my_torch_function(inp)
    except Exception as e:
        print('Expected error:', e)

    # ... unless we load the torch plugin
    converter.load_plugin('torch')
    out = my_torch_function(inp)
    print(trace(inp, out))
