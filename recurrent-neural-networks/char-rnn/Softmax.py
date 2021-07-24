import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    print("np.exp(x) ", np.exp(x))
    print('Sum of exp(x)', np.sum(np.exp(x), axis =0))
    return np.exp(x) / np.sum(np.exp(x), axis =0) # axis = 0 across the columns


x = np.array([2.0,1.0,0.1])
outputs = softmax(x)

print('softmax numpy', outputs)


x = torch.tensor([2.0, 1.0, 0.1])

outputs = torch.softmax(x, dim=0)