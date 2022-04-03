import torch
import math
import numpy as np


class Shader(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, current_u, current_v, X, width, height):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        return [X, X, X]

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        ans = torch.tensor(0.)
        return None, None, ans, None, None
    
width = 256
height = 256
extra_size = 2

xv, yv = np.meshgrid(np.arange(width + extra_size).astype('f') - 1, 
                     np.arange(height + extra_size).astype('f') - 1,
                     indexing='ij')

u = np.transpose(xv)
v = np.transpose(yv)

shader = Shader.apply

var = torch.tensor(0., requires_grad=True)

loss = shader(u, v, var, width, height)[0]
loss.backward()

print("success")