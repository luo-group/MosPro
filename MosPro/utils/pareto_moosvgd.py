import math
import torch
import numpy as np

def solve_min_norm_2_loss(grad_1, grad_2):
    b, l, k = grad_1.shape
    grad_1 = grad_1.reshape(grad_1.shape[0], -1)
    grad_2 = grad_2.reshape(grad_2.shape[0], -1)
    v1v1 = torch.sum(grad_1*grad_1, dim=-1)
    v2v2 = torch.sum(grad_2*grad_2, dim=-1)
    v1v2 = torch.sum(grad_1*grad_2, dim=-1)
    gamma = torch.zeros_like(v1v1)
    gamma = -1.0 * ( (v1v2 - v2v2) / (v1v1+v2v2 - 2*v1v2) )
    gamma[v1v2>=v1v1] = 0.999
    gamma[v1v2>=v2v2] = 0.001
    gamma = gamma.view(-1, 1)
    # print('gamma', gamma.shape)
    # print('grad_1', grad_1.shape)
    # print('grad_2', grad_2.shape)
    # print('gamma.repeat(1, grad_1.shape[1])', gamma.repeat(1, grad_1.shape[1]).shape)
    g_w = gamma.repeat(1, grad_1.shape[1],)*grad_1 + (1.-gamma.repeat(1, grad_2.shape[1]))*grad_2
    # print('g_w', g_w.shape)
    g_w = g_w.reshape(b, l, k)
    # print('g_w', g_w.shape)
    # input()
    return g_w

def median(tensor):
    """
    torch.median() acts differently from np.median(). We want to simulate numpy implementation.
    """
    tensor = tensor.detach().flatten()
    tensor_max = tensor.max()[None]
    return (torch.cat((tensor, tensor_max)).median() + tensor.median()) / 2.

def kernel_functional_rbf(losses):
    n = losses.shape[0]
    # print(f'n: {n}')
    # print(f'losses: {losses}')
    pairwise_distance = torch.norm(losses[:, None] - losses, dim=2).pow(2)
    # print(f'pairwise_distance: {pairwise_distance}')
    h = median(pairwise_distance) / (math.log(n) if n > 1 else 1.)
    # print(f'h: {h}, {5e-6*h}')
    # print(f'')
    kernel_matrix = torch.exp(-pairwise_distance / (5e-6*h + 1e-5)) #5e-6 for zdt1,2,3 (no bracket)
    return kernel_matrix

def compose_two_gradients_moosvgd(grad_1, grad_2, inputs, score_1, score_2):
    scores = torch.cat([score_1.unsqueeze(1), score_2.unsqueeze(1)], dim=1)
    # Perforam gradient normalization trick 
    grad_1 = torch.nn.functional.normalize(grad_1, dim=0)
    grad_2 = torch.nn.functional.normalize(grad_2, dim=0)
    n = inputs.size(0)
    # inputs = inputs.detach().requires_grad_(True)
    
    g_w = solve_min_norm_2_loss(grad_1, grad_2)
    ### g_w (100, x_dim)
    # See https://github.com/activatedgeek/svgd/issues/1#issuecomment-649235844 for why there is a factor -0.5
    kernel = kernel_functional_rbf(scores)
    kernel_grad = -0.5 * torch.autograd.grad(kernel.sum(), inputs, allow_unused=True)[0]
    # print('g_w', g_w.shape, g_w)
    # print('kernel', kernel.shape)
    # print('kernel_grad', kernel_grad.shape)
    # input()
    original_shape = g_w.shape
    gradient = (kernel.mm(g_w.reshape(g_w.shape[0], -1)).reshape(original_shape) - kernel_grad) / n
    # print(gradient.shape)
    # print(gradient)
    # input()
    return gradient