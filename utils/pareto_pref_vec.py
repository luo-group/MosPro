import torch
import numpy as np
from .min_norm_solvers import MinNormSolver

def circle_points(r, n):
    """
    generate evenly distributed unit preference vectors for two tasks
    """
    circles = []
    for r, n in zip(r, n):
        t = np.linspace(0, 0.5 * np.pi, n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        circles.append(np.c_[x, y])
    return circles

def get_d_paretomtl_init(grads,value,weights,i):
    """ 
    calculate the gradient direction for ParetoMTL initialization 
    """
    # print(f'called initial!')
    # input()
    flag = False
    nobj = value.shape
   
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
    # print(f'initial idx: {torch.sum(idx)}')
    # if torch.sum(idx) <= 2:
    #     print(f'initial idx: {torch.sum(idx)}')
    #     input()
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        flag = True
        return torch.zeros(nobj)
    if torch.sum(idx) == 1:
        sol = torch.ones(1).float().to(grads.device)
    else:
        vec =  torch.matmul(w[idx],grads)
        sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  torch.sum(torch.stack([sol[j] * w[idx][j ,0] for j in torch.arange(0, torch.sum(idx))]))
    weight1 =  torch.sum(torch.stack([sol[j] * w[idx][j ,1] for j in torch.arange(0, torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
   
    
    return weight

def get_d_paretomtl(grads,value,weights,i):
    """ calculate the gradient direction for ParetoMTL """
    # print(f'grads: {grads.shape}')
    # print(f'value: {value.shape}')
    # print(f'weights: {weights.shape}')
    # print(f'i: {i}')
    # input()
    # check active constraints
    current_weight = weights[i]
    rest_weights = weights 
    w = rest_weights - current_weight
    
    gx =  torch.matmul(w,value/torch.norm(value))
    idx = gx >  0
    # print(f'idx: {torch.sum(idx)}')
    # if torch.sum(idx) <= 2:
    #     print(f'idx: {torch.sum(idx)}')
    #     input()
        
    # calculate the descent direction
    if torch.sum(idx) <= 0:
        sol, nd = MinNormSolver.find_min_norm_element([[grads[t]] for t in range(len(grads))])
        return torch.tensor(sol).float().to(grads.device)


    vec =  torch.cat((grads, torch.matmul(w[idx],grads)))
    sol, nd = MinNormSolver.find_min_norm_element([[vec[t]] for t in range(len(vec))])


    weight0 =  sol[0] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,0] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight1 =  sol[1] + torch.sum(torch.stack([sol[j] * w[idx][j - 2 ,1] for j in torch.arange(2, 2 + torch.sum(idx))]))
    weight = torch.stack([weight0,weight1])
    # print(f'weight: {weight.shape}, {weight}')
    return weight
from tqdm import tqdm
def get_d_paretomtl_batch(grads,value,weights,i, initial=False):
    """ calculate the gradient direction for ParetoMTL """
    grad1, grad2 = grads
    value1, value2 = value
    assert len(grad1) == len(grad2)
    n = len(grad1)
    all_weight = []
    for idx in range(n):
        grad1_ = grad1[idx].flatten().unsqueeze(0)
        grad2_ = grad2[idx].flatten().unsqueeze(0)
        value1_ = value1[idx]
        value2_ = value2[idx]
        grad_ = torch.cat((grad1_, grad2_))
        value_ = torch.stack([value1_, value2_])
        # print(f'grad_: {grad_.shape}')
        # print(f'value_: {value_.shape}')
        # input()
        if not initial:
            weight = get_d_paretomtl(grad_, value_, weights, i)
        else:
            weight = get_d_paretomtl_init(grad_, value_, weights, i)
        # if n > 1:
        #     print(n)
        #     print(f'grad1_: {grad1_}')
        #     print(f'grad2_: {grad2_}')
        #     print(f'value1_: {value1_}')
        #     print(f'value2_: {value2_}')
        #     print(f'weight: {weight}')

        all_weight.append(weight)
    # if n > 1:
    #     print(f'all_weight: {all_weight}')
    #     all_weight = torch.vstack(all_weight)
    #     print(f'all_weight: {all_weight}')
    #     input()
    # all_weight = torch.cat(all_weight).reshape(2, -1)
    all_weight = torch.vstack(all_weight)
    # if n > 1:
    #     all_weight = all_weight.unsqueeze(-1).unsqueeze(-1)
    # print('all_weight.shape:', all_weight.shape)
    # print(all_weight[0], all_weight[1])
    # input()
    return all_weight