import random
import copy
import torch
import numpy as np


def pc_backward(losses, optimizer):
    def collect_task_grad():
        grads = []
        has_grads = []
        shapes = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    grads.append(torch.zeros_like(p.data))
                    has_grads.append(torch.zeros_like(p.data))
                    shapes.append(p.shape)
                    continue
                grads.append(p.grad.clone())
                has_grads.append(torch.ones_like(p.data))
                shapes.append(p.shape)

        grads = torch.cat([g.flatten() for g in grads])
        has_grads = torch.cat([g.flatten() for g in has_grads])
        return grads, has_grads, shapes

    grads = []
    has_grads = []
    shapes = []
    for loss in losses:
        optimizer.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)
        grad, has_grad, shape = collect_task_grad()
        grads.append(grad)
        has_grads.append(has_grad)
        shapes.append(shape)

    pc_grad = copy.deepcopy(grads)

    # Fix conflicting gradients
    for g_i in pc_grad:
        random.shuffle(grads)
        for g_j in grads:
            innner_product = torch.dot(g_i, g_j)
            if innner_product < 0:
                g_i -= innner_product * g_j / (g_j.norm() ** 2)  # Inplace operation

    shared = torch.stack(has_grads).prod(0).bool()
    merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
    merged_grad[shared] = torch.stack([g[shared]
                                       for g in pc_grad]).mean(dim=0)

    merged_grad[~shared] = torch.stack([g[~shared]
                                        for g in pc_grad]).sum(dim=0)

    pc_grad = merged_grad

    def unflatten_grad(grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    pc_grad = unflatten_grad(pc_grad, shapes[0])

    def set_grad(grads):
        """ set the modified gradients to the network """
        idx = 0
        for group in optimizer.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1

    set_grad(pc_grad)
