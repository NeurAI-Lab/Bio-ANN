__all__ = ['DaleDendrites_SGD_UpdatePolicy', 'DaleDendrites_cSGD_UpdatePolicy']

import numpy as np
import torch
from backbone.dale_nn.base_models import BaseUpdatePolicy


class DaleDendrites_SGD_UpdatePolicy(BaseUpdatePolicy):
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g

    This should be inherited on the furthest right for correct MRO.
    '''
    def update(self, layer, lr, lr_ei=None, lr_ix=None, **args):
        """
        Args:
            lr : learning rate
        """
        with torch.no_grad():
            if hasattr(layer.module, 'g'):
                layer.module.g -= layer.module.g.grad * lr
            layer.module.b -= layer.module.b.grad * lr
            layer.module.Wex -= layer.module.Wex.grad * lr
            if lr_ei:
                layer.module.Wei -= layer.module.Wei.grad * lr_ei
            else:
                layer.module.Wei -= layer.module.Wei.grad * lr
            if lr_ix:
                layer.module.Wix -= layer.module.Wix.grad * lr_ix
            else:
                layer.module.Wix -= layer.module.Wix.grad * lr

            if hasattr(layer.module, 'alpha'):
                layer.module.alpha -= layer.module.alpha.grad * lr

            layer.segments.weights -= layer.segments.weights.grad * lr

            layer.module.Wix.data = torch.clamp(layer.module.Wix, min=0)
            layer.module.Wex.data = torch.clamp(layer.module.Wex, min=0)
            layer.module.Wei.data = torch.clamp(layer.module.Wei, min=0)
            if hasattr(layer.module, 'g'):
                layer.module.g.data = torch.clamp(layer.module.g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()


class cSGD_Mixin():
    "DANN update corrections"

    def update(self, layer, **args):
        with torch.no_grad():
            layer.module.Wix.grad = layer.module.Wix.grad / np.sqrt(layer.module.ne)
            layer.module.Wei.grad = layer.module.Wei.grad / layer.module.n_input
            if hasattr(layer.module, 'alpha'):
                layer.module.alpha.grad = layer.module.alpha.grad / (np.sqrt(layer.module.ne) * layer.module.n_input)
        super().update(layer, **args)


class HebbianUpdate(BaseUpdatePolicy):
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g

    This should be inherited on the furthest right for correct MRO.
    '''
    def update(self, layer, lr, context, **args):
        """
        Args:
            lr : learning rate
        """
        with torch.no_grad():
            with torch.no_grad():
                context = context[:1]
                y = torch.einsum("ijk,bk->bij", layer.segments.weights, context)
                max_values = y.abs().max(dim=2, keepdim=True).values
                y_mask = (y.abs() == max_values).type(torch.float)
                y = y_mask * y

                yx = 0
                chunk = 32
                n_splits = np.ceil(context.shape[0] / chunk)
                start_idx = 0
                for i in range(int(n_splits)):
                    yx += torch.einsum("ijk,il->ijkl", y[start_idx: start_idx + chunk], context[start_idx: start_idx + chunk])
                    start_idx += chunk
                yx = yx.mean(dim=0)
                y2 = (y ** 2).mean(dim=0)
                y2w = torch.einsum("ij,ijk->ijk", y2, layer.segments.weights)
                dW = yx - y2w
                layer.segments.weights += lr * dW


class DaleDendrites_cSGD_UpdatePolicy(cSGD_Mixin, DaleDendrites_SGD_UpdatePolicy):
    pass
