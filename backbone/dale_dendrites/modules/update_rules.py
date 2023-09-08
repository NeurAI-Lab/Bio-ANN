__all__ = ['DaleDendrites_SGD_UpdatePolicy', 'DaleDendrites_cSGD_UpdatePolicy']

import numpy as np
import torch
from backbone.dale_nn.base_models import BaseUpdatePolicy


class DaleDendrites_SGD_UpdatePolicy(BaseUpdatePolicy):
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g

    This should be inherited on the furthest right for correct MRO.
    '''
    def update(self, layer, lr, **args):
        """
        Args:
            lr : learning rate
        """
        with torch.no_grad():
            if hasattr(layer, 'g'):
                layer.g -= layer.g.grad * lr
            layer.b -= layer.b.grad * lr
            layer.Wex -= layer.Wex.grad * lr
            layer.Wei -= layer.Wei.grad * lr
            layer.Wix -= layer.Wix.grad * lr
            if hasattr(layer, 'alpha'):
                layer.alpha -= layer.alpha.grad * lr

            layer.segments.weights -= layer.segments.weights.grad * lr

            layer.Wix.data = torch.clamp(layer.Wix, min=0)
            layer.Wex.data = torch.clamp(layer.Wex, min=0)
            layer.Wei.data = torch.clamp(layer.Wei, min=0)
            if hasattr(layer, 'g'):
                layer.g.data = torch.clamp(layer.g, min=0)
            # layer.alpha does not need to be clamped as is exponetiated in forward()


class cSGD_Mixin():
    "DANN update corrections"

    def update(self, layer, **args):
        with torch.no_grad():
            layer.Wix.grad = layer.Wix.grad / np.sqrt(layer.ne)
            layer.Wei.grad = layer.Wei.grad / layer.n_input
            if hasattr(layer, 'alpha'):
                layer.alpha.grad = layer.alpha.grad / (np.sqrt(layer.ne) * layer.n_input)
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

            context = context[:1]
            y = torch.einsum("ijk,bk->bij", layer.segments.weights, context)
            max_values = y.abs().max(dim=2, keepdim=True).values
            y_mask = (y.abs() == max_values).type(torch.float)
            y = y_mask * y

            yx = 0
            chunk = 16
            n_splits = np.ceil(context.shape[0] / chunk)
            start_idx = 0
            for i in range(int(n_splits)):
                yx += torch.einsum("ijk,il->ijkl", y[start_idx: start_idx + chunk], context[start_idx: start_idx + chunk])
                start_idx += chunk
            # yx = yx / n_splits
            # yx = torch.einsum("ijk,il->ijkl", y, context)
            yx = yx.mean(dim=0)

            y2 = (y ** 2).mean(dim=0)
            y2w = torch.einsum("ij,ijk->ijk", y2, layer.segments.weights)
            dW = yx - y2w
            # dW = yx
            # layer.segments.weights -= lr * dW
            layer.segments.weights += lr * dW

            # yx_ref = torch.einsum("ijk,il->ijkl", y, context)
            # yx_ref = yx_ref.mean(dim=0)
            # num_units, num_segments, context_dim = layer.segments.weights.shape
            # batch_size, context_dim = context.shape
            # context = context[:batch_size, None, None, :]
            # context_vec = context[:batch_size, None, None, :context_dim]
            # y = y[:batch_size, :num_units, :num_segments, None]
            # yx_ref = torch.einsum("ijk,il->ijkl", y, context)
            # yx = context_vec * y
            # yx = yx.mean(dim=0)
            # y2 = (y**2).mean(dim=0)
            # y2 = (y ** 2)
            # y2w = y2 * layer.segments.weights
            # y2w = torch.einsum("ij,ijk->ijk", y2, layer.segments.weights)
            # dW = yx - y2w
            # layer.segments.weights += lr * dW
            # print('done')


class DaleDendrites_cSGD_UpdatePolicy(cSGD_Mixin, DaleDendrites_SGD_UpdatePolicy):
    pass
