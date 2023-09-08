__all__ = ['EiDense', 'EiDenseWithShunt', 'EiDense_MultipleInt_LayerNorm_WeightInitPolicy',
           'EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy', 'DalesANN_SGD_UpdatePolicy', 'cSGD_Mixin',
           'ScaleLR_Mixin', 'ClipGradNorm_Mixin', 'DalesANN_UpdatePolicy', 'DalesANN_cSGD_UpdatePolicy']

import numpy as np
import torch
import torch.nn as nn
from .utils import all_utils as utils
from .base_models import BaseDenseLayer, DenseNet, build_densenet
from .base_models import BaseWeightInitPolicy, BaseUpdatePolicy, set_attr_for_all_layers


# =============================================================================
# Define Dale Inhibition Layers Classes
# =============================================================================
class EiDense(BaseDenseLayer):
    """
    Class modeling a Feed-forward inhibition layer without shunting
    """
    def __init__(self, n_input, layer_params: 'tuple', nonlinearity=None, weight_init_gain=None):
        """
        layer_params: (tuple) (no. excitatory, no. inhibitory)
        """
        super().__init__(n_input, layer_params[0], nonlinearity, weight_init_gain)

        self.in_features = n_input
        self.out_features = layer_params[0]

        self.ne = layer_params[0]
        self.ni = layer_params[1]

        # to-from notation - W_post_pre and the shape is n_output x n_input
        self.Wex = nn.Parameter(torch.empty(self.ne, self.n_input))
        self.Wix = nn.Parameter(torch.empty(self.ni, self.n_input))
        self.Wei = nn.Parameter(torch.empty(self.ne, self.ni))
        self.b = nn.Parameter(torch.empty(self.ne, 1))

        self.weight_init_policy = None
        self.update_policy = None

    @property
    def W(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    @property
    def weight(self):
        return self.Wex - torch.matmul(self.Wei, self.Wix)

    def forward(self, x):
        self.x = x
        self.z = torch.matmul(self.weight, self.x.T)
        self.z = self.z + self.b
        self.z = self.z.T   # maintain batch as first axis
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z

        if self.z.requires_grad:
            self.z.retain_grad()
            self.h.retain_grad()
        return self.h


class EiDenseWithShunt(EiDense):
    """
    Class modeling a Feed-forward inhibition layer including a shunting component of inhibition
    """
    def __init__(self, n_input, layer_params: '(ne, ni)', nonlinearity=None, weight_init_gain=None):
        """
        layer_params: (tuple) (no. excitatory, no. inhibitory)
        """
        super().__init__(n_input, layer_params, nonlinearity, weight_init_gain)
        self.alpha = nn.Parameter(torch.ones(size=(1, self.ni))) # row vector
        self.g = nn.Parameter(torch.ones(size=(self.ne, 1)))
        self.epsilon = 1e-8

    def forward(self, x):
        # print(x.shape)
        self.x = x
        self.ze = self.Wex@self.x.T  # ne x batch
        self.zi = self.Wix@self.x.T  # ni x btch

        # ne x batch = ne x batch - nexni ni x batch
        self.z_hat = self.ze - self.Wei@self.zi
        self.exp_alpha = torch.exp(self.alpha) # 1 x ni

        # ne x batch = (1xni * ^ne^xni ) @ nix^btch^ +  nex1
        self.gamma = ((self.exp_alpha*self.Wei)@self.zi) + self.epsilon

        # ne x batch = ne x batch * ne x batch
        self.z = (1/ self.gamma) * self.z_hat

        # ne x batch = nex1*ne x batch + nex1
        self.z = self.g*self.z + self.b
        # batch x ne
        self.z = self.z.T       # return batch to be first axis
        if self.nonlinearity is not None:
            self.h = self.nonlinearity(self.z)
        else:
            self.h = self.z.clone()

        # retaining grad for ngd calculations
        if self.zi.requires_grad:
            self.zi.retain_grad()
            self.z.retain_grad()
            self.gamma.retain_grad()
        return self.h


# =============================================================================
# Weight Initializations
# =============================================================================
class EiDense_MultipleInt_LayerNorm_WeightInitPolicy(BaseWeightInitPolicy):
    """
    This weight init policy assumes model with attrs:
    Wex,Wix,Wei,b, where ni >= 1.
    """
    def __init__(self, inhib_iid_init=False):
        """
        inhib_iid_init : bool, if true draw Wei and Wie iid.
                               if false draw Wei Wee iid, and Wie 1/ni
        """
        self.inhib_iid_init = inhib_iid_init

    def init_weights(self, layer):
        if layer.parent_index == 0:
            target_std = np.sqrt(1/(2*layer.n_input))
        else:
            target_std = np.sqrt(2*np.pi/(layer.n_input*(2*np.pi-1)))

        exp_scale = target_std  # The scale parameter, \beta = 1/\lambda = std
        Wex_np = np.random.exponential(scale=exp_scale, size=(layer.ne, layer.n_input))

        if layer.ni == 1:  # for example the output layer
            Wix_np = Wex_np.mean(axis=0, keepdims=True)  # not random as only one int
            Wei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni

        elif layer.ni != 1:
            if self.inhib_iid_init:
                # both sets of inhib units weights are drawn from the same distribution
                inhib_scale = np.sqrt(exp_scale/layer.ni)
                Wix_np = np.random.exponential(scale=inhib_scale, size=(layer.ni, layer.n_input))
                Wei_np = np.random.exponential(scale=inhib_scale, size=(layer.ne, layer.ni))
            else:
                # We consider wee ~ wie, and the inhib outputs are Wei <- 1/ni
                Wix_np = np.random.exponential(scale=exp_scale, size=(layer.ni, layer.n_input))
                Wei_np = np.ones(shape=(layer.ne, layer.ni))/layer.ni

        layer.Wex.data = torch.from_numpy(Wex_np).float()
        layer.Wix.data = torch.from_numpy(Wix_np).float()
        layer.Wei.data = torch.from_numpy(Wei_np).float()

        nn.init.zeros_(layer.b)


class EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy(EiDense_MultipleInt_LayerNorm_WeightInitPolicy):
    def init_weights(self, layer):
        super().init_weights(layer)
        a_numpy = np.sqrt((2*np.pi-1)/layer.n_input) * np.ones(shape=layer.alpha.shape)
        a = torch.from_numpy(a_numpy)
        alpha_val = torch.log(a)
        layer.alpha.data = alpha_val.float()


# =============================================================================
# Update Policy
# =============================================================================
class DalesANN_SGD_UpdatePolicy(BaseUpdatePolicy):
    '''
    This update does SGD (i.e. p.grad * lr) and then clamps Wix, Wex, Wei, g

    This should be inherited on the furthest right for correct MRO.
    '''
    def update(self, layer, lr, lr_ei=None, lr_ix=None, **args):
        """
        Args:
            lr : learning rate
        """
        # lr = args['lr']
        with torch.no_grad():
            if hasattr(layer, 'g'):
                layer.g -= layer.g.grad * lr
            layer.b -= layer.b.grad * lr
            layer.Wex -= layer.Wex.grad * lr
            if lr_ei:
                layer.Wei -= layer.Wei.grad * lr_ei
            else:
                layer.Wei -= layer.Wei.grad * lr
            if lr_ix:
                layer.Wix -= layer.Wix.grad * lr_ix
            else:
                layer.Wix -= layer.Wix.grad * lr
            if hasattr(layer, 'alpha'):
                layer.alpha -= layer.alpha.grad * lr

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


class ScaleLR_Mixin():
    '''
    Scales the lr so that cosine of angle between new params before and after clipping is
    within a constraint of 1-epsilon.
    '''
    max_trials = 100

    def __init__(self, cosine_angle_epsilon, lr_max, lr_min, lr_n, *args, **kwargs):
        """
        Args:
            cosine_angle_epsilon : Cosine of angle should be within 1-epsilon. If 1, then this
                      Mixin does not scale lr,  instead just computes the angle.
            lr_max :
            lr_min :
            lr_n :
        """

        self.cosine_angle_epsilon = cosine_angle_epsilon
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.lr_n = lr_n
        self.lr_i = 0  # keep track of the current index for lr_arr
        self.lr_arr = np.linspace(start=lr_max, stop=lr_min, num=lr_n)
        self.cos_angle = None  # initialise a variable to store this
        self.params_to_consider = {}  # parameters that will be clipped

        super().__init__(*args, **kwargs)

        self.verbose = False  # to print output or not
        self.scalelr_counter = 0  # for counting times lr was scaled

    @property
    def min_cosine_angle(self):
        return 1 - self.cosine_angle_epsilon

    @staticmethod
    def check_candidate_lr_angle(lr, theta, grad_vec):
        """ Returns the cosine of angle between new params before and after clipping"""
        with torch.no_grad():
            theta_new = theta - (lr * grad_vec)
            theta_clip = torch.clamp(theta_new, min=0)
            angle = torch.norm(theta_clip, 2) / torch.norm(theta_new, 2)
        return angle

    def set_params_to_consider(self, layer):
        self.params_to_consider = {}
        for k, p in layer.named_parameters():
            if p.grad is not None and k not in ["alpha", 'b']:
                self.params_to_consider[k] = p

    @property
    def theta(self):
        return utils.vectorise({k:p for k, p in self.params_to_consider.items()})

    @property
    def grad_vec(self):
        return utils.vectorise({k:p.grad for k, p in self.params_to_consider.items()})

    def update(self, layer, *args, cosine_angle_epsilon=None,lr_max=None,lr_min=None,lr_n=None, **kwargs):
        """
        Args:
            see self.__init__ docstring
        """

        changed = False

        assert 'lr' in kwargs.keys()

        if cosine_angle_epsilon is not None: self.cosine_angle_epsilon = cosine_angle_epsilon
        self.set_params_to_consider(layer)

        if self.min_cosine_angle == 0: # check if the main codeblock needs to be run
            # calculate the angle for logging regardless
            lr = kwargs['lr']
            self.cos_angle = self.check_candidate_lr_angle(lr, self.theta, self.grad_vec)
        else:
            # update attributes if corresponding args are passed in
            if any(arg is not None for arg in [lr_max, lr_min, lr_n]):
                try:
                    self.lr_arr = np.linspace(start=lr_max, stop=lr_min, num=lr_n)
                except:
                    print("Error, unable to build lr_arr, check the args")
                    raise
            if lr_max is not None: self.lr_max = lr_max
            if lr_min is not None: self.epsilon = lr_min
            if lr_n is not None: self.epsilon = lr_n

            cos_angle = 0
            with torch.no_grad():
                lr = self.lr_arr[self.lr_i]
                cos_angle = self.check_candidate_lr_angle(lr, self.theta, self.grad_vec)
                if cos_angle < self.min_cosine_angle:
                    # we need to decrease the learning rate
                    while cos_angle < self.min_cosine_angle:
                        if self.lr_i == self.lr_arr.shape[0]-1: # at end end of list, can't get smaller
                            break
                        if self.lr_i == self.lr_arr.shape[0]-2: # no point checking as can't get smaller after this
                            self.lr_i += 1
                            break

                        self.lr_i += 1
                        changed=True
                        lr = self.lr_arr[self.lr_i]
                        cos_angle = self.check_candidate_lr_angle(lr, self.theta, self.grad_vec)
                        if self.verbose: print(f'    lr_i is {self.lr_i}:, using lr {lr:.6f}: Cos(angle): {cos_angle:.6f} ')

                elif cos_angle >= self.min_cosine_angle and self.lr_i != 0:
                    # try to increase the learning rate
                    while cos_angle >= self.min_cosine_angle:
                        if self.lr_i == 0:
                            break
                        else:
                            self.lr_i -= 1
                            changed= True
                            lr = self.lr_arr[self.lr_i]
                            cos_angle = self.check_candidate_lr_angle(lr, self.theta, self.grad_vec)
                            if self.verbose: print(f'    lr_i is {self.lr_i}:, using lr {lr:.6f}: Cos(angle): {cos_angle:.6f} ')

                    if cos_angle < self.min_cosine_angle:
                        # we have gone one step too far and now angle is under the constraint
                        # we could also have broken from the loop because lr_i is 0
                        self.lr_i += 1

            self.cos_angle = self.check_candidate_lr_angle(lr,self.theta, self.grad_vec)
            if changed:
                self.scalelr_counter +=1
                pass

            if self.verbose: print(f'Verbose END!: lr_i is {self.lr_i}:, using lr {lr:.6f}: Cos(angle): {self.cos_angle:.6f}')
            kwargs['lr'] = lr
        super().update(layer, *args, **kwargs)


class ClipGradNorm_Mixin():

    def __init__(self, max_grad_norm, *args,**kwargs):
        self.max_grad_norm = max_grad_norm
        self.gn_counter = 0  # for counting how many times grads were scaled

        super().__init__(*args, **kwargs)

    def update(self, layer, max_grad_norm=None, *args, **kwargs):
        """
        Constrains the norm of the layer params gradient to be within a certain norm
        """
        if max_grad_norm is not None: self.max_norm =max_grad_norm

        if self.max_grad_norm > 0:
            layer_params = {k: p.grad for k, p in layer.named_parameters() if p.grad is not None}

            grad_vec = utils.vectorise(layer_params)
            grad_norm = torch.norm(grad_vec, p=2)

            clip_coef = self.max_grad_norm / (grad_norm + 1e-6)
            if clip_coef < 1:
                self.gn_counter += 1
                for k, g in layer_params.items():

                    g.detach().mul_(clip_coef.to(g.device))

        super().update(layer, *args, **kwargs)


# ------ Convenience update policies ------
class DalesANN_UpdatePolicy(cSGD_Mixin, ClipGradNorm_Mixin, ScaleLR_Mixin, DalesANN_SGD_UpdatePolicy):
    pass


class DalesANN_cSGD_UpdatePolicy(cSGD_Mixin, DalesANN_SGD_UpdatePolicy):
    pass
