
__all__ = ['FisherLayerMixin', 'FisherDenseNet', 'FisherCalculator']

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base_models import DenseNet, DenseLayer, build_densenet
from .ei_models import EiDenseWithShunt, EiDense
from .ei_models import EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy
from .utils import all_utils as utils




class FisherLayerMixin():
    """
    Class meant to be used as a mixin class in multiple inheritance, e.g.:

    class FisherLayerClass(FisherLayerMixin, EiDenseWithShunt)
        pass

    WARNING: At the moment, vectorize grad is hardcoded for this to be EiDense
    """
    def __init__(self, *args, **kwargs):
        """

        """
        self.F = None # Fisher matrix (diagonal)
        self.fisher_calculator_object = None
        self.f_lambda = 1/40000 # Fisher learning rate
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def fisher_step(self,c=.5):
        """method should be called like optimizer.step() whenever
        the grads are populated from sampled targets and you want
        to step the estimate of the fisher information.

        Assumes the moving average scaling factor c has 0 < c <= 1."""

        F = self.fisher_calculator_object.calc_fisher(self)
        if self.F is not None:
            F = self.F*(1-c) + F*c
        self.F = F
        return self.F

    @torch.no_grad()
    def vectorize_grad(self):
        """returns a vectorized param gradient using the vectorize method of FisherCalculator

        Note, we have hardcoded this at the moment to be for eimodels, if needed change this to named
        params or something
        """
        return FisherCalculator.vectorize({'b':self.b.grad,'g':self.g.grad,'Wex':self.Wex.grad,
                                       'Wei':self.Wei.grad,'Wix':self.Wix.grad,'alpha':self.alpha.grad})



class FisherDenseNet(DenseNet):
    """Extension of DenseNet to estimate the diagonal of the Fisher information matrix"""
    def __init__(self,*args,fisher_movingavg_scaling=.5,**kwargs):
        super().__init__(*args,**kwargs)
        self.fisher_movingavg_scaling=fisher_movingavg_scaling
        self.scale_finv_by_norm_wex_finv = False
        self.NGD = True # set this false to test the normal updates

    def update(self, lr, input_batch):
        """
        Assumes grads are already populated with the standard gradient. Sets each param.grad
        to the (diagonal) NGD grad then updates parameters.
        Note that this uses a layer block diagonal fisher matrix and not the full fisher,
        and thus it isn't true NGD

        Regular update method if self.NGD=False, but this is just for testing
        """
        if not self.NGD:
            with torch.no_grad():
                super().update(lr=lr)
            return

        # samples a batch of targets according to the output distribution
        zout=self(input_batch)

        # need to get rid of nans since it breaks sampler with a cryptic error, may want assertion instead
        if torch.isnan(zout).any() or torch.isinf(zout).any():
            print('Warning: nan or inf in model logit while trying to sample for fisher calculation')
            zout[torch.isnan(zout)]=0.
            zout[torch.isinf(zout)]=0.

        probs=torch.nn.functional.softmax(zout,dim=-1)
        distribution=torch.distributions.categorical.Categorical(probs)
        sampled_targets=distribution.sample()

        # saves regular gradient per layer as a vectorized tensor
        for k, l in self.layers.items():
            l.saved_gradvec = l.vectorize_grad()

        # populates gradient with cross-entropy on the sampled_targets
        self.zero_grad()
        crossent=torch.nn.functional.cross_entropy(zout,sampled_targets)
        crossent.backward()

        # per layer: steps fisher matrix, inverts it, calculates natural grad for the layer,
        # un_vectorize natural grad, populates .grad, and then calls update for the layer
        with torch.no_grad():
            for k, l in self.layers.items():

                l.fisher_step(c=self.fisher_movingavg_scaling)

                F=l.F
                Finv = 1/(F+l.epsilon)

                if self.scale_finv_by_norm_wex_finv:
                    wex_finv = FisherCalculator.un_vectorize(Finv,l)['Wex']
                    Finv = Finv/ wex_finv.norm()


                naturalgrad = Finv * l.saved_gradvec
                l.saved_gradvec = None
                graddict = FisherCalculator.un_vectorize(naturalgrad,l)
                for k,v in graddict.items():
                    if torch.isnan(v).any() or torch.isinf(v).any():
                        print('Warning: nan or inf in grads after fisher calculation')
                        # in case of nan of inf sets them to zero
                        v[torch.isnan(v)] = 0.
                        v[torch.isinf(v)] = 0.
                    setattr(getattr(l,k),'grad',v.to(device=getattr(l,k).grad.device).clone()) # need to clone because grads can't be made with .view

                l.update(lr=lr)

class FisherCalculator():
    '''
    Here we provide methods for calculating the diagonal Fisher matrix in a layer.
    '''
    @torch.no_grad()
    def calc_grad(self, layer):
        """Here we assume yhat has been sampled and passed back through the model
        """
        l = layer
        dldz = l.z.grad
        dldz = torch.unsqueeze(dldz, dim=-1)
        # dldz.shape is now [batch_size, ne, 1]

        # bias
        dldb = dldz
        # dldb.shape is [batch, ne, 1]

        # g
        dzdg = 1/l.gamma * (l.ze - l.Wei@l.zi)
        dldg = (dzdg.t()*dldz.squeeze(dim=-1))
        # dzdg.shape [ne, batch]
        # dldg.shape [batch,ne]

        # Wee.grad
        g_gamma = l.g/l.gamma
        # g_gamma shape ne x batch
        dldze=g_gamma.t().unsqueeze(dim=-1)*dldz
        dldwee = torch.bmm(dldze,l.x.unsqueeze(dim=1))
        # shapes of g_gamma,dldze,dldwee are [ne,batch],[batch,ne,1],[batch,ne,n_input] respectively

#         checked grads up to here for multiple ints
        # Wei.grad
        dzdwei =-(l.g*l.ze)/ (l.Wei**2 * l.exp_alpha.t()*l.zi) # def wrong grads for multiple
        dldwei = dldz*dzdwei.t().unsqueeze(-1)

        # dzdwei.shape [ne,batch], dldwei.shape [batch,*l.Wei.shape]

        # Inhib grads
        dldzi = l.zi.grad.t()
        # dldzi.shape -> torch.Size([batch,1])

        # wie
        dzidwie = l.x
        dldwie = (dldzi * dzidwie)
        # dldwie.shape [batch,n_input]

        # alpha
        dzdalpha = -(l.g/l.gamma)* (l.ze - l.Wei@ l.zi)
        dzdalpha =dzdalpha.t().unsqueeze(1)
        dldalpha = torch.bmm(dzdalpha, dldz).squeeze(-1)
        # dldalpha.shape [batch,1]

        # stack up into a vector here
        batchgrad = self.vectorize({'b':dldb,'g':dldg,'Wex':dldwee,'Wei':dldwei,
                                    'Wix':dldwie,'alpha':dldalpha},batched=True)
        # check grads are computed correctly
        # self.gradcheck(layer,{'b':dldb,'g':dldg,'Wex':dldwee,'Wei':dldwei,
        #                             'Wix':dldwie,'alpha':dldalpha})
        return batchgrad

    def gradcheck(self,layer,batchgraddict,rtol=1e-05,atol=1e-06):
        for k, p in layer.named_parameters():
            autograd= p.grad
            if len(list(autograd.shape))==0:
                autograd=autograd.unsqueeze(0)
            handgrad=batchgraddict[k].sum(dim=0).view(*list(autograd.shape))
            assert torch.allclose(handgrad, autograd,rtol=rtol,atol=atol)

    @torch.no_grad()
    def calc_fisher(self,layer):
        grad = self.calc_grad(layer)

        # we correct the gradient so that regular grad is caculated by mean(dim=0) and not sum(dim=0),
        # which means we have computed the correct loss for each batch element and it does
        # not depend on batch size
        batch_size = grad.shape[0]
        grad = grad * batch_size

        # diagonal Fisher
        F = grad**2
        F = F * layer.f_lambda # multiply Fisher learning rate
        F = F.mean(dim=0)
        return F

    @staticmethod
    @torch.no_grad()
    def vectorize(layerparams_valuedict,batched=False):
        """vectorizes a dict of values named by layer parameters and returns the
        vectorized values tensor (shape [dim,1] or [batch,dim,1]).
        If batched=True the values supplied for each parameter are considered batched
        (in the first dimension).

        NOTE: the keys of values must correspond to the following params,
        and are returned vectorized in the order corresponding to:
        'b','g','Wex','Wei','Wix', 'alpha'

        ALSO: assumes n_input,ne > 1 and batch_size > 1 if batched
        """
        pdict={}
        for k,v in layerparams_valuedict.items():
        # all params are now mats
            if batched:
                #v=v.reshape(-1,np.prod(v.shape[-1]*v.shape[-2])
                v=v.reshape(-1,np.prod(v.shape[1:]))
            else:
                v=v.reshape(v.shape[-1]*v.shape[-2])
            pdict[k]=v
        to_cat=[pdict['b'],pdict['g'],pdict['Wex'],pdict['Wei'],pdict['Wix'],pdict['alpha']]
        #print([p.shape for p in to_cat])
        return torch.cat(to_cat,dim=-1).unsqueeze(dim=-1)

    @staticmethod
    @torch.no_grad()
    def un_vectorize(tensor,layer,batched=False):
        """returns a param-named value dict of values named by layer parameters in 'layer' from
        a tensor returned by the vectorize method (essentially inverts the vectorize method
        for the layer 'layer')
        If batched=True the values supplied for each parameter are considered batched
        (in the first dimension).

        NOTE: the keys of values are:
        'b','g','Wex','Wei','Wix', 'alpha'

        ALSO: assumes n_input,ne > 1 and batch_size > 1 if batched
        """

        v=tensor.squeeze()
        paramshapelist=[list(layer.b.shape),list(layer.g.shape),list(layer.Wex.shape),
                       list(layer.Wei.shape),list(layer.Wix.shape),list(layer.alpha.shape)]
        paramlengthlist=[int(np.prod(s)) for s in paramshapelist]
        params = torch.split(v,paramlengthlist,dim=-1)

        pshaped=[]
        for i in range(len(params)):
            p=params[i]
            if batched:
                pshaped += [p.reshape(-1,*paramshapelist[i])]
            else:
                pshaped += [p.reshape(*paramshapelist[i])]

        pdict={'b':pshaped[0],'g':pshaped[1],'Wex':pshaped[2],'Wei':pshaped[3],
              'Wix':pshaped[4],'alpha':pshaped[5]}

        return pdict