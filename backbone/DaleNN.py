from backbone.dale_nn.base_models import DenseLayer, DenseNet, build_densenet
from backbone.dale_nn.base_models import LayerNormLayer, BatchNormLayer
from backbone.dale_nn.ei_models import (
    EiDense,
    EiDenseWithShunt,
    DalesANN_SGD_UpdatePolicy,
    DalesANN_cSGD_UpdatePolicy,
    EiDense_MultipleInt_LayerNorm_WeightInitPolicy,
    EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy
)


# -------------------------------------------
#   Build model based on flags
# -------------------------------------------
def build_dale_nn(input_dim: int = 64, output_dim: int = 10, n_e: int = 500, n_i: int = 50, n_hidden: int = 3,
                  c_sgd: bool = True, i_iid_i: bool=False, layer_type='DANN',
                  ):
    """
    Instantiates a DenseNet with Inhibition.
    :param output_dim: number of output units
    :param input_dim: number of input input units
    :param n_e: number of excitatory units in a layer
    :param n_i: number of inhibitory units in a layer
    :param n_hidden: number of hidden layers
    :param c_sgd: correct sgd
    :param i_iid_i: i_iid_i
    :return: ResNet network
    """
    if layer_type == 'MLP':
        LayerClass = DenseLayer
    elif layer_type == 'LayerNorm':
        LayerClass = LayerNormLayer
    elif layer_type == 'BatchNorm':
        LayerClass = BatchNormLayer
    elif layer_type == 'DANN':
        LayerClass = EiDenseWithShunt  # DANN model
    elif layer_type == 'DANNwoShunt':
        LayerClass = EiDense
    else:
        print('Layer type not recognised!')
        raise

    if layer_type in ['MLP', 'LayerNorm', 'BatchNorm']:
        if n_i > 0:
            print("WARNING: LayerClass is not EiShunt, but n_i is not 0. Setting n_i to zero")
            n_i = 0

        hidden_dims = n_e
    else:
        hidden_dims = (n_e, n_i)
        output_dim = (output_dim, 1)  # always one output inhib (10%), even if multiple hidden inhib

    layer_dims = [input_dim] + [hidden_dims] * n_hidden + [output_dim]

    #  Build model:
    # ---------------------------------------------------
    model = build_densenet(DenseNet, LayerClass, layer_dims)

    if layer_type == 'DANN':
        for i, (key, layer) in enumerate(model.layers.items()):
            if c_sgd:
                layer.update_policy = DalesANN_cSGD_UpdatePolicy()  # (cSGD_Mixin, DalesANN_SGD_UpdatePolicy)
            else:
                layer.update_policy = DalesANN_SGD_UpdatePolicy()  # (DalesANN_SGD_UpdatePolicy)

            layer.weight_init_policy = EiDenseWithShunt_MultipleInt_LayerNorm_WeightInitPolicy(inhib_iid_init=i_iid_i)

    if layer_type == 'DANNwoShunt':
        for i, (key, layer) in enumerate(model.layers.items()):
            if c_sgd:
                layer.update_policy = DalesANN_cSGD_UpdatePolicy()  # (cSGD_Mixin, DalesANN_SGD_UpdatePolicy)
            else:
                layer.update_policy = DalesANN_SGD_UpdatePolicy()  # (DalesANN_SGD_UpdatePolicy)

            layer.weight_init_policy = EiDense_MultipleInt_LayerNorm_WeightInitPolicy(inhib_iid_init=i_iid_i)

    return model
