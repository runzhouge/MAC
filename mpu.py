from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.nn import dropout as drop
from cnn import conv_layer as conv
from cnn import conv_relu_layer as conv_relu
from cnn import pooling_layer as pool
from cnn import fc_layer as fc
from cnn import fc_relu_layer as fc_relu


def vs_multilayer(input_batch,name,middle_layer_dim=1000,reuse=False):
    with tf.variable_scope(name):
        if reuse==True:
            print name+" reuse variables"
            tf.get_variable_scope().reuse_variables()
        else:
            print name+" doesn't reuse variables"

        layer1 = conv_relu('layer1', input_batch,
                        kernel_size=1,stride=1,output_dim=middle_layer_dim)
        sim_score = conv('layer2', layer1,
                        kernel_size=1,stride=1,output_dim=3)
    return sim_score
