# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:09:49 2020

@author: Mustafa Hajij
"""

import numpy as np
from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.python.ops import math_ops
import tensorflow as tf
import os
os.system('cls')



def ZxZ_group_rep_net(a_op,b_op,input_shape=2):
    
    """
    ZxZ=<a,b| ab=ba >
    
    a_op : a network R^n->R^n represent the generator a in the presentation above.
    b_op : a network R^n->R^n represent the generator b in the presentation above.
    
    input_shape : dimension of the rep.
    
    purpose : this is an axiluray network that is trained to force the relations on the input generators.
    
    the network returns the sides of the relations of this algebraic structure.
    
    """
        
    
    input_tensor_1=Input(shape=(input_shape,))
        
    # a*b side 
    R_output=b_op(input_tensor_1) 
    
    side_1_equation_1=a_op(R_output)
    
    
    # b*a side
    R_output=a_op(input_tensor_1)    
    
    side_2_equation_1=b_op(R_output)

    
    output_tensor=Concatenate()([side_1_equation_1,side_2_equation_1])    
    M=Model(inputs=[input_tensor_1],outputs=output_tensor) 
    
    return M


def ZxZ_group_rep_loss(input_dim=2):
    
    def loss(y_true,y_pred):
        
        equation_1_out=tf.slice(y_pred,[0,0],[-1,input_dim])
        equation_2_out=tf.slice(y_pred,[0,input_dim],[-1,input_dim])
        
        A=K.mean(math_ops.square(equation_1_out - equation_2_out), axis=-1) 
                    
        return A
    
    return loss



