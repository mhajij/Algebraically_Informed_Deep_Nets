# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:09:49 2020

@author: Mustafa Hajij
"""

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.python.ops import math_ops
import tensorflow as tf

import os
os.system('cls')



def ZxZ_group_rep_net(a_op,b_op,input_shape=2):
    
    """
    Purpose
    -------
        This is an axiluray network that is trained to force the relations on the input generators.
    
        The network returns the sides of the relations of this algebraic structure.
        
    
    Parameters
    ----------
    
        a_op : a network R^n->R^n represent the generator a in the presentation above.
        b_op : a network R^n->R^n represent the generator b in the presentation above.
    
        input_shape : dimension of the rep.
    
    Return
    ------
        Keras model, an auxiliary network used to fornce the relations on the generators of the ZxZ group.
                     the output of this Keras model is the relations tensors which used inside a loss function 
                     in order to make the these relations hold.
      
    
    Note
    ----
        The group ZxZ has the presentation: ZxZ=<a,b| ab=ba >. In other words, it has two generators and one relation. 
    
    """
    input_tensor_1=Input(shape=(input_shape,))     
    # a*b side 
    R_output=b_op(input_tensor_1)  
    side_1_equation_1=a_op(R_output)
    
    # b*a side
    R_output=a_op(input_tensor_1)     
    side_2_equation_1=b_op(R_output)
    
    # Concatenate the two tensors and make the model
    output_tensor=Concatenate()([side_1_equation_1,side_2_equation_1])    
    M=Model(inputs=[input_tensor_1],outputs=output_tensor) 
    
    return M

def ZxZ_group_rep_loss(input_dim=2):
    
    """
    Purpose
    -------
        loss for the ZxZ group. When the loss is minimal, the relation a*b=b*a is satisfied for the generators a and b.
    
    Parameter
    ---------
    
        input_dim, the dim of the a_op and b_op generators.
        
    """
    
    def loss(y_true,y_pred):
        
        # split the output relation tensor of the model 
        equation_1_out=tf.slice(y_pred,[0,0],[-1,input_dim])
        equation_2_out=tf.slice(y_pred,[0,input_dim],[-1,input_dim])   
        
        A=K.mean(math_ops.square(equation_1_out - equation_2_out), axis=-1)                 
        return A
    
    return loss



