# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:09:49 2020

@author: Mustafa Hajij
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.python.ops import math_ops

import utilities as ut

import os


os.system('cls')



def symmetric_group_rep_net(R_op,input_shape=1):
        
    """  
    Purpose:
    --------    
        This is an auxiliary network that is trained to force the relations on the input generators.
        The network returns the sides of the relations of this algebraic structure.
    Parameters:
    -----------    
        R_op : a Keras model R^n->R^n represent a transposition element in symmetric group.
    
        input_shape : dimension of the rep.
    Returns:
    --------
        Keras model, an auxiliary network used to fornc the relations on the generator of the symmetric group     
    
    """
    
    input_tensor_1=Input(shape=(input_shape,))      
    input_tensor_2=Input(shape=(input_shape,))   
    input_tensor_3=Input(shape=(input_shape,)) 


    #R_3 first side

    outs=ut.tensor_generator_with_identity(R_op, [input_tensor_1,input_tensor_2,input_tensor_3], gen_position=1, total_dimension=3,input_dim=input_shape )  
    
    outs=ut.tensor_generator_with_identity(R_op,outs,gen_position=2, total_dimension=3,input_dim=input_shape) 
    
    outs_side1_equation_1=ut.tensor_generator_with_identity(R_op,outs,gen_position=1, total_dimension=3,input_dim=input_shape)
       

    #R_3 second side


    outs=ut.tensor_generator_with_identity(R_op, [input_tensor_1,input_tensor_2,input_tensor_3], gen_position=2, total_dimension=3,input_dim=input_shape )  
    
    outs=ut.tensor_generator_with_identity(R_op,outs,gen_position=1, total_dimension=3,input_dim=input_shape)     
    
    outs_side2_equation_1=ut.tensor_generator_with_identity(R_op,outs,gen_position=2, total_dimension=3,input_dim=input_shape)   
    
    # R_2 first side

    conc=Concatenate(name="conc_input")([input_tensor_1,input_tensor_2]) 

    R_output=R_op(conc) 
    
    side_1_equation_1=R_op(R_output)
    
    
    output_tensor=Concatenate()([Concatenate()(outs_side1_equation_1),Concatenate()(outs_side2_equation_1),side_1_equation_1 ])

    
    M=Model(inputs=[input_tensor_1,input_tensor_2,input_tensor_3],outputs=output_tensor) 
    
    return M


def symmetric_group_rep_loss(input_dim=1):
    
    def loss(y_true,y_pred):
        
        equation_1_out=tf.slice(y_pred,[0,0],[-1,3*input_dim])
        equation_2_out=tf.slice(y_pred,[0,3*input_dim],[-1,3*input_dim])
        
        final_R_2_out_1=tf.slice(y_pred,[0,6*input_dim],[-1,2*input_dim])
        
        yang_baxter_relation=K.mean(math_ops.square(equation_1_out - equation_2_out), axis=-1) # YangBaxter relation
            
        r_2_relation=K.mean(math_ops.square(y_true-final_R_2_out_1), axis=-1) # R2 move relation  
        
        return r_2_relation+yang_baxter_relation
    
    return loss

