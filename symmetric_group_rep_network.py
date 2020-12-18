# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 22:09:49 2020

@author: Mustafa Hajij
"""

import numpy as np

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Dense
from tensorflow.keras.layers import Lambda

import cus_layers as cl


from tensorflow.python.ops import math_ops
import tensorflow as tf
import os
os.system('cls')





def Id(x):
    return x 

def identity(dim=2):
    
    inputs = Input(shape=(dim,))
    
    out = Lambda(Id, name="identity")(inputs)
    
    model=Model(inputs,out)
    
    return model


def sigma_operator(input_dim=2,activation_function='linear',bias=False):


    inputs = Input(shape=(input_dim,))

    x=Dense(input_dim+2,use_bias=bias, activation=activation_function)(inputs)

    x=Dense(input_dim+2,use_bias=bias, activation=activation_function)(inputs)

    x=Dense(100,use_bias=bias, activation=activation_function)(x)

    x=Dense(20,use_bias=bias, activation=activation_function)(x)

    x=Dense(10,use_bias=bias, activation=activation_function)(x)

  
    predictions=Dense(input_dim,use_bias=bias, activation='linear' ,name='final_output')(x)


    model = Model(inputs=inputs, outputs=predictions)
    
    return model


def symmetric_generator_ins_outs(braid_generator_network, inputs, gen_position=2, total_dimension=3,input_dim=2 ):
    

    con_list=[]
        
    generator_input=inputs[gen_position-1:gen_position+1] 
    
    tensor=Concatenate()(generator_input)
    
    out_gen=braid_generator_network(tensor)
        
    
    
    for i in range(0,gen_position-1):
        con_list.append(inputs[i])
    
    con_list.append(out_gen)
        
    for i in range(gen_position+1,total_dimension):
        
        con_list.append(inputs[i])
            
    
            
    con_final=Concatenate()(con_list)  
    
    outs=[]
        
    for i in range(0,total_dimension):
        
        x_i=cl.slice_layerA(0,2*input_dim,-1,input_dim)(con_final)
        outs.append(x_i)
    
    return outs 



def symmetric_group_rep_net(R_op,input_shape=1):
        
    
    input_tensor_1=Input(shape=(input_shape,))
        
    input_tensor_2=Input(shape=(input_shape,)) 
    
    input_tensor_3=Input(shape=(input_shape,)) 


    #R_3 first side

    outs=symmetric_generator_ins_outs(R_op, [input_tensor_1,input_tensor_2,input_tensor_3], gen_position=1, total_dimension=3,input_dim=input_shape )  
    
    outs=symmetric_generator_ins_outs(R_op,outs,gen_position=2, total_dimension=3,input_dim=input_shape) 
    
    outs_side1_equation_1=symmetric_generator_ins_outs(R_op,outs,gen_position=1, total_dimension=3,input_dim=input_shape)
    
    

    #R_3 second side


    outs=symmetric_generator_ins_outs(R_op, [input_tensor_1,input_tensor_2,input_tensor_3], gen_position=2, total_dimension=3,input_dim=input_shape )  
    
    outs=symmetric_generator_ins_outs(R_op,outs,gen_position=1, total_dimension=3,input_dim=input_shape)     
    
    outs_side2_equation_1=symmetric_generator_ins_outs(R_op,outs,gen_position=2, total_dimension=3,input_dim=input_shape)
        
    
    
    
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
        
        A=K.mean(math_ops.square(equation_1_out - equation_2_out), axis=-1) # YangBaxter
            
        B=K.mean(math_ops.square(y_true-final_R_2_out_1), axis=-1) # R2 moves  
        
        return A+B
    
    return loss

