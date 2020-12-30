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

import cus_layers as cl


def braid_generator_ins_outs(braid_generator_network, inputs, gen_position=2, total_dimension=3,input_dim=2 ):

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
        
        x_i=cl.SliceLayer(0,2*input_dim,-1,input_dim)(con_final)
        outs.append(x_i)
    
    return outs 



def braid_group_rep_net(R_op,R_op_inv,input_shape=1):

    """
    Purpose
    -------
    
        This is an axiluray network that is trained to force the relations on the input generators.
    
        The network returns the sides of the relations of this algebraic structure.
    Parameters
    ----------
    
        R_op : A Keras model R^n->R^n represent a crossing element in braid group.
        R_op_inv : A Keras model R^n->R^n represent the inverse crossing element in braid group.
    
        input_shape : dimension of the rep.
    
    Returns
    -------
    
        Keras model, an auxiliary network used to fornc the relations on the generators of the braid group     
    
    """        
    
    input_tensor_1=Input(shape=(input_shape,))      
    input_tensor_2=Input(shape=(input_shape,))   
    input_tensor_3=Input(shape=(input_shape,)) 


    #R_3 first side

    outs=braid_generator_ins_outs(R_op, [input_tensor_1,input_tensor_2,input_tensor_3], gen_position=1, total_dimension=3,input_dim=input_shape )  
    
    outs=braid_generator_ins_outs(R_op,outs,gen_position=2, total_dimension=3,input_dim=input_shape) 
    
    outs_side1_equation_1=braid_generator_ins_outs(R_op,outs,gen_position=1, total_dimension=3,input_dim=input_shape)    

    #R_3 second side


    outs=braid_generator_ins_outs(R_op, [input_tensor_1,input_tensor_2,input_tensor_3], gen_position=2, total_dimension=3,input_dim=input_shape )  
    
    outs=braid_generator_ins_outs(R_op,outs,gen_position=1, total_dimension=3,input_dim=input_shape)     
    
    outs_side2_equation_1=braid_generator_ins_outs(R_op,outs,gen_position=2, total_dimension=3,input_dim=input_shape)
        
 
    # R_2 first side

    conc=Concatenate(name="conc_input")([input_tensor_1,input_tensor_2]) 

    R_output=R_op(conc) 
    
    side_1_equation_2=R_op_inv(R_output)
    
    # R_2 first side
   
    R_inve_output=R_op_inv(conc)   
    side_2_equation_2=R_op(R_inve_output)
 
    
    output_tensor=Concatenate()([Concatenate()(outs_side1_equation_1),Concatenate()(outs_side2_equation_1),side_1_equation_2,side_2_equation_2 ])

    
    M=Model(inputs=[input_tensor_1,input_tensor_2,input_tensor_3],outputs=output_tensor) 
    
    return M


def braid_group_rep_loss(input_dim=1):

    """
    Purpose
    -------
        
        loss for the braid group. When the loss is minimal, the braid group relations are satisfied for the generator R_op.
    Parameters
    ----------    
        input_dim, the dimension of the R_op generator for the braid group.
        
    """
    
    def loss(y_true,y_pred):
        
        equation_1_out=tf.slice(y_pred,[0,0],[-1,3*input_dim])
        equation_2_out=tf.slice(y_pred,[0,3*input_dim],[-1,3*input_dim])
        
        final_R_2_out_1=tf.slice(y_pred,[0,6*input_dim],[-1,2*input_dim])
        final_R_2_out_2=tf.slice(y_pred,[0,8*input_dim],[-1,2*input_dim])
        
        A=K.mean(math_ops.square(equation_1_out - equation_2_out), axis=-1) # YangBaxter
            
        B=K.mean(math_ops.square(y_true-final_R_2_out_1), axis=-1) # R2 moves  
        
        C=K.mean(math_ops.square(y_true-final_R_2_out_2), axis=-1) # R2 moves   
        
        return A+B+C
    
    return loss