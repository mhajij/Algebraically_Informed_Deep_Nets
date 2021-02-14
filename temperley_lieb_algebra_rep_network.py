"""
Created on Tue Oct 20 22:09:49 2020

@author: Mustafa Hajij
"""

import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.python.ops import math_ops

import utilities as ut

os.system('cls')
    
def tl_algebra_net(Ugen,delta=2,input_dim=2):
        
    """

    Purpose
    --------     
    
        training model Ugen to satisfy the relations of the TL algebera.
        (1) U_i*U_{i-1}*U_i=U_i
        (2) U_i*U_{i+1}*U_i=U_i  
        (3) U_i^2=\delta*U_i
        (4) UiUj = UjUi if |i-j|>1 (this relation is automatially satisfied so we do not include it in the model)
        
         From the paper: the presentation of the TL algebra is given by :
         TL_m=< U1,...,U_{m-1} | U_i*U_{i-1}*U_i=U_i, U_i*U_{i+1}*U_i=U_i ,U_i^2=\delta*U_i, UiUj = UjUi > 
         
         where 
         
         Ui=id^{i-1} X Ugen X id^{m-i+1}    
         hence it is enough to train Ugen for the entire rep of the TL algebra

    Parameters:
    ---------- 
        
         Ugen :  a network that is trained inside this network.
         delta : a constant in the TL algebra.
         input_dim : input dimension for Ugen.
     
    Return
    ------
         A Keras model that outputs the both sides of the TL relations. These tensors are later used in the loss function for training.
        
    
         
    """
    
    # define the input tensors
    input_tensor_1=Input(shape=(input_dim,))       
    input_tensor_2=Input(shape=(input_dim,))   
    input_tensor_3=Input(shape=(input_dim,)) 
    
    
    # relation : U_i*U_{i-1}*U_i=U_i
    #side1 
    outs=ut.tensor_generator_with_identity(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=2, total_dimension=3,input_dim=input_dim)   
    outs=ut.tensor_generator_with_identity(Ugen,outs,gen_position=1, total_dimension=3,input_dim=input_dim)   
    outs_side1_equation_1=ut.tensor_generator_with_identity(Ugen,outs,gen_position=2, total_dimension=3,input_dim=input_dim)
    #side2    
    outs_side2_equation_1=ut.tensor_generator_with_identity(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=2, total_dimension=3,input_dim=input_dim)


    # relation : U_i*U_{i+1}*U_i=U_i  
    #side1 
    outs=ut.tensor_generator_with_identity(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=1, total_dimension=3,input_dim=input_dim)   
    outs=ut.tensor_generator_with_identity(Ugen,outs,gen_position=2, total_dimension=3,input_dim=input_dim)   
    outs_side1_equation_2=ut.tensor_generator_with_identity(Ugen,outs,gen_position=1, total_dimension=3,input_dim=input_dim)
    #side2
    outs_side2_equation_2=ut.tensor_generator_with_identity(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=1, total_dimension=3,input_dim=input_dim)
   
    # relation : U_{i}^2=\delta*U_{i} 
    out_gen=Ugen(Concatenate()([input_tensor_1,input_tensor_2]) ) # requires two inputs
    #side1 
    outs_side1_equation_3=Ugen(out_gen)
    #side2       
    outs_side2_equation_3 =Lambda(lambda x: x * delta) (  Ugen(Concatenate()([input_tensor_1,input_tensor_2])))

    
    final_out123=Concatenate()([Concatenate()(outs_side1_equation_1),Concatenate()(outs_side2_equation_1),
                             Concatenate()(outs_side1_equation_2),Concatenate()(outs_side2_equation_2),
                             outs_side1_equation_3,outs_side2_equation_3])

    
    M=Model(inputs=[input_tensor_1,input_tensor_2,input_tensor_3],outputs=final_out123) 
    
    return M


def tl_loss_wrapper(input_dim=2):
    
    """
    Purpose
    -------
        
        loss for the TL algebra. When the loss is minimal, the TL relations are satisfied for the generator U.


    Parameters
    ----------    
        input_dim, the dimension of the U generator for the TL algebra.
        
    """
    
    def loss(y_true,y_pred):
           
        outs_side1_equation_1=tf.slice(y_pred,[0,0],[-1,3*input_dim])  # three strands each one of them is a tensor of dim input_dim 
        outs_side2_equation_1=tf.slice(y_pred,[0,3*input_dim],[-1,3*input_dim])  # three strands each one of them is a tensor of dim input_dim 
        
        outs_side1_equation_2=tf.slice(y_pred,[0,6*input_dim],[-1,3*input_dim])  # three strands each one of them is a tensor of dim input_dim 
        outs_side2_equation_2=tf.slice(y_pred,[0,9*input_dim],[-1,3*input_dim])  # three strands each one of them is a tensor of dim input_dim 
    
        outs_side1_equation_3=tf.slice(y_pred,[0,12*input_dim],[-1,2*input_dim])  # two strands each one of them is a tensor of dim input_dim 
        outs_side2_equation_3=tf.slice(y_pred,[0,14*input_dim],[-1,2*input_dim]) # two strands each one of them is a tensor of dim input_dim 
        
        A=K.mean(math_ops.square(outs_side1_equation_1 - outs_side2_equation_1), axis=-1) 
        
        B=K.mean(math_ops.square(outs_side1_equation_2 - outs_side2_equation_2), axis=-1) 
    
        C=K.mean(math_ops.square(outs_side1_equation_3 - outs_side2_equation_3), axis=-1) 
    
       
        return A+B+C
    return loss
