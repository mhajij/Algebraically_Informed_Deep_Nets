"""
Created on Tue Oct 20 22:09:49 2020

@author: Mustafa Hajij
"""

from tensorflow.keras import backend as K

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate,  Dense
from tensorflow.keras.layers import Lambda
import utilities as ut

import cus_layers as cl


from tensorflow.python.ops import math_ops
import tensorflow as tf

import os
os.system('cls')





def TL_generator_ins_outs(Ugen, inputs, gen_position=2, total_dimension=3,input_dim=2 ):
    

    con_list=[]
        
    generator_input=inputs[gen_position-1:gen_position+1] 
    
    tenosor=Concatenate()(generator_input)
    
    out_gen=Ugen(tenosor)
        
    
    
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
    
    
    
    



def TL_algebra_net(Ugen,delta=2,input_dim=2):
        
    """
    input:
        
     Ugen :  a network that is trained inside this function.
     delta : a constant in the TL algebra.
     input_dim : input dimension for Ugen.
    
    return : a model that outputs the both sides of the TL relations. These tensors are later used in the loss function for training.
    
    purpose : training Ugen to satisfy the relations of the TL algebera.
            (1) U_i*U_{i-1}*U_i=U_i
            (2) U_i*U_{i+1}*U_i=U_i  
            (3) U_i^2=\delta*U_i
    
    """
    
    # define the input tensors
    input_tensor_1=Input(shape=(input_dim,))       
    input_tensor_2=Input(shape=(input_dim,))   
    input_tensor_3=Input(shape=(input_dim,)) 
    

    
    # U_i*U_{i-1}*U_i=U_i
    outs=TL_generator_ins_outs(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=2, total_dimension=3,input_dim=input_dim)   
    outs=TL_generator_ins_outs(Ugen,outs,gen_position=1, total_dimension=3,input_dim=input_dim)   
    outs_side1_equation_1=TL_generator_ins_outs(Ugen,outs,gen_position=2, total_dimension=3,input_dim=input_dim)
    
    outs_side2_equation_1=TL_generator_ins_outs(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=2, total_dimension=3,input_dim=input_dim)



    # U_i*U_{i+1}*U_i=U_i  
    outs=TL_generator_ins_outs(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=1, total_dimension=3,input_dim=input_dim)   
    outs=TL_generator_ins_outs(Ugen,outs,gen_position=2, total_dimension=3,input_dim=input_dim)   
    outs_side1_equation_2=TL_generator_ins_outs(Ugen,outs,gen_position=1, total_dimension=3,input_dim=input_dim)

    outs_side2_equation_2=TL_generator_ins_outs(Ugen,[input_tensor_1,input_tensor_2,input_tensor_3],gen_position=1, total_dimension=3,input_dim=input_dim)





    
    ## U_{i}^2=\delta*U_{i} 
    out_gen=Ugen(Concatenate()([input_tensor_1,input_tensor_2]) )

    outs_side1_equation_3=Ugen(out_gen)
       
    outs_side2_equation_3 =Lambda(lambda x: x * delta) (  Ugen(Concatenate()([input_tensor_1,input_tensor_2])))

    
    final_out123=Concatenate()([Concatenate()(outs_side1_equation_1),Concatenate()(outs_side2_equation_1),
                             Concatenate()(outs_side1_equation_2),Concatenate()(outs_side2_equation_2),
                             outs_side1_equation_3,outs_side2_equation_3])

    
    M=Model(inputs=[input_tensor_1,input_tensor_2,input_tensor_3],outputs=final_out123) 
    
    return M


def TL_loss_wrapper(input_dim=2):
    
    """
    input : input_dim, the dim of the U generator.
    
    purpose : loss for the TL algebra. When the loss is minimal, the TL relations are satisfied for the generator U.
    
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













