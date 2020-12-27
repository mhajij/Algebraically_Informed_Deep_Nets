
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:01:54 2020

@author: Mustafa Hajij
"""
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.python.ops import math_ops
import tensorflow as tf



class RTLayer(tf.keras.layers.Layer):
    def __init__(self,eye_size=2,loop_constant=2):
        super(RTLayer, self).__init__()
        self.eyesize=eye_size
        self.loop_constant=loop_constant
      
    def build(self, input_shape):
        
        
        """
        # deep nets can be utilized to built R matrices. However, results are not much better in this case than shallow nets.
        
        self.w_1 = self.add_weight(
            shape=(self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )

        self.w_2 = self.add_weight(
            shape=(10*self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )

        self.w_3 = self.add_weight(
            shape=(10*self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )
        self.w_4 = self.add_weight(
            shape=(10*self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )

        self.w_5 = self.add_weight(
            shape=(10*self.eyesize**2, self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )
        
        w1=tf.matmul(self.w_4,self.w_5)
        w2=tf.matmul(self.w_3,w1)
        w3=tf.matmul(self.w_2,w2)
        w4=tf.matmul(self.w_1,w3)
        
        self.w=w4
        """
        self.w = self.add_weight(
            shape=(self.eyesize**2, self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )
        """
        self.w_1_ = self.add_weight(
            shape=(self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )

        self.w_2_ = self.add_weight(
            shape=(10*self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )

        self.w_3_ = self.add_weight(
            shape=(10*self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )
        self.w_4_ = self.add_weight(
            shape=(10*self.eyesize**2, 10*self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )

        self.w_5_ = self.add_weight(
            shape=(10*self.eyesize**2, self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )        
        
        w1_=tf.matmul(self.w_4_,self.w_5_)
        w2_=tf.matmul(self.w_3_,w1_)
        w3_=tf.matmul(self.w_2_,w2_)
        w4_=tf.matmul(self.w_1_,w3_)
        
        self.w_invserse=w4_       
        """
        self.w_invserse = self.add_weight(
            shape=(self.eyesize**2, self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )  

        self.n = self.add_weight(
            shape=( 1,self.eyesize**2),
            initializer="random_normal",
            trainable=True,
        )
              
        self.u =self.add_weight(
            shape=( self.eyesize**2,1),
            initializer="random_normal",
            trainable=True,
        )
        

        self.eye=tf.diag(tf.ones(self.eyesize) )
        
        
        
        
        # R3 tensors
        #_________________
        self.RxId=tf.tensordot( self.w, self.eye, axes=0)
      
        self.IdxR=tf.tensordot( self.eye, self.w, axes=0)

        size=self.eyesize*self.eyesize*self.eyesize   

        self.RxId=tf.transpose(self.RxId,(0,2,1,3))
        
        self.RxId_final_tensor=tf.reshape(self.RxId, shape=(size,size) )
        
        self.IdxR=tf.transpose(self.IdxR,(0,2,1,3))
        
        self.IdxR_final_tensor=tf.reshape(self.IdxR, shape=(size,size) )
        #_________________
        
        # R2 tensors
        self.RinvxId=tf.tensordot( self.w_invserse, self.eye, axes=0)
        self.RinvxId=tf.transpose(self.RinvxId,(0,2,1,3))
        self.RinvxId_final_tensor=tf.reshape(self.RinvxId, shape=(size,size) )
        
        
        self.IdxRinv=tf.tensordot( self.eye, self.w_invserse, axes=0) 
        self.IdxRinv=tf.transpose(self.IdxRinv,(0,2,1,3))
        self.IdxRinv_final_tensor=tf.reshape(self.IdxRinv, shape=(size,size) )
        
         #_______________________________
 
    

        # T move for the n operator
        self.nxId=tf.tensordot( self.n, self.eye, axes=0)
        
        self.nxId=tf.transpose(self.nxId,(0,2,1,3))
        
        self.nxId_final_tensor=tf.reshape(self.nxId, shape=(size,self.eyesize) )
        
        
        # T_s move  for the n operator
        self.Idxn=tf.tensordot( self.eye, self.n, axes=0) 
        
        self.Idxn=tf.transpose(self.Idxn,(0,2,1,3))
        
        self.Idxn_final_tensor=tf.reshape(self.Idxn, shape=(size,self.eyesize) )
        
        

        ##  T move  for the u operator

        
        self.uxId=tf.tensordot( self.u, self.eye, axes=0)
        
        self.uxId=tf.transpose(self.uxId,(0,2,1,3))
        
        self.uxId_final_tensor=tf.reshape(self.uxId, shape=(self.eyesize,size) )
        
        
        # T_s move
        self.Idxu=tf.tensordot( self.eye, self.u, axes=0) 
        
        self.Idxu=tf.transpose(self.Idxu,(0,2,1,3))
        
        self.Idxu_final_tensor=tf.reshape(self.Idxu, shape=(self.eyesize,size) )

        self.u_reshaped=tf.reshape(self.u, shape=(self.eyesize,self.eyesize) )
        self.n_reshaped=tf.reshape(self.n, shape=(self.eyesize,self.eyesize) )
        

    def call(self, inputs):
        
        #R3 move computations
        #first side
        x1=tf.matmul(inputs, self.RxId_final_tensor)
      
        x2=tf.matmul(x1, self.IdxR_final_tensor)
      
        x3=tf.matmul(x2, self.RxId_final_tensor)

        #second side
        y1=tf.matmul(inputs, self.IdxR_final_tensor)
      
        y2=tf.matmul(y1, self.RxId_final_tensor)
      
        y3=tf.matmul(y2, self.IdxR_final_tensor)
        
        
        # R2 move first side
        
        inputs_to_r2=inputs[:,:self.eyesize*self.eyesize]
        
        z1=tf.matmul(inputs_to_r2, self.w)

        # R2 move second side

        
        side_1_r2=tf.matmul(z1, self.w_invserse)
        

        w1=tf.matmul(inputs_to_r2, self.w_invserse)
        
        side_2_r2=tf.matmul(w1, self.w)
        
        # R1 first_side
        
        inputs_to_r1=inputs[:,:1]
        
        z1=tf.matmul(inputs_to_r1, self.n)
        
        side_1_r1=tf.matmul(z1, self.w)
        
        # R1 second_side
        
        side_2_r1=tf.matmul(inputs_to_r1, self.n)
        

        # R1 for u operator first side

        
        k1=tf.matmul(inputs_to_r2, self.w)
        
        side_1_r1_u=tf.matmul(k1, self.u)
        
        # R1 for u operator second_side
        
        side_2_r1_u=tf.matmul(inputs_to_r2, self.u)        




        # R1 first_side Rinv
        
        inputs_to_r1=inputs[:,:1]
        
        z1=tf.matmul(inputs_to_r1, self.n)
        
        side_1_r1_Rinv=tf.matmul(z1, self.w_invserse)
        
        # R1 second_side Rinv
        
        side_2_r1_Rinv=tf.matmul(inputs_to_r1, self.n)
        

        # R1 for u operator first side Rinv

        
        k1=tf.matmul(inputs_to_r2, self.w_invserse)
        
        side_1_r1_u_Rinv=tf.matmul(k1, self.u)
        
        # R1 for u operator second_side Rinv
        
        side_2_r1_u_Rinv=tf.matmul(inputs_to_r2, self.u)        
        
        
        
        # normalization
        
        n=tf.matmul(inputs_to_r1, self.n)
        
        normalized_first_side=tf.matmul(n, self.u)
        
        normalized_second_side=self.loop_constant*inputs_to_r1
        
        
        #---------------

        """

        inputs_to_tm=inputs[:,:self.eyesize]

        f1=tf.matmul(inputs_to_tm, self.uxId_final_tensor)
      
        f2=tf.matmul(f1, self.IdxR_final_tensor)           


        j1=tf.matmul(inputs_to_tm, self.Idxu_final_tensor)
      
        j2=tf.matmul(j1, self.RxId_final_tensor)  



        
        


         
        ff1=tf.matmul(inputs_to_tm, self.uxId_final_tensor)
      
        ff2=tf.matmul(ff1, self.IdxRinv_final_tensor)           


        jj1=tf.matmul(inputs_to_tm, self.Idxu_final_tensor)
      
        jj2=tf.matmul(jj1, self.RinvxId_final_tensor)          


        inputs_to_tm=inputs[:,:self.eyesize]
        
        jj1=tf.matmul(inputs_to_tm, self.u_reshaped)
      
        jj2=tf.matmul(jj1, self.n_reshaped)    
        

        kk1=tf.matmul(inputs_to_tm, self.n_reshaped)
      
        kk2=tf.matmul(kk1, self.u_reshaped)       

        """

        inputs_to_tm=inputs[:,:self.eyesize]

        f1=tf.matmul(inputs_to_tm, self.Idxu_final_tensor)

        f2=tf.matmul(f1, self.nxId_final_tensor)



        ff1=tf.matmul(inputs_to_tm, self.uxId_final_tensor)

        ff2=tf.matmul(ff1, self.Idxn_final_tensor)
            
        #--------------------

     
        # tmove first side
        
        
        a1=tf.matmul(inputs, self.IdxR_final_tensor)
      
        a2=tf.matmul(a1, self.nxId_final_tensor)        
                
        # tmove second side
        
        b1=tf.matmul(inputs, self.RxId_final_tensor)
      
        b2=tf.matmul(b1, self.Idxn_final_tensor)   



        # stmove first side        
        

        c1=tf.matmul(inputs, self.IdxRinv_final_tensor)
      
        c2=tf.matmul(c1, self.nxId_final_tensor)        


        # stmove second side        
        
        d1=tf.matmul(inputs, self.RinvxId_final_tensor)
      
        d2=tf.matmul(d1, self.Idxn_final_tensor)   

              
        return tf.concat([x3,y3,side_1_r2,side_2_r2,side_1_r1,side_2_r1,a2,b2,c2,d2,side_1_r1_u,
                          side_2_r1_u,normalized_first_side,normalized_second_side,side_1_r1_Rinv,
                          side_2_r1_Rinv,side_1_r1_u_Rinv,side_2_r1_u_Rinv,f2,ff2],1)
       

def RT_training_net(id_dim=2,loop_constant=2):

    inputs = tf.keras.layers.Input(shape=(id_dim**3,))
    
    F=RTLayer(id_dim,loop_constant)  
    
    out=F(inputs)
    
    model = Model(inputs, out)
    
    return model
    

def RT_loss(id_dim=2):
    
        
    def cus_loss(y_true,y_pred):
        equation_1_out=tf.slice(y_pred,[0,0],[-1,id_dim**3])#R3
        equation_2_out=tf.slice(y_pred,[0,id_dim**3],[-1,id_dim**3])#R3
        
        final_R_2_out_1=tf.slice(y_pred,[0,2*(id_dim**3)],[-1,id_dim**2])
        final_R_2_out_2=tf.slice(y_pred,[0,2*(id_dim**3)+id_dim**2],[-1,id_dim**2])
    
        R1_s1=tf.slice(y_pred,[0,2*(id_dim**3)+id_dim**2+id_dim**2],[-1,id_dim**2])
        R1_s2=tf.slice(y_pred,[0,2*(id_dim**3)+id_dim**2+id_dim**2+id_dim**2],[-1,id_dim**2])
        
        
        dim_so_far=2*(id_dim**3)+id_dim**2+id_dim**2+id_dim**2+id_dim**2
        
        T1_s1=tf.slice(y_pred,[0,dim_so_far],[-1,id_dim])
        T1_s2=tf.slice(y_pred,[0,dim_so_far+id_dim],[-1,id_dim])
    
    
        T2_s1=tf.slice(y_pred,[0,dim_so_far+2*id_dim],[-1,id_dim])
        
        T2_s2=tf.slice(y_pred,[0,dim_so_far+3*id_dim],[-1,id_dim])
        
        
        dim_so_far_2=dim_so_far+4*id_dim
        
        R1_u_firstside=tf.slice(y_pred,[0,dim_so_far_2],[-1,1])
        
        R1_u_secondside=tf.slice(y_pred,[0,dim_so_far_2+1],[-1,1])
        
        N_1_firstside=tf.slice(y_pred,[0,dim_so_far_2+2],[-1,1])
        N_1_secondside=tf.slice(y_pred,[0,dim_so_far_2+3],[-1,1])
        
        
        
        dim_so_far_3=dim_so_far_2+4
        
        side_1_r1_Rinv=tf.slice(y_pred,[0,dim_so_far_3],[-1,id_dim**2])
        side_2_r1_Rinv=tf.slice(y_pred,[0,dim_so_far_3+id_dim**2],[-1,id_dim**2])
        side_1_r1_u_Rinv=tf.slice(y_pred,[0,dim_so_far_3+2*(id_dim**2)],[-1,1])
        side_2_r1_u_Rinv=tf.slice(y_pred,[0,dim_so_far_3+2*(id_dim**2)+1],[-1,1])
        
        
        sim_so_far_4=dim_so_far_3+2*(id_dim**2)+2
        
        
        side1_fixing=tf.slice(y_pred,[0,sim_so_far_4],[-1,id_dim])
        side2_fixing=tf.slice(y_pred,[0,sim_so_far_4+id_dim],[-1,id_dim])
        
        """
        side1_tm_u=tf.slice(y_pred,[0,sim_so_far_4],[-1,id_dim**3])
        side2_tm_u=tf.slice(y_pred,[0,sim_so_far_4+id_dim**3],[-1,id_dim**3])
        
        side1_tm_u_Rinv=tf.slice(y_pred,[0,sim_so_far_4+2*(id_dim**3)],[-1,id_dim**3])
        side2_tm_u_Rinv=tf.slice(y_pred,[0,sim_so_far_4+3*(id_dim**3)],[-1,id_dim**3])
        """
        
        B=K.mean(math_ops.square(y_true[:]-final_R_2_out_1), axis=-1) # R2 moves  
        C=K.mean(math_ops.square(y_true[:]-final_R_2_out_2), axis=-1) # R2 moves           
        
        
        
        
        
        A=K.mean(math_ops.square(equation_1_out - equation_2_out), axis=-1) # YangBaxter
            
        B=K.mean(math_ops.square(y_true-final_R_2_out_1), axis=-1) # R2 moves  
        C=K.mean(math_ops.square(y_true-final_R_2_out_2), axis=-1) # R2 moves   
        
    
        D=K.mean(math_ops.square(R1_s1-R1_s2), axis=-1) # R1 moves   
    



        G=K.mean(math_ops.square(T1_s1-T1_s2), axis=-1) # T moves   
    
       
        R1=K.mean(math_ops.square(side_1_r1_Rinv-side_2_r1_Rinv), axis=-1) # R1 moves


        fixing_1=K.mean(math_ops.square(side1_fixing-y_true[:,:id_dim] ), axis=-1) # tm for u and R

        fixing_2=K.mean(math_ops.square(side2_fixing-y_true[:,:id_dim] ), axis=-1) # tm for u and R


        #optional
        E=K.mean(math_ops.square(R1_u_firstside-R1_u_secondside), axis=-1) # R1 moves   

        KK=K.mean(math_ops.square(T2_s1-T2_s2), axis=-1) # T moves   

        R11=K.mean(math_ops.square(side_1_r1_u_Rinv-side_2_r1_u_Rinv), axis=-1) # R1 moves

        F=K.mean(math_ops.square(N_1_firstside-N_1_secondside), axis=-1) # normalization : n*u=loop_value

        #tm_u=K.mean(math_ops.square(side1_tm_u-y_true[:,:id_dim] ), axis=-1) # tm for u and R
        
        #tm_u_rinv=K.mean(math_ops.square(side2_tm_u-y_true[:,:id_dim]), axis=-1) # tm for u and R_inv

       
        return A+B+C+D+G+fixing_1+fixing_2+R1
    return cus_loss

def get_weights(model,id_size):
    
    weights=model.layers[-1].get_weights()
    R=weights[0]
    R_inv=weights[1]
    n=weights[2]
    u=weights[3]
    
    return R,R_inv,n,u

def tensor(a,b,shape):
        
    temp=np.tensordot( a, b, axes=0) 
        
    d=np.transpose(temp,(0,2,1,3))
        
    d=np.reshape(d,shape)
    
    return d

def composition(a,b):
         
    return np.dot(a,b)

