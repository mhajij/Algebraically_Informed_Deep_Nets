# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 12:01:54 2020

@author: Mustafa Hajij
"""
import Temperley_Lieb_algebra_rep_network as tlnet 
import braid_group_rep_network as bgnet
import symmetric_group_rep_network as symgnet
import ZxZ_group_rep_network as ZSnet
import utilities as ut

import numpy as np
import argparse

def model_string_gen(model_name):
    
    name=model_name+str(args.bias)+"_activation="+  str(args.network_generator_activation)+"_"+str(args.generator_dimension)+"_to_"+str(args.generator_dimension)+"_delta="+str(args.delta)+"_.h5"
    
    return name
    
weight_folder='weights/'   
 
data_folder='data/'    

    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # testing or training argument

    parser.add_argument('-m', '--mode',type=str,default='training',required=True,help='Specify if you want to train a network or testing it.')   

    # type of structure, options : TL, braid_group, symmetric_group, ZxZ

    parser.add_argument('-st', '--structure',type=str,default="TL",required=True,help='Which structure you want to train. Options are : "TL" , "braid_group", "ZxZ_group" and "symmetric_group". Any other option will generate an error.')
    
    
    #type, dim of network arguments : linear, affine, nonlinear. Activation determines the type.
    
    #(1) dim of the rep
    parser.add_argument('-dim', '--generator_dimension',type=int,default=2,help='domain dimension of the gen network.')

    #(2) determines the type of the of the rep
  
    parser.add_argument('-a', '--network_generator_activation',type=str,default='linear',required=False, help='Activation of the type. Options are : linear, tanh. Uniformly chosen for all layers when linear activation are chosen and sigmoid otherwise..')
   
    #(3) determines the type of the of the rep
    
    parser.add_argument('-bias', '--bias',type=str,default=False,required=False, help='Use bias in your network.')
   
    # TL algebra argument : delta.
    #* determines delta for the TL algebra. Not effective when training a group.
    
    
    parser.add_argument('-delta', '--delta',type=float,required=False,default=1,help='delta, the TL parameter.')

    #__________________________________________________________

    # training arguments
    #(1) number of epochs
    
    parser.add_argument('-e', '--epoch',type=int,default=2,help='Number of epochs.')
    
    #(2) learning rate
    
    parser.add_argument('-lr', '--learning_rate',type=float,required=False,default=0.002,help='learning rate.')

    #(3) batchsize

    parser.add_argument('-b', '--batch_size',type=float,required=False,default=2000,help='batch size')

    #___________________________________________________________


    args = parser.parse_args()

    dim=args.generator_dimension
    d=dim//2

    if dim in [2,4,6] and args.structure in ["TL",'braid_group','symmetric_group']:
        
        data1=np.load(data_folder+str(d)+'d_data_1.npy')
        data2=np.load(data_folder+str(d)+'d_data_2.npy')
        data3=np.load(data_folder+str(d)+'d_data_3.npy')

    elif dim in list(range(2,11)) and args.structure=="ZxZ_group" :
        
        
        data1=np.load(data_folder+str(args.generator_dimension)+'d_data.npy')
    
    else:        
        
        raise ValueError("generator dimension " +str(dim) +"is not in the constrained dimensions for this structure: dimension must be in [2,4,6] for the TL,'braid_group','symmetric_group' and between (2,10) for ZxZ_group. ")


    if args.structure=='TL':
            
            
        if args.delta==0:
            
            raise ValueError("delta, a parameter for TL algebra, must be nonzero.")


        print("training the TL algebra generator.")
      
        print("generator function : R^"+str(args.generator_dimension) +"-> R^"+str(args.generator_dimension) ) 
     
        Ugen=ut.operator(input_dim=args.generator_dimension,bias= args.bias, activation_function=args.network_generator_activation)
        

        M=tlnet.TL_algebra_net(Ugen,delta =args.delta  ,input_dim=dim//2)
        
        
        model_name=model_string_gen("TL_algebra_relations_trainer_use_bias=")
        model_name_U_gen=model_string_gen("TL_algebra_generator_use_bias=")
        
        data_in=[data1,data2,data3]
        data_out=data1
        
        
        if args.mode=='training':
            
            print("choosing the training mode. ")   
            
            
            ut.train_net(M,data_in,data_out, weight_folder+model_name ,tlnet.TL_loss_wrapper(dim//2),args.learning_rate,args.batch_size,args.epoch)
   
            print("saving the model.." )

            
            Ugen.save(weight_folder+model_name_U_gen) 
            print("model saved.")

        elif args.mode=='testing':
            
          
            relations_tensor=ut.get_relation_tensor(weight_folder+model_name,M,data_in)
            

            print("testing the relations..")

            
            print("testing the relation:  U_i*U_{i-1}*U_i=U_i")
            
            print(np.linalg.norm(relations_tensor[:,0:3*d]-relations_tensor[:,3*d:6*d]))
            
            print("testing the relation:  U_i*U_{i+1}*U_i=U_i")
            
            print(np.linalg.norm(relations_tensor[:,6*d:9*d]-relations_tensor[:,9*d:12*d]))

            print("testing the relation: U_{i}^2=\delta*U_{i} ")
            
            print(np.linalg.norm(relations_tensor[:,12*d:14*d]-relations_tensor[:,14*d:]))

        else:
            raise ValueError("Mode options are either training or testing.")
                          
            
    elif args.structure=='braid_group': 
        
                      
        
        print("training the braid group generators")
                 
        R_oP1,R_oP2=ut.get_n_operators(dim=args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias,n_of_operators=2 )
        
        M=bgnet.braid_group_rep_net(R_oP1,R_oP2,input_shape=dim//2)

        model_name1=model_string_gen("braid_group_sigma_generator_use_bias=")
        model_name2=model_string_gen("braid_group_sigma_inverse_generator_use_bias=")
        model_name=model_string_gen("braid_group_relations_trainer_use_bias=")
        
        data_in=[data1,data2,data3]
        
        data_out=np.hstack([data1,data2])       
        if args.mode=='training':
            
            print("choosing the training mode. ")  

            
            ut.train_net(M,data_in,data_out,weight_folder+model_name ,bgnet.braid_group_rep_loss(dim//2),args.learning_rate,args.batch_size,args.epoch)
            
 
            print("saving the models.." )
                        
            R_oP1.save(weight_folder+model_name1) 
            R_oP2.save(weight_folder+model_name2) 
 
            print("model saved.")
            
            
        elif args.mode=='testing':

            relations_tensor=ut.get_relation_tensor(weight_folder+model_name,M,data_in)


            print("testing the braid group generators")


            print("testing the relation:  R3 ")

            print(np.linalg.norm(relations_tensor[:,0:3*d]-relations_tensor[:,3*d:6*d]))
            
            print("testing the relation:  R2 ")
            
            print(np.linalg.norm(relations_tensor[:,6*d:8*d]-data_out))

            print("testing the relation:  R2 ")
            
            print(np.linalg.norm(relations_tensor[:,8*d:]-data_out))
        else:
            raise ValueError("Mode options are either training or testing.")
     
        
    elif args.structure=='symmetric_group': 
                      
        print("training the symmetric_group generator")        

     
        R_oP1=ut.operator(input_dim=args.generator_dimension,activation_function=args.network_generator_activation ,bias=args.bias)
        
    
        M=symgnet.symmetric_group_rep_net(R_oP1,input_shape=dim//2)

        model_name=model_string_gen("symmetric_group_relations_trainer_use_bias=")
        model_name_generator=model_string_gen("symmetric_group_generator_use_bias=")
        
        data_in=[data1,data2,data3]
        data_out=np.hstack([data1,data2])
        
        if args.mode=='training':
            
            print("choosing the training mode. ")  
 
       
            ut.train_net(M,data_in, data_out, weight_folder+model_name ,symgnet.symmetric_group_rep_loss(dim//2),args.learning_rate,args.batch_size,args.epoch)
 
            print("saving the model..")
            
                            
            R_oP1.save(weight_folder+model_name_generator) 
            
                        
            print("model saved.")
            
        elif args.mode=='testing':
            
            relations_tensor=ut.get_relation_tensor(weight_folder+model_name,M,data_in)

            print("testing the relation:  R3 ")

            print(np.linalg.norm(relations_tensor[:,0:3*d]-relations_tensor[:,3*d:6*d]))
            
            print("testing the relation:  R2 ")
            
            print(np.linalg.norm(relations_tensor[:,6*d:]-data_out))
        else:
            raise ValueError("Mode options are either training or testing.")
                          
    elif args.structure=='ZxZ_group': 
        
        A_oP,B_oP=ut.get_n_operators(dim=args.generator_dimension,activation_function=args.network_generator_activation,bias=args.bias,n_of_operators=2 )      

        M=ZSnet.ZxZ_group_rep_net(A_oP,B_oP,input_shape=dim)


        model_name=model_string_gen("ZxZ_group_relations_trainer_use_bias=")    
        
        aname=model_string_gen("ZxZ_group_a_generator_use_bias=")
        
        bname=model_string_gen("ZxZ_group_b_generator_use_bias=")
        
        data_in=data1
        data_out=data1

        if args.mode=='training':
               
            
            print("choosing the training mode. ")  
      
            ut.train_net(M,data_in,data_out,weight_folder+model_name ,ZSnet.ZxZ_group_rep_loss(dim),args.learning_rate,args.batch_size,args.epoch)
   

            print("saving the generator operator to file : " + aname )

            A_oP.save(weight_folder+aname) 
            print("saving the generator operator to file : " + bname )
            
            B_oP.save(weight_folder+bname) 
 
            print("model saved.")
            

        elif args.mode=='testing':  
            relations_tensor=ut.get_relation_tensor(weight_folder+model_name,M,data_in)
           
            print("testing the relation:  ")

            print( np.linalg.norm( relations_tensor[:,:dim][:100]-relations_tensor[:,dim:][:100]))
        else:
            raise ValueError("Mode options are either training or testing.")           
    else:     
        raise ValueError("structures must be : TL, braid_group, symmetric_group, or ZxZ_group")

      