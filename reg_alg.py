from sklearn import datasets as ds,model_selection as ms
import matplotlib.pyplot as plt
import numpy as np
import random
import math
class Lmodel:
    def params_exp(params):
        total=0.0
        for i in range(len(params)-1):
            total +=(params[i]**2)
        return total
    def get_sgd_iter_set(x_train,y_train,sgd_size):
        total_size=len(y_train)
        t_range=list(range(0,total_size))
        random.shuffle(t_range)
        sgd_list=t_range[0:sgd_size]
        x_sgd=[]
        y_sgd=[]
        for i in sgd_list:
            x_sgd.append(x_train[i])
            y_sgd.append(y_train[i])
        return (x_sgd,y_sgd)
    def h_x(x_data,params):
        '''
        f(x)=wx+b
        '''
        total=0.0
        for i in range(len(x_data)):
            total +=x_data[i]*params[i]
        out=1/(1+math.exp(-total))
        #print("out:"+str(out))
        return out

    def logistic_loss(x_data,y_data,model_params):
        '''
        calculate loss func 
        '''
        total_loss=0.0
        for i in range(len(y_data)):
            predict_y=Lmodel.h_x(x_data[i],model_params)
            total_loss+=y_data[i]*math.log(predict_y)+(1-y_data[i])*math.log(1-predict_y)
        
        total_loss/=-len(x_data)
        return total_loss

    def logistic_grads(x_data,y_data,model_params):
        x_grads=[0.0]*len(x_data[0].data)
        for i in range(len(y_data)):
            predict_y=Lmodel.h_x(x_data[i].data,model_params)
            for j in range(len(x_grads)):
                x_grads[j]+=(predict_y-y_data[i])*x_data[i][j]/len(x_data)
        return x_grads
    def update_norm(x_grads,learning_rate):
        '''
        calculate gradient func 
        '''
        for i in range(len(x_grads)):
            model_params[i]+=(-1)*learning_rate*x_grads[i]


    def update_nag(x_data,y_data,learning_rate,model_params,d_params,belta):
        '''
        update params func 
        '''
        prev_model_params=[0.0]*len(model_params)
        for i in range(len(prev_model_params)):
            prev_model_params[i]=model_params[i]-learning_rate*belta*d_params[i]
        grads=Lmodel.logistic_grads(x_data,y_data,prev_model_params)

        for i in range(len(model_params)):
            d_params[i]=belta*d_params[i]+grads[i]
            model_params[i]+=(-1)*learning_rate*d_params[i]
    def update_RMSProp(x_grads,learning_rate,model_params,cache,decay_rate):
        '''
        update params func 
        '''
        for i in range(len(x_grads)):
            model_params[i]+=(-1)*learning_rate*x_grads[i]/(np.sqrt(cache)+0.01)#assume eps ==0.01


    def update_AdaDelta(x_grads,theta_cache,model_params,gradient_cache):
        '''
        update params func 
        '''
        eps=0.001
        theta=[0.0]*len(x_grads)
        for i in range(len(x_grads)):
            model_params[i]+=-1*(np.sqrt(theta_cache+eps)/np.sqrt(gradient_cache+eps))*x_grads[i]

    def update_Adam(x_grads,learning_rate,beta1,beta2,m,v):
        '''
        update params func 
        '''
        eps=1e-6
        v = beta2*v + (1-beta2)*Lmodel.params_exp(x_grads)
        for i in range(len(x_grads)):
            m[i] = beta1*m[i] + (1-beta1)*x_grads[i]
            model_params[i]+= - learning_rate*m[i]/(np.sqrt(v) + eps)
        return(m,v)

    def norm_train(x_train,y_train,x_val,y_val,model_params,learning_rate,iter_num,sgd_size):
        '''
        the whole train process
        '''
        train_loss_arr=[]
        val_loss_arr=[]
        for i in range(iter_num):
            train_loss=Lmodel.logistic_loss(x_train,y_train,model_params)
            val_loss=Lmodel.logistic_loss(x_val,y_val,model_params)

            x_sgd,y_sgd=Lmodel.get_sgd_iter_set(x_train,y_train,sgd_size)
            temp_x_grads=Lmodel.logistic_grads(x_sgd,y_sgd,model_params)
            
            Lmodel.update_norm(temp_x_grads,learning_rate)

            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
        outcome=(train_loss_arr,val_loss_arr)
        return outcome
    def nag_train(x_train,y_train,x_val,y_val,model_params,d_params,learning_rate,iter_num,sgd_size,belta):
        '''
        the whole train process
        '''
        train_loss_arr=[]
        val_loss_arr=[]
        for i in range(iter_num):
            train_loss=Lmodel.logistic_loss(x_train,y_train,model_params)
            val_loss=Lmodel.logistic_loss(x_val,y_val,model_params)

            x_sgd,y_sgd=Lmodel.get_sgd_iter_set(x_train,y_train,sgd_size)
            temp_x_grads=Lmodel.logistic_grads(x_sgd,y_sgd,model_params)
            
            Lmodel.update_nag(x_sgd,y_sgd,learning_rate,model_params,d_params,belta)

            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
        outcome=(train_loss_arr,val_loss_arr)
        return outcome
    def RMSProp_train(x_train,y_train,x_val,y_val,model_params,learning_rate,iter_num,sgd_size,decay_rate):
        '''
        the whole train process
        '''
        train_loss_arr=[]
        val_loss_arr=[]
        cache =0.0
        for i in range(iter_num):
            train_loss=Lmodel.logistic_loss(x_train,y_train,model_params)
            val_loss=Lmodel.logistic_loss(x_val,y_val,model_params)

            x_sgd,y_sgd=Lmodel.get_sgd_iter_set(x_train,y_train,sgd_size)
            temp_x_grads=Lmodel.logistic_grads(x_sgd,y_sgd,model_params)
            if (i==0):
                cache=Lmodel.params_exp(temp_x_grads)**2
            else:
                cache=decay_rate*cache+(1-decay_rate)*Lmodel.params_exp(temp_x_grads)
            Lmodel.update_RMSProp(temp_x_grads,learning_rate,model_params,cache,decay_rate)

            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
        outcome=(train_loss_arr,val_loss_arr)
        return outcome
    def AdaDelta_train(x_train,y_train,x_val,y_val,model_params,learning_rate,iter_num,sgd_size,decay_rate):
        '''
        the whole train process
        '''
        train_loss_arr=[]
        val_loss_arr=[]
        theta_cache=0.0
        gradient_cache=0.0
        for i in range(iter_num):
            train_loss=Lmodel.logistic_loss(x_train,y_train,model_params)
            val_loss=Lmodel.logistic_loss(x_val,y_val,model_params)

            x_sgd,y_sgd=Lmodel.get_sgd_iter_set(x_train,y_train,sgd_size)
            temp_x_grads=Lmodel.logistic_grads(x_sgd,y_sgd,model_params)      

            if (i==0):
                theta_cache=Lmodel.params_exp(model_params)**2
                gradient_cache=Lmodel.params_exp(temp_x_grads)**2
            else:
                theta_cache=decay_rate*theta_cache+(1-decay_rate)*Lmodel.params_exp(model_params)
                gradient_cache=decay_rate*gradient_cache+(1-decay_rate)*Lmodel.params_exp(temp_x_grads)

            Lmodel.update_AdaDelta(temp_x_grads,theta_cache,model_params,gradient_cache)

            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
        outcome=(train_loss_arr,val_loss_arr)
        return outcome
    def Adam_train(x_train,y_train,x_val,y_val,model_params,iter_num,sgd_size,learning_rate,beta1,beta2):
        '''
        the whole train process
        '''
        train_loss_arr=[]
        val_loss_arr=[]
        m=[0.0]*len(x_train[0])
        v=0
        for i in range(iter_num):
            train_loss=Lmodel.logistic_loss(x_train,y_train,model_params)
            val_loss=Lmodel.logistic_loss(x_val,y_val,model_params)

            x_sgd,y_sgd=Lmodel.get_sgd_iter_set(x_train,y_train,sgd_size)
            temp_x_grads=Lmodel.logistic_grads(x_sgd,y_sgd,model_params)      

            m,v=Lmodel.update_Adam(temp_x_grads,learning_rate,beta1,beta2,m,v)

            train_loss_arr.append(train_loss)
            val_loss_arr.append(val_loss)
        outcome=(train_loss_arr,val_loss_arr)
        return outcome

    def draw_pic(train_list,val_list,name_list,iter_num):
        '''
        draw pic 
        '''
        plt.title('TRAIN LOSS')  
        #plt.title('VAL LOSS')  
        plt.xlabel('iteration')  
        plt.ylabel('avg loss')  
        for i in range(len(name_list)):
            plt.plot(range(iter_num), train_list[i],label=name_list[i]+'_train')  
            #plt.plot(range(iter_num), val_list[i],label=name_list[i]+'_val')  

        plt.xticks(range(iter_num), rotation=0)  
        plt.legend(bbox_to_anchor=[0.3, 1])  
        plt.grid()  
        plt.show()  
    pass


if __name__=="__main__":  
    #load data
    x_train,y_train=ds.load_svmlight_file("./a9a")
    x_train=x_train.toarray('c')
    x_val,y_val=ds.load_svmlight_file("./a9a.t")
    x_val=x_val.toarray('c')
    model_params=[0.0]*len(x_train[0])


   
    train_loss_list=[]
    val_loss_list=[]
    name_loss_list=[]
    iter_num=8
    sgd_size=16
    learning_rate=0.01
    #norm train
    norm_train_loss,norm_val_loss=Lmodel.norm_train(x_train,y_train,x_val,y_val,model_params,learning_rate,iter_num,sgd_size)
    train_loss_list.append(norm_train_loss)
    val_loss_list.append(norm_val_loss)
    name_loss_list.append("norm method")
    
 
    #for nag
    model_params=[0.0]*len(x_train[0])
    d_params=[0.0]*len(x_train[0])
    belta=0.1
    nag_train_loss,nag_val_loss=Lmodel.nag_train(x_train,y_train,x_val,y_val,model_params,d_params,learning_rate,iter_num,sgd_size,belta)
    train_loss_list.append(nag_train_loss)
    val_loss_list.append(nag_val_loss)
    name_loss_list.append("nag method")

    #for RMSProp
    model_params=[0.0]*len(x_train[0])
    decay_rate=0.9
    RMSProp_train_loss,RMSProp_val_loss=Lmodel.RMSProp_train(x_train,y_train,x_val,y_val,model_params,learning_rate,iter_num,sgd_size,decay_rate)
    train_loss_list.append(RMSProp_train_loss)
    val_loss_list.append(RMSProp_val_loss)
    name_loss_list.append("RMSProp method")

    #for AdaDelta
    model_params=[0.0]*len(x_train[0])
    decay_rate=0.9
    AdaDelta_train_loss,AdaDelta_val_loss=Lmodel.AdaDelta_train(x_train,y_train,x_val,y_val,model_params,learning_rate,iter_num,sgd_size,decay_rate)
    train_loss_list.append(AdaDelta_train_loss)
    val_loss_list.append(AdaDelta_val_loss)
    name_loss_list.append("AdaDelta method")

    #for Adam
    beta1=0.9
    beta2=0.999
    model_params=[0.0]*len(x_train[0])
    decay_rate=0.9
    Adam_train_loss,Adam_val_loss=Lmodel.Adam_train(x_train,y_train,x_val,y_val,model_params,iter_num,sgd_size,learning_rate,beta1,beta2)
    train_loss_list.append(Adam_train_loss)
    val_loss_list.append(Adam_val_loss)
    name_loss_list.append("Adam method")


    Lmodel.draw_pic(train_loss_list,val_loss_list,name_loss_list,iter_num)