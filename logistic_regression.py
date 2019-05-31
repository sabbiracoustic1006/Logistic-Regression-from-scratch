
# coding: utf-8

# In[8]:


import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from random import shuffle
from tqdm import tqdm


# In[9]:


class Dataset:
    def __init__(self,dataset):
        for key in dataset.keys():
            setattr(self,key,dataset[key])
            

            
iris = Dataset(datasets.load_iris())


# In[35]:


class Classifier:
    """args should be x,y,y_categorical,classifier_type,validation,split,l_r,num_repeat"""
    def __init__(self,**args):
        for key in args.keys():
            setattr(self,key,args[key])
        
        self.x = np.concatenate((np.ones((self.x.shape[0],1)),self.x),1)
        
        if self.validation:
            self.x_train,self.x_val,self.y_train,self.y_val = train_test_split(self.x,self.y,test_size=self.split)
        else:
            self.x_train = shuffle(self.x,0)
            self.y_train = shuffle(self.y,0)
            
        self.parameters = {}
        
        for class_ in self.y_categorical:
            self.parameters[class_] = np.random.rand(1,self.x.shape[1])/self.x.shape[1]**0.5
        
    @staticmethod
    def sigmoid(x,w):
        return 1/(1+np.exp(-np.sum(x*w,1)))
        
    
    def loss(self,x,w,y):
        total_loss = y*np.log(self.sigmoid(x,w))+(1-y)*np.log(1-self.sigmoid(x,w))
        return -np.sum(total_loss)
    
    def accuracy(self,x,w,y):
        y_hat = (self.sigmoid(x,w)>=0.5).astype('int')
        return sum(y == y_hat)/len(y)
                
    def train(self):
        for key in self.parameters.keys():
            learning_rate = self.l_r
            
            parameters = self.parameters[key]
            
            class_num = list(self.y_categorical).index(key)
            
            pseudo_y_train = (self.y_train == class_num).astype('float32')
            pseudo_y_val = (self.y_val == class_num).astype('float32')
            
            best_loss = self.loss(self.x_val,parameters,pseudo_y_val)
            print('initial val loss %.3f'%best_loss)
            
            tdm = tqdm(range(self.num_repeat),'train_accuracy')
            
            for epoch in tdm:
                
                grads = np.expand_dims((pseudo_y_train-self.sigmoid(self.x_train,self.parameters[key])),-1)*self.x_train
                
                if self.grad_avg: grads = [np.mean(grads,0)]
                
                for grad in grads:
                    parameters += learning_rate*grad
                    train_loss = self.loss(self.x_train,parameters,pseudo_y_train)
                    train_acc = self.accuracy(self.x_train,parameters,pseudo_y_train) 
                    tdm.set_description('%.3f'%train_acc)
                    
                val_loss = self.loss(self.x_val,parameters,pseudo_y_val)
                val_accuracy = self.accuracy(self.x_val,parameters,pseudo_y_val)
                
                if epoch % 5 == 0 : learning_rate /= 1.5
                
                if val_loss < best_loss: 
                    self.parameters[key] = parameters
                    print('val loss improved from %.3f to %.3f parameters updated'%(best_loss,val_loss))
                    print('accuracy for %s classifier is %.3f'%(key,val_accuracy))
                    best_loss = val_loss
                else: print('val loss did not improve')
                    
            


# In[48]:


classifier = Classifier(x=cancer.data,y=cancer.target,y_categorical=cancer.target_names,
                        classifier_type='logistic',validation=True,split=0.25,l_r=0.0001,num_repeat=100,grad_avg=False)


classifier.train()

