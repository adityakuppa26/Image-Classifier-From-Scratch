import numpy as np
import matplotlib.pyplot as plt
import h5py

def sigmoid(Z):
    A=1/(1+np.exp(-Z))
    cache=Z
    return A,cache

def relu(Z):
    A=np.maximum(0,Z)
    assert(A.shape==Z.shape)
    cache=Z
    return A,cache

def initialize_parameters(layers_dims):
    parameters={}
    L=len(layers_dims)
    for l in range(1,L):
        parameters['W'+str(l)]=np.random.randn(layers_dims[l],layers_dims[l-1])/np.sqrt(layers_dims[l-1])
        parameters['b'+str(l)]=np.zeros((layers_dims[l],1))
        
        assert(parameters['W'+str(l)].shape==(layers_dims[l],layers_dims[l-1]))
        assert(parameters['b'+str(l)].shape==(layers_dims[l],1))
        
    return parameters

def forward_pass(A_prev,W,b,activation):
    
    Z=np.dot(W,A_prev)+b
    linear_cache=(A_prev,W,b)
    
    if(activation=="sigmoid"):
        A,activation_cache=sigmoid(Z)
    
    elif(activation=="relu"):
        A,activation_cache=relu(Z)
    
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache=(linear_cache,activation_cache)
    
    return A,cache

def L_model_forward(X,parameters):
    
    L=len(parameters)//2
    A=X
    caches=[]
    
    for l in range(1,L):
        A_prev=A
        A,cache=forward_pass(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],"relu")
        caches.append(cache)
        
    AL,cache=forward_pass(A,parameters['W'+str(L)],parameters['b'+str(L)],"sigmoid")
    caches.append(cache)
    
    assert(AL.shape==(1,X.shape[1]))
    return AL,caches

def compute_cost(AL,Y):
    
    m=Y.shape[1]
    
    cost=(-1/m)*(np.dot(Y,np.log(AL).T) + np.dot(1-Y,np.log(1-AL).T))
    
    cost=np.squeeze(cost)
    assert(cost.shape==())
    
    return cost

def backward_pass(dA,cache,activation):
    
        linear_cache,activation_cache=cache
        
        if (activation=="sigmoid"):
            
            Z=activation_cache
            
            s= 1/(1+np.exp(-Z))
            
            dZ=dA*s*(1-s)
            assert(dZ.shape==Z.shape)
            
            A_prev,W,b=linear_cache
            m = A_prev.shape[1]
            
            dW= (1/m)*np.dot(dZ,A_prev.T)
            db = (1/m)*np.sum(dZ,axis=1,keepdims=True)
            dA_prev= np.dot(W.T,dZ)
            
            return dA_prev, dW, db
        
        elif (activation=="relu"):
            
            Z=activation_cache
            
            dZ = np.array(dA,copy=True)
            
            dZ[Z<=0]=0
            assert(dZ.shape==Z.shape)
            
            A_prev,W,b=linear_cache
            m = A_prev.shape[1]
            
            dW = (1/m)*np.dot(dZ,A_prev.T)
            db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
            dA_prev=np.dot(W.T,dZ)
            
            return dA_prev,dW,db
        
def L_model_backward(AL,Y,caches):
    
    grads={}
    L=len(caches)
    Y=Y.reshape(AL.shape)
    
    dAL= - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    current_cache=caches[L-1]
    grads["dA" + str(L-1)],grads["dW" + str(L)],grads["db" + str(L)] = backward_pass(dAL,current_cache,"sigmoid")
    
    for l in reversed(range(L-1)):
        
        current_cache=caches[l]
        grads["dA"+str(l)],grads["dW"+str(l+1)],grads["db"+str(l+1)] = backward_pass(grads["dA"+str(l+1)],current_cache,"relu")
        
    return grads

def update_parameters(parameters,grads,learning_rate):
    
    L=len(parameters)//2
    
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
        
    return parameters

def predict(X,y,parameters):
    
    m = X.shape[1]
    p=np.zeros((1,m))
    
    probas,caches = L_model_forward(X,parameters)
    
    for l in range(0,probas.shape[1]):
        
        if (probas[0,l]>0.5):
            p[0,l]=1
            
        else:
            p[0,l]=0
            
    print("Accuracy: "  + str(np.sum((p == y)/m)))
            
    return p

def load_data():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


            

            

