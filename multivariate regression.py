
# coding: utf-8

# In[1]:


#importing libraries for modelling
import numpy as np
import pandas as pd

#taking care of varibles required for our linear model
dataset=pd.read_csv('student.csv')
x=dataset.iloc[:,:2].values
y=dataset.iloc[:,-1].values
theta=np.array([[0,0,0]])
y=y.reshape(1000,1)
x=np.c_[np.ones(x.shape[0]),x]
m=y.shape[0]
alpha=0.0001

#calculating error differance between predtion and original before implementing
pred=np.dot(theta,x.T) 
pred=pred.reshape(1000,1)
print(sum(y-pred))

#defing a cost function to calculate cost actually we don't need it but its a good practice
def cost_function(x,y,theta,m):
    return np.sum((np.dot(x,theta.T)-y)**2)/(2*m);
#printing cost intial cost for referance for future use
print("intial_cost=", cost_function(x,y,theta,m))


#gradient desecent function but i am doing it in loop rather than creating a gradient function
for i in range(300):
    prediction=np.dot(x,theta.T)
    error=prediction-y
    theta=theta-alpha*(1/m)*np.dot(error.T,x)

#again calculating error after model is trained    
pred=np.dot(theta,x.T) 
pred=pred.reshape(1000,1)
print(sum(pred-y))

print("final cost",cost_function(x,y,theta,m))
print(theta)

