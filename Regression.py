#!/usr/bin/env python
# coding: utf-8
Ex1
# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[48]:


data = pd.read_csv("data2.csv")
N = data.shape[0]


# In[50]:


X = data.values[:,0].reshape(-1,1)
y = data.values[:,1].reshape(-1,1)
plt.scatter(X,y,marker="o")
plt.xlabel("S")
plt.ylabel("price")
plt.show()


# In[51]:


X1 = np.hstack((np.ones((N,1)), X))
w = np.array([0.,1.]).reshape(-1,1)
w


# In[52]:


X1.shape,w.shape


# In[53]:


learning_rate = 0.000001
iteration = 100
cost = np.zeros((iteration,1))
for i in range(1,iteration):
    error = np.dot(X1,w)-y
    cost[i] = 0.5*np.sum(error*error)
    w[0] -= learning_rate*np.sum(error)
    w[1] -= learning_rate*np.sum(np.multiply(error,X1[:,1].reshape(-1,1)))
print(f"predict line: y = {w[0]}+{w[1]}x")


# In[56]:


predict = np.dot(X1, w)
plt.plot((X1[0][1], X1[N-1][1]),(predict[0], predict[N-1]), 'r')
plt.scatter(X,y,marker="o")
plt.xlabel("S")
plt.ylabel("price")
plt.show()


# In[60]:


# du doan gia can nhà có S = 50,100,150
x = np.array([[50],[100],[150]])
y = w[0] + w[1]*x
print(y)

Ex2
# In[3]:


import seaborn as sns


# In[4]:


column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing = pd.read_csv("housing.csv",header=None,delimiter=r"\s+",names=column_names)


# In[5]:


housing.describe()


# In[6]:


for k, v in housing.items(): #k is variable
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(housing)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))


# In[7]:


housing = housing[~(housing['MEDV'] >= 50.0)]
print(np.shape(housing))


# In[8]:


plt.figure(figsize=(20, 10))
sns.heatmap(housing.corr().abs(),  annot=True) #TAX and RAD are highly correlated features


# In[39]:


#we need to normalize the features using mean normalization  
housing = (housing - housing.mean())/housing.std()  
  
#setting the matrixes  
X = housing.iloc[:,0:13]  
ones = np.ones([X.shape[0],1])  
X = np.concatenate((ones,X),axis=1)  # X (n,14)
  
y = housing.iloc[:,-1].values.reshape(-1,1) #.values converts it from pandas.core.frame.DataFrame to numpy.ndarray  
theta = np.zeros([1,14]) 
  
#computecost  
def computeCost(X,y,theta):  
    tobesummed = np.power(((X @ theta.T)-y),2)  
    return np.sum(tobesummed)/(2 * len(X))  
  
def gradientDescent(X,y,theta,iters,alpha):  
    cost = np.zeros(iters)  
    for i in range(iters):  
        theta = theta - (alpha/len(X)) * np.sum(X * (X @ theta.T - y), axis=0)  
        cost[i] = computeCost(X, y, theta)  
      
    return theta,cost  
  
#set hyper parameters  
alpha = 0.01  
iters = 100
  
g,cost = gradientDescent(X,y,theta,iters,alpha)  
print(g)  
  
finalCost = computeCost(X,y,g)  
print(finalCost)  
  
fig, ax = plt.subplots()    
ax.plot(np.arange(iters), cost, 'r')    
ax.set_xlabel('Iterations')    
ax.set_ylabel('Cost')    
ax.set_title('Error vs. Training Epoch')  


# In[ ]:




