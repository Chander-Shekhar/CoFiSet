import numpy as np 
import pandas as pd


# Data Preprocessing
data = pd.read_table('../dataset/u1base.txt', delim_whitespace=True, header = None, parse_dates=False)
dataset=data.values[:,:2]
feedback=data.values[:,2]
users=data.values[:,0]
observed_data=dataset[feedback>=3]
d=10
#Data Initialization
Users=np.random.normal(0,1,(users[-1],d))
Items=np.random.normal(0,1,(max(dataset[:,1]),d))
Bias=np.random.normal(0,1,(max(dataset[:,1])))

P_users=[]
A_users=[]

total_items=np.array(range(1,max(dataset[:,1])+1))
for i in range(dataset[-1,0]):
    P_users.append(dataset[users==i+1,1])
    A_users.append(np.array([item for item in list(total_items) if item not in list(P_users[i])]))

T=10
learning_rate=0.1
alpha_u=alpha_v=beta_v=0.1
        
perm=np.random.permutation(Users.shape[0])
Users=Users[perm]

def sigmoid_deriv(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm * (1. - sigm)


def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm

def preference(user,P,A):
    rp=0
    for i in range(P.shape[0]):
        rp+=np.dot(Users[user],Items[P[i]-1])+Bias[P[i]-1]
        
    ra=0
    for i in range(A.shape[0]):
        ra+=np.dot(Users[user],Items[A[i]-1])+Bias[A[i]-1]
    
    return rp/P.shape[0]-ra/A.shape[0]

def calc_mean(d,P,A):
    P_mean=np.zeros((d))
    for i in P:
        P_mean+=Items[i-1]
    A_mean=np.zeros((d))
    for i in A:
        A_mean+=Items[i-1]
    P_mean=P_mean/P.shape[0]
    A_mean=A_mean/A.shape[0]
    return P_mean,A_mean




for t1 in range(1000):
    loss=0
    for user in range(Users.shape[0]):
        p_size=np.random.randint(d)+1
        a_size=np.random.randint(p_size)+1
        P_indices=np.random.randint(P_users[user].shape[0],size=p_size)
        P=P_users[user][P_indices]
        A_indices=np.random.randint(int(len(A_users[user])),size=a_size)
        A=A_users[user][A_indices]
        rpa=preference(user,P,A)
        #loss
        loss+=-np.log(sigmoid(rpa))
        #derivate of loss fn with respect to R_upa
        
        del_rpa=-sigmoid(-rpa)
        P_mean,A_mean=calc_mean(d,P,A)
        
        
        
        #update equation
        Users[user]-= learning_rate*((P_mean-A_mean)*del_rpa + alpha_u*Users[user])
        for i in P:
            Items[i-1]-= learning_rate*(del_rpa*Users[user]/P.shape[0] + alpha_v*Items[i-1])
            
        P_mean,A_mean=calc_mean(d,P,A)
        Users[user]-= learning_rate*((P_mean-A_mean)*del_rpa + alpha_u*Users[user])
        
        for i in A:
            Items[i-1]-= learning_rate*(-del_rpa*Users[user]/A.shape[0] + alpha_v*Items[i-1])
        
        P_mean,A_mean=calc_mean(d,P,A)
        Users[user]-= learning_rate*((P_mean-A_mean)*del_rpa + alpha_u*Users[user])
        
        for i in P:
            Bias[i-1]-= learning_rate*(del_rpa/P.shape[0] + beta_v*Bias[i-1] )
        for i in A:
            Bias[i-1]-= learning_rate*(del_rpa/A.shape[0] + beta_v*Bias[i-1] )
    loss_epoch=loss/Users.shape[0]
    print(loss_epoch)

#testing


# Data Preprocessing
data_test = pd.read_table('../dataset/u1test.txt', delim_whitespace=True, header = None, parse_dates=False)
dataset_test=data_test.values[:,:2]
feedback_test=data_test.values[:,2]
users_test=data_test.values[:,0]
observed_data_test=dataset_test[feedback_test>=3] 
P_users_test=[]
for i in range(dataset_test[-1,0]):
    P_users_test.append(dataset_test[users_test==i+1,1])

print(len(P_users_test))
print(Users.shape[0])

K=5
R=[]
Ranking=[]
for u in range(Users.shape[0]):
    temp=[]
    for i in range(Items.shape[0]):
        temp.append(np.dot(Users[u],Items[i])+Bias[i])
    Ranking.append(np.argsort(temp)[::-1])
    R.append(np.sort(temp)[::-1])
    
print(len(Ranking))
Precision=0
for u in range(Users.shape[0]):
    Precision_u=0
    for i in range(K):
        if(Ranking[u][i] in dataset_test[users_test==u,:]):
            Precision_u+=1
    Precision+=Precision_u/K
Precision=Precision/Users.shape[0]
print(Precision)







    
        
        
        
        