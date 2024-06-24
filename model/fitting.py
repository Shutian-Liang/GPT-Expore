import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from constrNMPy import constrNMPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Gaussian Process
# just with the scikit-learn library

# Option learning
# Bayesian Mean Tracker
def Tracker(mu:np.array,v:np.array,option:int,theta:float,y):
    """ Update the mean and variance of the posterior distribution of the Gaussian Process

    Args:
        mu (np.array): the mean estimation of the posteriors
        v (np.array): the variance estimation of the posteriors
        option (int): the choice of the 
        theta (float): hyperparameter of the model it seems the sensitivity
        y ([type]): the reward of the choice
    """
    
    # set the choice
    choices = np.zeros(mu.shape)
    choices[option] = 1

    # set the Git
    Git = v/(v+theta**2)

    # update 
    mu += choices*Git*(y-mu)
    v = v*(1-choices*Git)

    return mu,v

# upper confidence bound
def UCB(mu:np.array,sigma:np.array,beta:float):
    """return the upper confidence bound of the Gaussian Process

    Args:
        mu (np.array): the mean estimation of the posteriors
        sigma (np.array): the variance estimation of the posteriors
        beta (float): direct the exploration

    Returns:
        _type_: the ultility of the choices
    """
    return mu+beta*sigma

# Entropy Regularization
def Entropy(mu:np.array,sigma:np.array,beta:float):
    """return the entropy of the Gaussian Process

    Args:
        mu (np.array): the mean estimation of the posteriors
        sigma (np.array): the variance estimation of the posteriors

    Returns:
        _type_: _description_
    """
    return mu+beta*0.5*np.log(2*np.pi*np.e*sigma**2)

# Pure Exploitation
def Exploit(mu:np.array):
    return mu

# Pure Exploration
def Explore(sigma:np.array):
    return sigma

# Expected Improvement
def EXI(x:np.array,mu:np.array,sigma:np.array):
    """return the expected improvement of the Gaussian Process

    Args:
        x (np.array): the choices
        mu (np.array): the mean estimation of the posteriors
        sigma (np.array): the variance estimation of the posteriors

    Returns:
        _type_: the ultility of the choices
    """
    x_star = x[np.argmax(mu[x])]
    z = (mu-mu[x_star])/sigma
    
    cdf = norm.cdf(z)
    pdf = norm.pdf(z)
    ultility = cdf*(mu-mu[x_star])+sigma*pdf
    ultility = np.where(ultility>1e-4,ultility,1e-4)
    return ultility

# Probability of Improvement
def POI(x:np.array,mu:np.array,sigma:np.array):
    """return the probability of improvement of the Gaussian Process

    Args:
        x (np.array): the choices
        mu (np.array): the mean estimation of the posteriors
        sigma (np.array): the variance estimation of the posteriors

    Returns:
        _type_: the ultility of the choices
    """
    x_star = x[np.argmax(mu[x])]
    z = (mu-mu[x_star])/sigma
    cdf = norm.cdf(z)
    cdf = np.where(cdf>1e-4,cdf,1e-4)
    return cdf

# win-stay lose-sample
def WSLS(x:np.array,y:np.array):
    """return the win-stay lose-sample of the Gaussian Process

    Args:
        x (np.array): the choices
        y (np.array): the reward of the choices

    Returns:
        _type_: the ultility of the choices
    """
    if len(x) > 1:
        x_prev,y_prev = x[:-1],y[:-1]
        x_curr,y_curr = x[-1],y[-1]
    else:
        x_prev,y_prev = x[0],[0]
        x_curr,y_curr = x[0],y[0]
    probalility = np.zeros(30)
    if y_curr >= np.max(y_prev):
        if x_curr == 0:
            probalility[x_curr:x_curr+2] = 1/2
        elif x_curr == 29:
            probalility[x_curr-1:x_curr+1] = 1/2
        else:
            probalility[x_curr-1:x_curr+2] = 1/3
    else:
        unchoiced_index = np.delete(np.arange(30),x_curr)
        probalility[unchoiced_index] = 1/len(unchoiced_index)
    
    return probalility

# Local Search
def IMD(x:np.array,y:np.array,beta:float,horizon,iter):
    """_summary_

    Args:
        x (np.array): the choice history

    Returns:
        _type_: _description_
    """
    x_range = np.arange(30)
    x_prev = x[-1]
    mu = np.ones(30)*0.5

    distance = abs(x_range-x_prev)
    distance = np.where(distance == 0,1,distance)

    for _ in range(len(x)):
        x_now = x[_]
        mu[x_now] = y[_]

    ultility = 1/distance + beta*np.exp(iter-horizon)*mu
    return ultility

def softmax(x, tau):
    max_x = np.max(x)
    shifted_x = x - max_x
    exp_x = np.exp(shifted_x / tau)
    sum_exp_x = np.sum(exp_x)
    prob = exp_x / sum_exp_x
    prob = np.maximum(prob, 1e-15)  # 避免 prob 过小
    return prob

# load data
per_gptdata = pd.read_csv('../analysis/per_gptdata.csv',index_col=0)
per_gptdata.x = per_gptdata.x -1 
per_gptdata

# class function learning
size = 30
def GP(paras:list,method:str,data:pd.DataFrame)->float:
    """_summary_

    Args:
        method (str): _sampling method_
        paras (list): _paras for modeling_
        data (pd.DataFrame): _sub data_

    Returns:
        float: _likelihood of the data_
    """
    options = np.arange(size)
    mu = np.array([0. for _ in range(size)])
    sigma = np.array([0.5 for _ in range(size)])
    length = paras[-1]
    kernel = RBF(length_scale = length)
    gp = GaussianProcessRegressor(kernel=kernel,optimizer=None,normalize_y=True)
    likelihood = 0

    # with 3 paras
    if (method == 'UCB') or (method == 'Entropy') or (method == 'Explore') or (method == 'Exploit'):
        # directed exploration, random exploration, and exploitation

        if (method == 'UCB') or (method == 'Entropy'):
            beta,tau,length = paras[0],paras[1],paras[2]
            for i in range(len(data)):
                params = (mu,sigma,beta)
                prob = softmax(eval(method)(*params),tau)
                choice = data.x.values[i]
                if i != 0:
                    likelihood += np.log(prob[choice])

                # update the GP
                gp.fit(data.x.values[:i+1].reshape(-1, 1),data.z.values[:i+1].reshape(-1, 1))
                mu,sigma = gp.predict(options.reshape(-1,1),return_std=True)
        else:
            tau,length = paras[0],paras[1]
            for i in range(len(data)):
                if method == 'Exploit':
                    prob = softmax(Exploit(mu),tau)
                else:
                    prob = softmax(Explore(sigma),tau)
                    
                choice = data.x.values[i]
                if i != 0:
                    likelihood += np.log(prob[choice])

                # update the GP
                gp.fit(data.x.values[:i+1].reshape(-1, 1),data.z.values[:i+1].reshape(-1, 1))
                mu,sigma = gp.predict(options.reshape(-1,1),return_std=True)
        
    
    if (method == 'EXI') or (method == 'POI'):
        tau,length = paras[0],paras[1] # just the temperature
        for i in range(len(data)):
            x = data.x.values[:i+1]
            params = (x,mu,sigma)
            prob = softmax(eval(method)(*params),tau)
            choice = data.x.values[i]
            if i != 0:
                likelihood += np.log(prob[choice])
            
            # update the GP
            gp.fit(data.x.values[:i+1].reshape(-1, 1),data.z.values[:i+1].reshape(-1, 1))
            mu,sigma = gp.predict(options.reshape(-1,1),return_std=True)
    
    return -likelihood

# class option learning
def OL(paras:list,method:str,data:pd.DataFrame)->float:
    """return the likelihood of the option learning fitting

    Args:
        method (str): _sampling strategy_
        paras (list): _used for modeling_
        data (pd.DataFrame): _sub data_

    Returns:
        : _likelihood of the option learning fitting_
    """

    options = np.arange(size)
    mu = np.array([0.5 for _ in range(size)])
    sigma = np.array([5 for _ in range(size)])
    likelihood = 0
        
    if (method == 'UCB') or (method == 'Entropy') or (method == 'Explore') or (method == 'Exploit'):
        # with 3 paras
        # directed exploration, random exploration, and exploitation
        if (method == 'UCB') or (method == 'Entropy'):
            beta,tau,theta = paras[0],paras[1],paras[2]
            for i in range(len(data)):
                params = (mu,sigma,beta)
                prob = softmax(eval(method)(*params),tau)
                choice = data.x.values[i]
                if i != 0:
                    likelihood += np.log(prob[choice])
                # update the posterior with Bayesian Mean Tracker
                mu,sigma = Tracker(mu,sigma,choice,theta,data.z.values[i]/100)

        else:
            tau,theta= paras[0],paras[1]
            for i in range(len(data)):
                if method == 'Exploit':
                    prob = softmax(Exploit(mu),tau)
                else:
                    prob = softmax(Explore(sigma),tau)
                    
                choice = data.x.values[i]
                if i != 0:
                    likelihood += np.log(prob[choice])
                # update the posterior with Bayesian Mean Tracker
                mu,sigma = Tracker(mu,sigma,choice,theta,data.z.values[i]/100)
                
    if (method == 'EXI') or (method == 'POI'):
        # with 2 paras
        tau,theta = paras[0],paras[1]
        for i in range(len(data)):
            x = data.x.values[:i+1]
            params = (x,mu,sigma)
            prob = softmax(eval(method)(*params),tau)
            choice = data.x.values[i]
            if i != 0:
                likelihood += np.log(prob[choice])
            
            # update the posterior with Bayesian Mean Tracker
            mu,sigma = Tracker(mu,sigma,choice,theta,data.z.values[i]/100) 
            
    return -likelihood

# class heuristic learning
def heuristic(paras:list,method:str,data:pd.DataFrame)->float:
    """return the likelihood of the heuristic learning fitting

    Args:
        paras (list): _used for modeling_
        method (str): _sampling strategy_
        data (pd.DataFrame): _sub data_

    Returns:
        : _likelihood of the heuristic learning fitting_
    """
    options = np.arange(size)
    likelihood = 0
    if method == 'WSLS':
        tau = paras[0]
        for i in range(1,len(data)):
            params = (data.x.values[:i+1],data.z.values[:i+1]/100)
            prob = softmax(eval(method)(*params),tau)
            choice = data.x.values[i]
            if i != 0:
                likelihood += np.log(prob[choice])
    
    if method == 'IMD':
        beta,tau = paras[0],paras[1]
        for i in range(len(data)):
            params = (data.x.values[:i+1],data.z.values[:i+1]/100,beta,len(data)-1,i)
            prob = softmax(eval(method)(*params),tau)
            choice = data.x.values[i]
            if i != 0:
                likelihood += np.log(prob[choice])
    
    return -likelihood

'''
# start to fit with learning and decision making model
path = 'new_fit/'
learning = ['GP','OL']
methods = ['UCB','Entropy','Explore','Exploit','EXI','POI']
#per_gptdata = per_gptdata[per_gptdata.t < 1]
ids = per_gptdata.id.unique()
for i in range(len(learning)):
    for j in range(len(methods)):
        method = methods[j]

        for k in range(420,len(ids)):
            data = per_gptdata[per_gptdata.id == ids[k]]
            t = data.t.values[0]
            paras = []
            N_random=10               #重复取值20次
            optimal=[]
            args = [method,data]
            for _ in range(N_random):
                if j < 4:
                    LB = [1e-16]*3;UB = [10-1e-16]*3
                    x0 = np.random.uniform(0, 3, 3)
                else:
                    LB = [1e-16]*2;UB = [10-1e-16]*2
                    x0 = np.random.uniform(0, 3, 2)

                if learning[i] == 'GP':
                    xopt = constrNMPy.constrNM(GP, x0, LB, UB, args=args)
                elif learning[i] == 'OL':
                    xopt = constrNMPy.constrNM(OL, x0, LB, UB, args=args)
                optimal.append(xopt['xopt'])
                print(f' {learning[i]} {method} sub{k}//{len(ids)} iter{_} finished')
            if learning[i] == 'GP':
                y = list(map(lambda x: GP(x, method,data), optimal)) 
            else:
                y = list(map(lambda x: OL(x, method,data), optimal))
            opt = optimal[y.index(min(y))]

            # 保存最优参数
            for _ in range(len(data)):
                paras.append(opt)
            
            if j<4:
                if i == 0:
                    columns = ['beta','tau','length']
                else:
                    columns = ['beta','tau','theta']
            else:
                if i == 0:
                    columns = ['tau','length']
                else:
                    columns = ['tau','theta']

            file_name = f'{learning[i]}_{method}_{k}_{t}.csv'
            paras = pd.DataFrame(paras,columns=columns)
            paras.to_csv(path+file_name)


# start to fit with learning and decision making model
path = 'new_fit/'
learning = ['heuristic']
methods = ['IMD','WSLS']
ids = per_gptdata.id.unique()
for i in range(len(learning)):
    for j in range(len(methods)):
        method = methods[j]

        for k in range(420,len(ids)):
            data = per_gptdata[per_gptdata.id == ids[k]]
            t = data.t.values[0]
            paras = []
            N_random=10               #重复取值10次
            optimal=[]
            args = [method,data]
            for _ in range(N_random):  
                if j == 0:
                    LB = [1e-16]*2;UB = [10-1e-16]*2
                    x0 = np.random.uniform(0, 3, 2)
                else:   
                    LB = [1e-16];UB = [10-1e-16]
                    x0 = np.random.uniform(0, 3, 1)
                xopt = constrNMPy.constrNM(heuristic, x0, LB, UB, args=args)
                optimal.append(xopt['xopt'])
                print(f' {learning[i]} {method} sub{k}/{len(ids)} iter{_} finished')
            y = list(map(lambda x: heuristic(x,method,data), optimal))
            opt = optimal[y.index(min(y))]
            # 保存最优参数
            for _ in range(len(data)):
                paras.append(opt)
            
            if j == 1:
                columns = ['tau']
            else:
                columns = ['beta','tau']
            file_name = f'heuristics_{method}_{k}_{t}.csv'
            paras = pd.DataFrame(paras,columns=columns)
            paras.to_csv(path+file_name)
'''
