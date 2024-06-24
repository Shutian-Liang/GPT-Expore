import numpy as np
import pandas as pd
import json

path = '../kernels/'
def envir_values(env):
    '''
    return environment values according to generated data \ 
    env: 'smooth' or 'rough'
    '''
    # read reward data
    if env == 'smooth':
        file_name = 'kernel2.json'
    else:
        file_name = 'kernel1.json'
        
    with open(path+file_name) as file:
        # 2. 加载 JSON 数据
        data = json.load(file)
    
    # set list values to save environment reward structure
    env_nums = 40
    env_values = []   # final length is env_nums and it's a 2 dimensional vector
    for i in range(env_nums):
        per_values = []
        for _ in range(30):
            per_values.append(data[str(i)][str(_)]['y'])
        env_values.append(per_values)
    return env_values

smooth_env = envir_values('smooth')
rough_env = envir_values('rough')

def simulation(iter=10):
    envs = [smooth_env,rough_env]
    # 每个环境下随机游走
    horizons = [5,10]
    data = pd.DataFrame()
    for t in range(len(envs)):
        for _ in range(len(horizons)):
            horizon = horizons[_]
            env_type = envs[t]
            environment = 'smooth' if t == 0 else 'rough'
            for i in range(len(env_type)):
                env = env_type[i] #current environment
                for j in range(iter):
                    grid = j
                    for k in range(1,horizon+1):
                        x = np.random.randint(0,30)
                        z = env[x]*100
                        trials = k
                        kernel = i
                        condition = 'random'
                        rand = pd.DataFrame({'trials':[trials],'x':[x],'z':[z],'kernel':[kernel],'environment':[environment],'agent':[condition],'grid':[grid],'horizons':[horizon]})
                        data = pd.concat([data,rand],axis=0)
                    print(f'env {t} kernel {i} iter {j} done!')
    return data

random = simulation()
random.to_csv('random_walk.csv')