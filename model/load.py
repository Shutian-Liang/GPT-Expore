import pandas as pd  
import numpy as np
import json

def envir_values(env):
    '''
    return environment values according to generated data \ 
    env: 'smooth' or 'rough'
    '''
    path = '../kernels/'
    # read reward data
    if env == 'smooth':
        file_name = 'kernel2.json'
    else:
        file_name = 'kernel1.json'
        
    with open(path+file_name) as file:
        # 2. 加载 JSON 数据
        data = json.load(file)
    
    # set list values to save environment reward structure
    env_nums = len(data)  # 40
    env_values = []   # final length is env_nums and it's a 2 dimensional vector
    for i in range(env_nums):
        per_values = []
        for _ in range(30):
            per_values.append(data[str(i)][str(_)]['y'])
        env_values.append(per_values)
    return env_values

# data 有40个环境，每个环境有30个点
# give the examples
def show_example(smoothness:str) -> str:
    """return the example to the gpt

    Args:
        smoothness (str): the type of the environment
    """
    rows = 30
    envs = envir_values(smoothness)
    
    env_index1,env_index2,env_index3,env_index4 = np.random.randint(0,40,size=4)
    env1,env2,env3,env4 = envs[env_index1],envs[env_index2],envs[env_index3],envs[env_index4]
    envs = [env1,env2,env3,env4]
    example_data = [env_index1,env_index2,env_index3,env_index4]
    max1,max2,max3,max4 = np.random.randint(60,81,size=4)
    maxs = [max1,max2,max3,max4]
    example = ''
    for _ in range(len(envs)):
        current_env = ''
        env = envs[_]
        current_max = maxs[_]
        for i in range(rows):
            if i == 0:
                example += f'example{_} is:\n'
            current_env += f'{int(current_max*env[i])} '
            if i == 29:
                current_env += '\n\n'
                example += current_env
    example += 'If you understand the distribution, say NEXT'
    return example,example_data

# sample the sub environment not including the example index
def draw_numbers(index_label,size):
    """return 2 environment index

    Args:
        env_index1 (int): example environment index 1
        env_index2 (int): example environment index 2

    Returns:
        two int of the environment index not including the index above
    """

    numbers = np.delete(np.arange(0, 40), index_label)
    return np.random.choice(numbers, size, replace=False)

def draw_max(size):
    import numpy as np
    """return the global max of the current environment
    """
    return np.random.randint(60,81,size=size)

