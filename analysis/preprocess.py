import os
import pandas as pd
import numpy as np

# 预处理数据
# 获取data文件夹中所有的csv文件
###### gpt data
'''
csv_files = [f for f in os.listdir('../experiment/data') if f.endswith('.csv')]

# 读取每个csv文件，并将它们添加到一个列表中
dataframes = [pd.read_csv(os.path.join('../experiment/data/', f)) for f in csv_files]

# 根据文件名为每个数据框添加一个新的列id，是sub之后的数字
for i, df in enumerate(dataframes):
    df['id'] = i
    length = len(df)
    horizons = []
    for j in range(1,length):
        if df.loc[j,'trials'] == 1:
            horizons += [df.loc[j-1,'trials']]*df.loc[j-1,'trials']
    horizons += [df.loc[length-1,'trials']]*df.loc[length-1,'trials']
    df['horizons'] = horizons   

# 将这个列表中的所有数据框合并到一个数据框中
combined_df = pd.concat(dataframes, ignore_index=True)

#删掉combined_df第一列
first_column = combined_df.columns[0]
gptdata = combined_df.drop(first_column, axis=1)

# 数据预处理
gptdata.rename(columns={'zs':'z'}, inplace=True)
gptdata['z'] = gptdata['z']*100
gptdata['agent'] = 'gpt'

# 添加欧式距离列

def Edis_compute(df):
    dis, index =[-1],df.shape[0]
    for i in range(1,index):
        if df.loc[i,'horizons'] == df.loc[i-1,'horizons']:
            dis.append(abs(df.loc[i,'x']-df.loc[i-1,'x']))   #曼哈顿距离计算
        else:
            dis.append(-1)
    df['Edistance'] = np.array(dis) 
    return df

# 删掉第一列
gptdata = Edis_compute(gptdata)
gptdata.to_csv('gptdata.csv')
'''

# 加载单步拟合列
###### gpt data
import os 
import pandas as pd
import numpy as np
csv_files = [f for f in os.listdir('../experiment/new_data') if f.endswith('.csv')]

# 读取每个csv文件，并将它们添加到一个列表中
dataframes = [pd.read_csv(os.path.join('../experiment/new_data/', f)) for f in csv_files]

# 根据文件名为每个数据框添加一个新的列id，是sub之后的数字

for i, df in enumerate(dataframes):
    df['horizons'] = df['trials'].max()   
    df['id'] = i


# 将这个列表中的所有数据框合并到一个数据框中
combined_df = pd.concat(dataframes, ignore_index=True)

#删掉combined_df第一列
first_column = combined_df.columns[0]
per_gptdata = combined_df.drop(first_column, axis=1)

# 数据预处理
per_gptdata.rename(columns={'zs':'z'}, inplace=True)
per_gptdata['z'] = per_gptdata['z']*100
per_gptdata['agent'] = 'per_gpt'

# 读取数据
def Edis_compute(df):
    dis, index =[-1],df.shape[0]
    for i in range(1,index):
        if df.loc[i,'id'] == df.loc[i-1,'id']:
            dis.append(abs(df.loc[i,'x']-df.loc[i-1,'x']))   #曼哈顿距离计算
        else:
            dis.append(-1)
    df['Edistance'] = np.array(dis)
    return df
# 删掉第一列
per_gptdata = Edis_compute(per_gptdata)
per_gptdata.to_csv('per_gptdata.csv')



# human data
import pandas as pd 
import numpy as np
def Edis_compute(df):
    dis, index =[-1],df.shape[0]
    for i in range(1,index):
        if df.loc[i,'horizons'] == df.loc[i-1,'horizons']:
            dis.append(abs(df.loc[i,'x']-df.loc[i-1,'x']))   #曼哈顿距离计算
        else:
            dis.append(-1)
    df['Edistance'] = np.array(dis)
    return df
raw_data = pd.read_csv('experimentData1D.csv')
subs = len(raw_data)
human_data = pd.DataFrame()
for i in range(subs):
    # our condition
    condition = raw_data.loc[i,'scenario']
    if condition == 1:
        continue

    # x and y
    x = eval(raw_data.loc[i,'searchHistory'])['xcollect']
    y = eval(raw_data.loc[i,'searchHistory'])['ycollect']
    y_scaled = eval(raw_data.loc[i,'searchHistory'])['ycollectScaled']
    flatten_x = [item for sublist in x for item in sublist]
    flatten_y = [item for sublist in y for item in sublist]
    flatten_y_scaled = [item for sublist in y_scaled for item in sublist]
    #environment
    env = raw_data.loc[i,'kernel']
    environment = ['rough']*len(flatten_x) if env == 0 else ['smooth']*len(flatten_x)

    #horizons
    horizon = raw_data.loc[i,'horizon']
    horizon_number = ([5,10])*8 if horizon == 0 else ([10,5])*8
    horizon = ([5]*6+[10]*11)*8 if horizon == 0 else ([10]*11+[5]*6)*8

    # kernel
    kernel_list = eval(raw_data.loc[i,'envOrder'])
    kernels = []
    for j in range(len(kernel_list)):
        kernels += [kernel_list[j]]*(horizon_number[j]+1)

    # scale
    scale_list = eval(raw_data.loc[i,'scale'])
    scales = []
    for j in range(len(scale_list)):
        scales += [scale_list[j]]*(horizon_number[j]+1)
    print(len(flatten_x),len(flatten_y),len(flatten_y_scaled),len(kernels),len(scales),len(horizon),len(environment))
    data = pd.DataFrame({'x':flatten_x,'z':flatten_y,'z_scaled':flatten_y_scaled,'kernel':kernels,'globalmax':scales,'horizons':horizon,'environment':environment})
    data['id'] = i
    human_data = pd.concat([human_data,data],ignore_index=True,axis=0)


ids = human_data['id'].unique() 
trials = []
for i in ids:
    data = human_data[human_data['id']==i]
    times = len(data)
    for j in range(times):
        if j == 0:
            trials.append(0)
        else:
            if data.iloc[j,5] == data.iloc[j-1,5]:
                trials.append(trials[-1]+1)
            else:
                trials.append(0)
human_data['trials'] = trials
human_data['agent'] = 'human'
human_data = Edis_compute(human_data)
human_data.to_csv('humandata.csv',index=False)


# 读取数据
def Edis_compute(df):
    dis, index =[-1],df.shape[0]
    for i in range(1,index):
        if df.loc[i,'grid'] == df.loc[i-1,'grid']:
            dis.append(abs(df.loc[i,'x']-df.loc[i-1,'x']))   #曼哈顿距离计算
        else:
            dis.append(-1)
    df['Edistance'] = np.array(dis)
    return df

random = pd.read_csv('../model/random_walk.csv')
random = Edis_compute(random)
random.to_csv('../model/random_walk.csv')