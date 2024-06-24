import numpy as np
import pandas as pd
import load as ld
from openai import AzureOpenAI

# set the gpt4 agent
# configure the openai 

'''
add your account information here
api_key = 
azure_endpoint = 
path = 'data/'
'''

client = AzureOpenAI(
  api_key = api_key,  
  api_version = "2024-02-15-preview",
  azure_endpoint = azure_endpoint
)

# init the example background
smooth_envs = ld.envir_values('smooth')
rough_envs = ld.envir_values('rough')


env_type = ['smooth','rough']
horizons = [2,1]
xs = []
ys = []
zs = []
zs_scaled = []

pt = '../exp_instructions/'
instructions_files = [pt+'experiment_eng.txt',pt+'test_eng.txt',pt+'right_eng.txt',pt+'finish_eng.txt']
instructions = []
for name in instructions_files:
  file = open(name, 'r',encoding='utf-8')
  instructions.append(file.read())
  file.close()

# temperature range
temperatures = [1]
tem_range = len(temperatures)
subs = 1; # number of subjects in each temperature

# formal study
for i in range(tem_range):
  for j in range(subs):
    # special agent
    temperature = temperatures[i]
    horizons_order = np.random.permutation(horizons).tolist()
    agent_env = env_type[np.random.randint(0,2)]
    env_values = ld.envir_values(agent_env) # 40 environments
    examples,example_index = ld.show_example(agent_env)
    experiment_index = ld.draw_numbers(example_index,8)
    maxs = ld.draw_max(len(experiment_index))
    xs = []
    ys = []
    zs = []
    zs_scaled = []
    scores = 0
    flag = 0
    # show the example
    content = instructions[0]
    print("U: " + instructions[0] + "\n" + "-" * 80 + "\n")
    # init the conversation
    response_content = "New Conversation!"
    conversation=[{"role": "system", "content": "you are my participant in the experiment, please follow my instructions to complete the experiment."}]
    conversation.append({"role":"user","content":content})

    for k in range(len(horizons_order)):
      # in the given environment
      current_horizon = horizons_order[k]
      current_env = env_values[experiment_index[k]]
      trials = -1

      while True:
        if flag == 0:
            response = client.chat.completions.create(
                model="gpt4o", # use deployment name for Azure API
                messages=conversation,
                temperature=temperature
            )

            response_content = response.choices[0].message.content
            conversation.append({"role": "assistant", "content": response_content})
        elif flag == 1:
            response_content = 'CONTINUE'
            flag = 0

        print("A: " + response_content + "\n" + "-" * 80 + "\n")
        
        #show the examples
        if response_content == 'OK':
            conversation.append({"role":"user","content":examples})
            print("A: " + examples + "\n" + "-" * 80 + "\n")
        
        #understand the examples
        if 'NEXT' in response_content:
            content = instructions[1]
            conversation.append({"role":"user","content":content})
            print("U: " + content + "\n" + "-" * 80 + "\n")
        
        # judge whether it is right
        if response_content[0] == '(':
           if len(eval(response_content)) == 3:
              if eval(response_content) == (1,1,3):
                 content = instructions[2]
                 print("U: " + content + "\n" + "-" * 80 + "\n")
                 conversation.append({"role":"user","content":content})
              else:
                 content = 'some answers are wrong. Give me the right answers with the form of (X,Y,Z)'
                 print("U: " + content + "\n" + "-" * 80 + "\n")
                 conversation.append({"role":"user","content":content})

        if response_content[0] == '[' :
          trials += 1
          choice = eval(response_content)
          x = choice[0]
          z = current_env[x]
          z_scaled = max(0,int(np.round(z*maxs[k]+np.random.randn())))
          scores += z_scaled
          xs.append(x)
          zs.append(z)
          zs_scaled.append(z_scaled)
          if trials != current_horizon:
            content = f'You have {current_horizon - trials} choices left,you choose [{x}],its value is {z_scaled}, your sum of reveived values are {scores}.Please just give me your next choice just with the form of [x].Note Previously chosed boxes can also be reselected'
          else:
             content = 'You have finished the experiment in this environment, please say CONTINUE to start experiment in a new environment'
          conversation.append({"role":"user","content":content})
          print("U: "+ content+ "\n" + "-" * 80 + "\n")        

        if response_content == 'START':
            #start experiment
            trials += 1
            x = np.random.randint(0,30)
            z = current_env[x]
            z_scaled = max(0,int(np.round(z*maxs[k]+np.random.randn())))
            scores += z_scaled
            xs.append(x)
            zs.append(z)
            zs_scaled.append(z_scaled)
            if k == 0:
                content = f'Experiment start! You have {current_horizon-trials} choices left, randomly generated coordinate is [{x}],its value is {z_scaled}, your sum of reveived values are {scores}。You are my participant,please give me your next choice just with the form of [x] (x is the coordinate)'
            else:
                content = f'A new environment experiment starts! You have {current_horizon-trials} choices left, randomly generated coordinate is [{x}],its value is {z_scaled},your sum of reveived values are {scores}。You are my participant,please give me your next choice just with the form of [x]'            
            conversation.append({"role":"user","content":content})
            print("U: "+ content+ "\n" + "-" * 80 + "\n")


        if response_content == 'CONTINUE':
           if trials != -1:
                flag = 1
                break
           else:
                #start experiment
                trials += 1
                x = np.random.randint(0,30)
                z = current_env[x]
                z_scaled = max(0,int(np.round(z*maxs[k]+np.random.randn())))
                scores += z_scaled
                xs.append(x)
                zs.append(z)
                zs_scaled.append(z_scaled)
                if k == 0:
                    content = f'Experiment start! You have {current_horizon-trials} choices left, randomly generated coordinate is [{x}],its value is {z_scaled}, your sum of reveived values are {scores}。You are my participant,please just give me your next choice with the form of [x]'
                else:
                    content = f'A new environment experiment starts! You have {current_horizon-trials} choices left, randomly generated coordinate is [{x}],its value is {z_scaled},your sum of reveived values are {scores}。You are my participant,please just give me your next choice just with the form of [x]'            
                conversation.append({"role":"user","content":content})
                print("U: "+ content+ "\n" + "-" * 80 + "\n")
    content = instructions[3]
    conversation.append({"role":"user","content":content})
    print("U: " + content + "\n" + "-" * 80 + "\n")
    response = client.chat.completions.create(
        model="gpt4o", # use deployment name for Azure API
        messages=conversation,
        temperature=temperature
    )
    response_content = response.choices[0].message.content
    print("A: " + response_content + "\n" + "-" * 80 + "\n")
    # save data
    choices = []
    env_numbers = []
    agent_horizons = []
    globalmax = []
    t = []
    for p in range(len(horizons_order)):
        current_horizon = horizons_order[p]
        env_number = experiment_index[p]
        maxi = maxs[p]
        for q in range(current_horizon+1):
           choices.append(q)
           env_numbers.append(env_number)
           agent_horizons.append(current_horizon)
           globalmax.append(maxi)
           t.append(temperature)
           
    xs = np.array(xs)+1
    zs = np.array(zs)
    zs_scaled = np.array(zs_scaled)
    env_numbers = np.array(env_numbers)
    global_max = np.array(globalmax)
    t = np.array(t)
    print(len(xs))
    print(len(zs))
    print(len(zs_scaled))
    print(len(env_numbers))
    print(len(global_max))
    print(len(t))
    print(len(choices))

    data = pd.DataFrame({'trials':choices,'x':xs,'zs':zs,'z_scaled':zs_scaled,'kernel':env_numbers,'globalmax':global_max,'t':t})
    data['environment'] = agent_env
    data.to_csv(path+f'sub{j+10}_t{temperature}.csv')

    # save strategy
    with open('strategy/'+f"sub{j+10}_t{temperature}.txt", "w",encoding='utf-8') as file:
        file.write(response_content)
        file.close()