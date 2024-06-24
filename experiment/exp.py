import numpy as np
import pandas as pd
import load as ld
from openai import AzureOpenAI

# set the gpt4 agent
# configure the openai 

api_key ='5b1bf2a44b7a41f99b608f6e7259286a'
azure_endpoint = 'https://0125.openai.azure.com/'
path = 'per_data/'

client = AzureOpenAI(
  api_key = api_key,  
  api_version = "2024-02-15-preview",
  azure_endpoint = azure_endpoint
)

# init the example background
smooth_envs = ld.envir_values('smooth')
rough_envs = ld.envir_values('rough')


env_type = ['smooth','rough']
horizons = [5,10]
xs = []
ys = []
zs = []
zs_scaled = []

pt = '../exp_instructions/'
instructions_files = [pt+'perexperiment_eng.txt',pt+'test_eng.txt',pt+'right_eng.txt',pt+'finish_eng.txt']
instructions = []
for name in instructions_files:
  file = open(name, 'r',encoding='utf-8')
  instructions.append(file.read())
  file.close()

# temperature range
temperatures = [0.6,0.8]
tem_range = len(temperatures)
agent_env = ['rough']
subs = 15; # number of subjects in each temperature

# formal study
for i in range(tem_range):
  for j in range(subs):
    # special agent
    temperature = temperatures[i]
    for k in range(len(agent_env)):
      env = agent_env[k]
      env_values = ld.envir_values(env) # 40 environments
      examples,example_index = ld.show_example(env)
      horizon = 10
      experiment_index = ld.draw_numbers(example_index,1)[0]
      maxs = ld.draw_max(1)[0]
      xs = []
      ys = []
      zs = []
      zs_scaled = []
      scores = 0
      flag = 0
      trials = -1
      # show the example
      content = instructions[0]
      print("U: " + instructions[0] + "\n" + "-" * 80 + "\n")
      # init the conversation
      response_content = "New Conversation!"
      conversation=[{"role": "system", "content": "you are my participant in the experiment, please follow my instructions to complete the experiment."}]
      conversation.append({"role":"user","content":content})
      current_env = env_values[experiment_index]
      while True:
          response = client.chat.completions.create(
              model="gpt4o", # use deployment name for Azure API
              messages=conversation,
              temperature=temperature
          )

          response_content = response.choices[0].message.content
          conversation.append({"role": "assistant", "content": response_content})
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
                elif eval(response_content) == (1,1,4):
                  content = 'The third answer is wrong. Give me the right answers with the form of (X,Y,Z)'
                  print("U: " + content + "\n" + "-" * 80 + "\n")
                  conversation.append({"role":"user","content":content})

          if response_content[0] == '[' :
            trials += 1
            choice = eval(response_content)
            x = choice[0]
            z = current_env[x]
            z_scaled = max(0,int(np.round(z*maxs+np.random.randn())))
            scores += z_scaled
            xs.append(x)
            zs.append(z)
            zs_scaled.append(z_scaled)
            if trials != horizon:
              content = f'you choose [{x}],its value is {z_scaled}, your sum of reveived values are {scores}. You have {horizon - trials} trials left, please just give me your next choice with the form of [x] and not including other words.Note Previously chosed boxes can also be reselected'
            else:
              content = 'You have finished the experiment in this environment, please say CONTINUE to next part'
            conversation.append({"role":"user","content":content})
            print("U: "+ content+ "\n" + "-" * 80 + "\n")        

          if response_content == 'START':
              #start experiment
              trials += 1
              x = np.random.randint(0,30)
              z = current_env[x]
              z_scaled = max(0,int(np.round(z*maxs+np.random.randn())))
              scores += z_scaled
              xs.append(x)
              zs.append(z)
              zs_scaled.append(z_scaled)
              content = f'Experiment start!Randomly generated coordinate is [{x}],its value is {z_scaled}, your sum of reveived values are {scores}. You have {horizon-trials} trials left, you are my participant,please give me your next choice just with the form of [x] (x is the coordinate) not including other words'
              conversation.append({"role":"user","content":content})
              print("U: "+ content+ "\n" + "-" * 80 + "\n")

          if response_content == 'CONTINUE':
              break
      '''
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
      '''
      choices = []
      env_numbers = []
      agent_horizons = []
      globalmax = []
      t = []
      for q in range(horizon+1):
          choices.append(q)
          env_numbers.append(experiment_index)
          agent_horizons.append(horizon)
          globalmax.append(maxs)
          t.append(temperature)
          
          
      xs = np.array(xs)+1
      zs = np.array(zs)
      zs_scaled = np.array(zs_scaled)
      env_numbers = np.array(env_numbers)
      global_max = np.array(globalmax)
      t = np.array(t)
      print(len(choices),len(xs),len(zs),len(zs_scaled),len(env_numbers),len(global_max),len(t))
      data = pd.DataFrame({'trials':choices,'x':xs,'zs':zs,'z_scaled':zs_scaled,'kernel':env_numbers,'globalmax':global_max,'t':t})
      data['id'] = j
      data['environment'] = env
      data.to_csv(path+f'sub{j+515}_t{temperature}.csv')

    '''
    # save strategy
    with open('per_strategy/'+f"sub{j}_t{temperature}.txt", "w",encoding='utf-8') as file:
        file.write(response_content)
        file.close()
    '''
        





