
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
# import matplotlib.pyplot as plt
from datetime import datetime
import random
import os


# In[2]:


import sys
seed = int(sys.argv[1])
REWARD = int(sys.argv[2])
# seed = 9295
# REWARD = 10
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


# In[3]:


# Hyper Parameters
HIDDEN_LAYER        = 50
BATCH_SIZE          = 32
LR                  = 1e-4  # learning rate
EPSILON             = 0.9   # greedy policy
GAMMA               = 0.9   # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency
MEMORY_CAPACITY     = 100000
TERMINAL_BIAS       = 0.5   # no. of terminal memories in batch


# In[4]:


env = gym.make('CartPole-v0')
env.seed(seed);


# In[5]:


XTRA_FEAT   = 0 #masscart, masspole, length
N_ACTIONS   = env.action_space.n 
N_STATES    = env.observation_space.shape[0] + XTRA_FEAT
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape


# In[6]:


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)


# In[7]:


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(N_STATES, HIDDEN_LAYER)
        nn.init.kaiming_uniform_(self.fc1.weight)

        self.adv = nn.Linear(HIDDEN_LAYER, N_ACTIONS)
        nn.init.xavier_uniform_(self.adv.weight) 
    
        self.val = nn.Linear(HIDDEN_LAYER, 1)
        nn.init.xavier_uniform_(self.val.weight)
        
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        adv = self.adv(x)
        val = self.val(x)
        
        return val + adv - adv.mean()


# In[8]:


class D3QN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()

        self.learn_step_counter  = 0 # for target updating
        self.memory_counter      = 0
        self.memory              = np.zeros((int(MEMORY_CAPACITY), N_STATES * 2 + 2)) # initialize memory

        self.optimizer           = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func           = nn.MSELoss()
        
#         self.running_loss        = 0
#         self.loss_rec            = 0 
    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action
    
    def choose_greedy_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()
        action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % int(MEMORY_CAPACITY)
        self.memory[index, :] = transition
        self.memory_counter += 1
 
    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1
        
        sample_index_limit = min(MEMORY_CAPACITY, self.memory_counter)
        sample_index = np.random.choice(int(sample_index_limit),int(BATCH_SIZE))
        b_memory = self.memory[sample_index,:]
        
        b_s  = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a  = torch.LongTensor( b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r  = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval   = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        a_eval   = self.eval_net(b_s).max(1)[1].view(BATCH_SIZE, 1) #best action according to eval_net
        q_next   = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.gather(1, a_eval)   # shape (batch, 1)
        loss     = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
#         self.running_loss += loss.mean().item()
#         if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
#             self.loss_rec = np.append(self.loss_rec, self.running_loss / TARGET_REPLACE_ITER)
#             self.running_loss = 0.0


# In[9]:


dqn = D3QN()
NO_OF_EPISODES = 10000


# In[ ]:


# print('\nCollecting experience...')
TIMESTEP_LIMIT = 200
total_time_steps = 0
time_rec = []

SOLVE_METRIC = 195
SOLVE = False

state_max  = np.ones(N_STATES)*(-100)
state_min  = np.ones(N_STATES)*(100)
for i_episode in range(NO_OF_EPISODES):
    s = env.reset()
    time_steps = 0
    while True:
        state_max = np.maximum(state_max,s) 
        state_min = np.minimum(state_min,s)
        time_steps += 1
        total_time_steps += 1
        a = dqn.choose_action(s)

        # take action
        s_, r, done, info = env.step(a)
        if done:
            if time_steps > TIMESTEP_LIMIT:
                r = REWARD
            else:
                r = -REWARD
           
        dqn.store_transition(s, a, r, s_)
        
        if i_episode >= 2000:
            dqn.learn()
        
        if done:
            time_rec = np.append(time_rec, time_steps)
            break
        s = s_
    
    #if minimum of episode length of last 100 episodes is equal to or greater than upgrade_metric=195
    if time_rec[-100:].min() >= SOLVE_METRIC:
        SOLVE = True
        print("SEED: ",seed, "SUCCESS @ EP: ", i_episode )
        break

if not SOLVE:
    print("SEED: ",seed, "FAILURE")


# In[ ]:


# WINDOW = 10
# lossavg = running_mean(dqn.loss_rec, WINDOW)
# plt.plot(dqn.loss_rec,alpha=0.5,color='g')
# plt.plot(lossavg)


# In[ ]:


# WINDOW = 100
# ravg = running_mean(time_rec, WINDOW)


# fig = plt.figure(figsize = (15,6))

# ax2 = fig.add_subplot(2, 1, 1)
# data = time_rec
# ax2.plot(data, color = 'g',alpha=0.5)
# ax2.plot(np.ones_like(data)*1500, 'g--')
# ax2.plot(np.ones_like(data)*200, 'r--')

# ax2.set_xlabel('Iterations',color = 'k')
# ax2.set_ylabel('Time Steps',color = 'g')
# ax2.set_ylim([1,2.5e2])
# fig.tight_layout()
# ax2.grid()


# plt.plot(ravg)
# plt.show()


# In[ ]:


# MODELNAME = './models/BASE_F' + datetime.now().strftime("_%H_%M_%S")
# print(MODELNAME)


# In[ ]:


# torch.save(dqn.eval_net.state_dict(), MODELNAME)


# In[ ]:


# env.close()


# In[ ]:


# print("TOTAL TIMESTEPS: ", total_time_steps)


# In[ ]:


# ttest_time_rec = []
# for i_episode in range(1000):
#     env.length   = 0.5 + np.random.uniform(-0.3,0.3)
# #     xtra = [env.length]
#     s = env.reset()
# #     s = np.append(s, xtra)
#     time_steps = 0
#     while True:
# #         env.render()
#         time_steps += 1
#         a = dqn.choose_action(s)

#         # take action
#         s_, r, done, info = env.step(a)
# #         s_ = np.append(s_, xtra)

#         if done or time_steps >= TIMESTEP_LIMIT:
#             ttest_time_rec = np.append(ttest_time_rec, time_steps)
#             break
#         s = s_


# In[ ]:


# fig = plt.figure(figsize = (15,6))

# ax2 = fig.add_subplot(2, 1, 1)
# data = ttest_time_rec
# ax2.plot(data, color = 'g')
# ax2.plot(np.ones_like(data)*1500, 'g--')
# ax2.plot(np.ones_like(data)*200, 'r--')

# ax2.set_xlabel('Iterations',color = 'k')
# ax2.set_ylabel('Time Steps',color = 'g')
# ax2.set_ylim([1,2.5e2])
# fig.tight_layout()
# ax2.grid()
# plt.show()


# In[ ]:
