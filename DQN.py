import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class DQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class DQNAgent:
    def __init__(self, state_size, action_size,  
                 hidden_size=256, gamma=0.99, lr=0.001, 
                 batch_size=64, epsilon=0.05, history=1,
                 memory_size=1000):
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon

        # memory pool: (state, action, reward, next_state, done)
        self.memory = []
        self.memory_size = memory_size
        self.history = history
        self.state = None

        self.policy_net = DQNetwork(state_size, hidden_size, action_size)
        self.target_net = DQNetwork(state_size, hidden_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.iteration = 1
        self.update_iter = 100

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
    
    def getDNNAction(self, playType):        
        if playType == "train":
            if (random.random() <= self.epsilon) or (len(self.memory) < self.history):
                action = random.randrange(self.action_size)
            else:
                state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action = torch.argmax(self.policy_net(state)).item()
        elif playType == "test":
            state = torch.tensor(self.state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = torch.argmax(self.policy_net(state)).item()
        return action

    def replay(self):
        """Single Training Step"""
        if len(self.memory) < self.batch_size:
            return
        # sample from memory
        batch = random.sample(self.memory, self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.tensor(np.array(state_batch), dtype=torch.float32)
        action_batch = torch.tensor(action_batch, dtype=torch.int64).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, dtype=torch.float32)
        next_state_batch = torch.tensor(np.array(next_state_batch), dtype=torch.float32)
        done_batch = torch.tensor(done_batch, dtype=torch.float32)

        # current Q-value
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # target Q-value
        next_q_values = self.target_net(next_state_batch).max(1)[0]
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)

        # optimize policy net work
        loss = F.mse_loss(q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.iteration += 1
        if self.iteration % self.update_iter == 0:
            self.update_target_network()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
    def train(self, next_obs, action, reward, done):
        # Considering the multi-period observation idea, merges the last m-1 periods with the new state. 
        # next_state = torch.stack([self.state[1:], [next_obs]], axis = 0)
        next_state = next_obs
        self.remember(self.state, action, reward, next_state, done)
        self.replay()
        self.state = next_state
        # if terminal and state == "train":
        #     self.epsilonReduce()
        
    def set_init_state(self, obs=None):
        if obs is None:
            self.state = np.zeros(self.state_size)
        else:
            self.state = np.array(obs, dtype=torch.float32)
        # self.state = torch.stack([torch.tensor(obs) for _ in range(memory_length)], axis = 0)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))