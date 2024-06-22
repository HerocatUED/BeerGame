from new_env import BeerGame
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from env_cfg import Config
import numpy as np
import torch
import random


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    

def train_step(env: BeerGame):
    obs = env.reset(playType="train")
    for k in range(env.config.NoAgent):					
        if env.players[k].compType == "dqn":
            env.players[k].brain.set_init_state(obs[k])
    done = False
    # episode
    while not done:
        action = env.getAction()
        next_obs, reward, done_list, _ = env.step(action)
        for k in range(env.config.NoAgent):					
            if env.players[k].compType == "dqn":
                env.players[k].brain.train(next_obs[k], action[k], reward[k], done_list[k])
        done = all(done_list)
    return reward
  

def train(log_dir: str, model_path: str, env: BeerGame, episode_max: int = 1000):
    max_reword =  - 10
    # train
    summary_writer = SummaryWriter(log_dir=log_dir)
    for i in tqdm(range(episode_max)):
        reward = train_step(env)
        for k in range(env.config.NoAgent):
            summary_writer.add_scalar(f"train_reward_agent{k}", env.players[k].cumReward, i+1)
        # test every 20 episode
        if (i + 1) % 20 == 0:
            test_reward = test(env)
            for k in range(env.config.NoAgent):
                summary_writer.add_scalar(f"avg_test_reward_agent{k}", env.players[k].cumReward, i+1)
                if env.players[k].compType == "dqn" and test_reward[k] > max_reword:
                    max_reword = test_reward[k]
                    env.players[k].brain.save(model_path)
    summary_writer.close()
      
    
def test(env: BeerGame):
    env.test_mode = True
    reward_avg = np.array([0.0 for _ in range(env.config.NoAgent)])
    # iterate over 50 test games
    for i in range(50):
        obs = env.reset(playType="test")
        for k in range(env.config.NoAgent):					
            if env.players[k].compType == "dqn":
                env.players[k].brain.set_init_state(obs[k])
        done = False
        while not done:
            action = env.getAction()
            next_obs, reward, done_list, _ = env.step(action)
            done = all(done_list)
    reward_avg += np.array(reward)
    return reward_avg / 50


def main(log_dir: str, model_path: str, agent: int, episode_max: int = 1000):
    # initialize
    seed_everything(0)
    env = BeerGame(n_turns_per_game=100, test_mode=False)
    env.config.gameConfig = 2 + agent
    c = Config()
    c.setAgentType(env.config)
    print("AgentType:", env.config.agentTypes)
    train(log_dir, model_path, env, episode_max)


if __name__ == "__main__":
    exp = 1
    # for exp in range(1, 5):
    log_dir = f'logs/{exp}-dim128-bs64-lr0.001'
    model_path = f'logs/{exp}/model.pth'
    main(log_dir, model_path, exp)
    