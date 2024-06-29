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
  

def train(log_dir: str, model_path: str, env: BeerGame, episode_max: int = 1000):
    max_reword =  - 1000
    # train
    summary_writer = SummaryWriter(log_dir=log_dir)
    for i in tqdm(range(episode_max)):
        train_step(env)
        for k in range(env.config.NoAgent):
            summary_writer.add_scalar(f"train_reward_agent{k}", env.players[k].cumReward, i+1)
        # test every 20 episode
        # if (i + 1) % 20 == 0:
        #     test(env)
        #     for k in range(env.config.NoAgent):
        #         summary_writer.add_scalar(f"avg_test_reward_agent{k}", env.players[k].cumReward, i+1)
        #         if env.players[k].compType == "dqn" and env.players[k].cumReward > max_reword:
        #             max_reword = env.players[k].cumReward
        #             env.players[k].brain.save(model_path)
    summary_writer.close()
      
    
def test(env: BeerGame):
    env.test_mode = True
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


def main(log_dir: str, model_path: str, agent: int, episode_max: int = 1000):
    # initialize
    seed_everything(0)
    c = Config()
    config, unparsed = c.get_config()
    config.gameConfig = 2 + agent
    c.setAgentType(config)
    env = BeerGame(n_turns_per_game=100, test_mode=False, config=config)
    print("AgentType:", env.config.agentTypes)
    train(log_dir, model_path, env, episode_max)


if __name__ == "__main__":
    # exp = -1
    # label = 'BaseStock'
    label = 'dim64-bs64-eps0.05-long-fc3'
    for exp in range(2, 5):
        log_dir = f'logs/{exp}-{label}'
        model_path = f'logs/{exp}-{label}/model.pth'
        main(log_dir, model_path, exp, 3000)
    