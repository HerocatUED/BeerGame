from new_env import BeerGame
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from env_cfg import Config
import numpy as np
import torch
import random


def train(log_dir: str, model_path: str, env: BeerGame, episode_max: int = 1000):
    summary_writer = SummaryWriter(log_dir=log_dir)
    max_reword = 0
    for i in tqdm(range(episode_max)):
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
        for k in range(env.config.NoAgent):
            summary_writer.add_scalar(f"train_reward_agent{k}", reward[k], i+1)
        # test every 10 episode
        if (i + 1) % 10 == 0:
            test_reward = test(env)
            for k in range(env.config.NoAgent):
                summary_writer.add_scalar(f"avg_test_reward_agent{k}", test_reward[k], i+1)
                if env.players[k].compType == "dqn" and test_reward[k] > max_reword:
                    max_reword = test_reward[k]
                    env.players[k].brain.save(model_path)
    summary_writer.close()
    
    
def test(env: BeerGame):
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


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def main(log_dir: str, model_path: str, agent: int):
    seed_everything(0)
    c = Config()
    env = BeerGame(n_turns_per_game=100)
    env.config.gameConfig = 2 + agent
    c.setAgentType(env.config)
    print("AgentType:", env.config.agentTypes)
    train(log_dir, model_path, env, episode_max=1000)


if __name__ == "__main__":
    exp = 2
    # for exp in range(1, 5):
    log_dir = f'logs/{exp}'
    model_path = f'logs/{exp}/model.pth'
    main(log_dir, model_path, exp)
    