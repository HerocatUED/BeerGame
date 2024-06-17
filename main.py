from new_env import BeerGame
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train(log_dir, model_path, MAX = 1500):
    env = BeerGame(n_turns_per_game=100)
    summary_writer = SummaryWriter(log_dir=log_dir)
    for i in tqdm(range(MAX)):
        obs = env.reset(playType="train")
        done = False
        while not done:
            action = env.getAction()
            next_obs, reward, done_list, _ = env.step(action)
            for k in range(env.config.NoAgent):					
                if env.players[k].compType == "dqn":
                    env.players[k].brain.train(next_obs[k], action[k], reward[k], done_list[k])
            done = all(done_list)
            # cumReward = [env.players[i].cumReward for i in range(env.config.NoAgent)]
        for k in range(env.config.NoAgent):
            summary_writer.add_scalar(f"reward{k}", reward[k], i)

    for k in range(env.config.NoAgent):
        if env.players[k].compType == "dqn":
            env.players[k].brain.save(model_path)
            
    summary_writer.close()
    print("Game Over")
    

if __name__ == "__main__":
    exp = 2
    log_dir = f'BeerGame_demo/logs/{exp}'
    model_path = f'BeerGame_demo/logs/{exp}/model.pth'
    train(log_dir, model_path)
    
# TODO: current state