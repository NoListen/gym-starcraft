import argparse
import os
import gym_starcraft.envs.dynamic_battle_env as sc
from scipy.misc import imsave
from gym_starcraft.envs.dynamic_battle_env import MAP_SIZE
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


map_types_table = ["health", "shield", "type", "flag"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', help='server ip')
    parser.add_argument('--port', help='server port', type=int,default=11111)
    args = parser.parse_args()

    env = sc.MapBattleEnv(args.ip, args.port)
    env.seed(123)
    agent = RandomAgent(env.action_space)

    episodes = 0
    step = 0
    if not os.path.exists("obs"):
        os.mkdir("obs")

    for p in map_types_table:
        if not os.path.exists("obs/"+p):
            os.mkdir("obs/"+p)

    int_map_size = int(MAP_SIZE)
    while episodes < 20:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act()
            obs, reward, done, info = env.step(action)
            _, imgs, _ = obs
            c = imgs.shape[-1]
            step += 1
            for i in range(0,c,3):
                img = imgs[...,i:i+3]
                imsave("obs/"+map_types_table[i//3]+"/ep%i_step%i.png" % (episodes, step), img)
        episodes += 1
        step = 0
    env.close()
