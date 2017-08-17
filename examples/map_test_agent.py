import argparse
import os
import gym_starcraft.envs.map_battle_env as sc
from scipy.misc import imsave
from gym_starcraft.envs.map_battle_env import MAP_SIZE
class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self):
        return self.action_space.sample()


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
    if not os.path.exists("myself"):
        os.mkdir("myself")
    if not os.path.exists("enemy"):
        os.mkdir("enemy")
    int_map_size = int(MAP_SIZE)
    while episodes < 20:
        obs = env.reset()
        done = False
        while not done:
            action = agent.act()
            obs, reward, done, info = env.step(action)
            myself, enemy, imgs = obs
            step += 1
            print(myself[0], "ep%i_step%i" %(episodes, step))
            imsave("myself/myself_ep%i_step%i.png" % (episodes, step), imgs[:,:,0].reshape((int_map_size, int_map_size)))
            imsave("enemy/enemy_ep%i_step%i.png" % (episodes, step), imgs[:,:,1].reshape((int_map_size, int_map_size)))
        episodes += 1
        step = 0
    env.close()
