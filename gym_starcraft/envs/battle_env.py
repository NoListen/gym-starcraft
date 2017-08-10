import numpy as np
from functools import reduce
from gym import spaces
# from torchcraft_py import proto
import torchcraft.Constants as tcc
import gym_starcraft.utils as utils

import starcraft_env as sc

DISTANCE_FACTOR = 16
MYSELF_NUM = 5
ENEMY_NUM = 5

# this data needs to be regularized
# Not consider the air temporally
class data_unit(object):
    def __init__(self):
        self.health = None
        self.x = None
        self.y = None

        # the id may be assigned by myself.
        # an order. The id would be reordered from zero to one.
        self.id = None

        self.shield = None
        self.attackCD = None
        self.type = None
        # self.armor = 1
        self.groundRange = None
        self.groundATK = None


        # maintain a top K distance.
        # bread-first.


    # def observation(self):
        # return




class SingleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0,
                 self_play=False, max_episode_steps=2000):
        super(SingleBattleEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)
        self.myself_health = None
        self.enemy_heath = None

    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        # hit points, cooldown, ground range, is enemy, degree, distance (myself)
        # hit points, cooldown, ground range, is enemy (enemy)
        obs_low = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        obs_high = [100.0, 100.0, 1.0, 1.0, 1.0, 50.0, 100.0, 100.0, 1.0, 1.0]
        return spaces.Box(np.array(obs_low), np.array(obs_high))


    def _make_commands(self, action):
        cmds = []
        if self.state is None or action is None:
            return cmds

        myself_id = None
        myself = None
        enemy_id = None
        enemy = None
        for unit in self.state.units[0]:
            myself_id = unit.id

	# ut means what in original code.

        for unit in self.state.units[1]:
            enemy_id = unit.id

        if action[0] > 0:
            # Attack action
            if myself is None or enemy is None:
                return cmds
            # TODO: compute the enemy id based on its position
            # cmds.append(proto.concat_cmd(
            #     proto.commands['command_unit_protected'], myself_id,
            #     proto.unit_command_types['Attack_Unit'], enemy_id))
            print "HHHHHHHHHHHOLY SHIT"
            cmds.append([tcc.command_unit_protected, myself_id, tcc.unitcommandtypes.Attack_Unit, enemy_id])
        else:
            # Move action
            if myself is None or enemy is None:
                return cmds
            degree = action[1] * 180
            distance = (action[2] + 1) * DISTANCE_FACTOR
            x2, y2 = utils.get_position(degree, distance, myself.x, -myself.y)
            # cmds.append(proto.concat_cmd(
            #     proto.commands['command_unit_protected'], myself_id,
            #     proto.unit_command_types['Move'], -1, x2, -y2))
            cmds.append([tcc.command_unit_protected, myself_id, tcc.unitcommandtypes.Move, -1, x2, -y2])

        return cmds


    def _make_observation(self):
        myself = None
        enemy = None
        # Well, some units would die in the process

        myself_health = reduce(lambda x,y: x+y, [unit.health for unit in self.state.units[0]])
        enemy_health = reduce(lambda x,y: x+y, [unit.health for unit in self.state.units[1]])

        # the shape needs to be modified.
        obs = np.zeros(self.observation_space.shape)

        if myself is not None and enemy is not None:
            obs[0] = myself.health
            obs[1] = myself.groundCD
            obs[2] = myself.groundRange / DISTANCE_FACTOR - 1
            obs[3] = 0.0
            obs[4] = utils.get_degree(myself.x, -myself.y, enemy.x,
                                      -enemy.y) / 180
            obs[5] = utils.get_distance(myself.x, -myself.y, enemy.x,
                                        -enemy.y) / DISTANCE_FACTOR - 1
            obs[6] = enemy.health
            obs[7] = enemy.groundCD
            obs[8] = enemy.groundRange / DISTANCE_FACTOR - 1
            obs[9] = 1.0
        else:
            obs[9] = 1.0

        return obs

    # return reward as a list.
    # I need to know the range at first.
    def _compute_reward(self):
        reward = 0
        #too far.
        if self.obs[5] + 1 > 1.5:
            reward = -1
        # the loss of the enemy
        if self.obs_pre[6] > self.obs[6]:
            reward = 15
        # the loss of myself
        if self.obs_pre[0] > self.obs[0]:
            reward = -10
        # lose
        if self._check_done() and not bool(self.state.battle_won):
            reward = -500
        # enemy won
        if self._check_done() and bool(self.state.battle_won):
            reward = 1000
            self.episode_winfs += 1
        # stop before end.
        if self.episode_steps == self.max_episode_steps:
            reward = -500
        return reward

    def data_reset(self):
        self.myself_health = MYSELF_NUM * 100
        self.enemy_heath = ENEMY_NUM * 100
