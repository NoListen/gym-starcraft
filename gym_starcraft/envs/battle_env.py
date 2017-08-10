import numpy as np
from functools import reduce
from gym import spaces
# from torchcraft_py import proto
import torchcraft.Constants as tcc
import gym_starcraft.utils as utils

import starcraft_env as sc

DISTANCE_FACTOR = 16
MYSELF_NUM = 5.
ENEMY_NUM = 5.

# this data needs to be regularized
# Not consider the air temporally
class data_unit(object):
    def __init__(self, unit):
        self.health = unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.attackCD
        # the type is encoded to be one-hot, but is not considered temporally
        self.type = unit.type
        self.groundATK = unit.groundATK
        self.groundRange = unit.groundRange
        self.under_attack = unit.under_attack
        self.attacking = unit.attacking
        #TODO maintain a TOP-K list



    def update_data(self, unit):
        self.health  unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.attackCD
        self.under_attack = unit.under_attack
        self.attacking = unit.attacking
        if self.health <= 0: # die
            return False
        return True

    def extract_data(self):
        # type not included
        return [self.x, self.y, self.health, self.shield, self.attackCD, self.groundATK, self.groundRange,
                self.under_attack, self.attacking]

class data_unit_dict(object):
    # Do not consider those died.
    def __init__(self, units, flag):
        self.units_dict = {}
        self.id_mapping = {}
        # myself 0. enemy 1.
        self.flag = flag
        for i in range(len(units)):
            unit = units[i]
            self.id_mapping[unit.id] = i
            # use the idx to choose the input order | maybe not necessary
            self.units_dict[i] = data_unit(unit)

    def update(self, units):
        for u in units:
            id = self.id_mapping[u.id]
            alive = self.units_dict[id].update_date(u)
            if not alive:
                del(self.units_dict[id])

    def extract_data(self):
        data_list = []
        for u in units:
            data_list.append(u.extract_data())
        return np.array(data_list)



class SingleBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0,
                 self_play=False, max_episode_steps=2000):
        super(SingleBattleEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)
        self.myself_health = None
        self.enemy_health = None
        self.delta_myself_health = 0
        self.delta_enemy_health = 0

    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0]
        action_high = [1.0, 1.0, 1.0]
        return spaces.Box(np.array(action_low), np.array(action_high))

    def _observation_space(self):
        # hit points, cooldown, ground range, is enemy, degree, distance (myself)
        # hit points, cooldown, ground range, is enemy (enemy)
        # obs_low = [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        obs_low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #            x      y      health shield CD    ATK     range  under_attack
        obs_high = [400.0, 300.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0]
        # obs_high = [100.0, 100.0, 1.0, 1.0, 1.0, 50.0, 100.0, 100.0, 1.0, 1.0]
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
            myself = unit
            myself_id = unit.id

        # ut means what in original code.

        for unit in self.state.units[1]:
            enemy = unit
            enemy_id = unit.id

        if action[0] > 0:
            # Attack action
            if myself is None or enemy is None:
                return cmds
            # TODO: compute the enemy id based on its position
            # cmds.append(proto.concat_cmd(
            #     proto.commands['command_unit_protected'], myself_id,
            #     proto.unit_command_types['Attack_Unit'], enemy_id))
            cmds.append([tcc.command_unit_protected, myself_id, tcc.unitcommandtypes.Attack_Unit, enemy_id])
        else:
            # Move action
            if myself is None or enemy is None:
                return cmds
            degree = action[1] * 180
            distance = (action[2] + 1) * DISTANCE_FACTOR
            x2, y2 = utils.get_position(degree, distance, myself.x, myself.y)
            cmds.append([tcc.command_unit_protected, myself_id, tcc.unitcommandtypes.Move, -1, int(x2), int(y2)])
        return cmds


    def _make_observation(self):
        # used to compute the rewards.
        myself_health = reduce(lambda x,y: x+y, [unit.health for unit in self.state.units[0]])
        enemy_health = reduce(lambda x,y: x+y, [unit.health for unit in self.state.units[1]])
        self.delta_enemy_health = self.enemy_health - enemy_health
        self.delta_myself_health = self.myself_health - myself_health
        self.enemy_health = enemy_health
        self.myself_health = myself_health

        # the shape needs to be modified.
        # obs = np.zeros(self.observation_space.shape)

        # I'd like to represent the distance in that hot map.
        if self.myself_obs_dict is None:
            self.myself_obs_dict = data_unit_dict(self.state.units[0])

        if self.enemy_obs_dict is None:
            self.enemy_obs_dict = data_unit_dict(self.state.units[1])

        self.myself_obs_dict.update(self.state.units[0])
        self.enemy_obs_dict.update(self.state.units[1])


        return [self.myself_obs_dict.extract_data(), self.enemy_obs_dict.extract_data()]

    # return reward as a list.
    # I need to know the range at first.
    def _compute_reward(self):
        reward = self.delta_enemy_health/ENEMY_NUM - self.delta_myself_health/MYSELF_NUM
        return reward

    def data_reset(self):
        self.myself_health = MYSELF_NUM * 100
        self.enemy_health = ENEMY_NUM * 100
        self.delta_myself_health = 0
        self.delta_enemy_health = 0
        self.myself_obs_dict = None
        self.enemy_obs_dict = None