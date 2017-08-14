import numpy as np
from functools import reduce
from gym import spaces
# from torchcraft_py import proto
import torchcraft.Constants as tcc
import gym_starcraft.utils as utils

import starcraft_env as sc

# used to draw the map
import cv2

DISTANCE_FACTOR = 16
MYSELF_NUM = 5.
ENEMY_NUM = 5.
DATA_NUM = 10
MAP_SIZE = 72.
MYSELF_COLOR = 1
NORMALIZE = False


# this data needs to be regularized
# Not consider the air temporally
def extreme(cor, ext_cor):
    ext_cor[0] = min(ext_cor[0], cor[0])
    ext_cor[1] = min(ext_cor[1], cor[1])
    ext_cor[2] = max(ext_cor[2], cor[2])
    ext_cor[3] = max(ext_cor[3], cor[3)
    return ext_cor

def resize_map(map_cor):
    center = (int((map_cor[0] + map_cor[2])/2.), int((map_cor[1] + map_cor[3])/2.))
    range  = max(map_cor[2]-map_cor[0], map_cor[3]-map_cor[1])
    scale = range/MAP_SIZE #(I still want a circle rather than an oval)

    return center, range, scale

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
        self.moving = unit.moving
        #TODO maintain a TOP-K list
        self.die = False


    def update_data(self, unit):
        self.health  unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.attackCD
        self.under_attack = unit.under_attack
        self.attacking = unit.attacking
        self.moving = unit.moving
        if self.health <= 0: # die
            self.die = True
        return (self.x - self.groundRange, self.y - self.groundRange, self.x + self.groundRange, self.y + self.groundRange)

    def extract_data(self):
        # type not included
        if self.die:
            # still communicate but hope skip this one. ( convenient for experience store and replay )
            # I am afraid there will be some memory leakage using the object.
            return [0 for _ in range(DATA_NUM)]
        data = [self.x, self.y, self.health, self.shield, self.attackCD, self.groundATK, self.groundRange,
                self.under_attack, self.attacking, self.moving]
        assert(len(data) == DATA_NUM)
        return data

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
        # in a fixed order
        self.id_list = sorted(self.units_dict.keys())

    def update(self, units):
        map_cor = 10000, 10000, 0, 0
        for u in units:
            id = self.id_mapping[u.id]
            extreme(self.units_dict[id].update_date(u), map_cor)
        return map_cor

    def extract_data(self):
        data_list = []

        for id in self.id_list:
            # zero or useful information.
            data_list.append(self.units_dict[id].extract_data())
        return np.array(data_list)

    def draw_maps(self, center, range, scale):
        map_size = int(MAP_SIZE)
        img = np.zeros((map_size, map_size,1), dtype=np.uint8)
        for id in self.id_list:
            if self.units_dict[id].die:
                continue
            unit = self.units_dict[id]
            radius = int(unit.radius/scale)
            id_center = (int((unit.x - center[0])/scale), int((unit.y - center[1])/scale))
            cv2.circle(img, id_center, radius, MYSELF_COLOR, -1) # changed to one at first.

            if NORMALIZE:
                unit.x = (unit.x - center[0])*2/range # [-1, 1]
                unit.y = (unit.y - center[1])*2/range
        return img


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
        obs_low = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        #            x      y      health shield CD    ATK     range  under_attack attacking moving
        obs_high = [400.0, 300.0, 100.0, 100.0, 100.0, 100.0, 100.0, 1.0, 1.0, 1.0]
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

        map_cor1 = self.myself_obs_dict.update(self.state.units[0])
        map_cor2 = self.enemy_obs_dict.update(self.state.units[1])
        map_cor = extreme(map_cor1, map_cor2)
        center, range, scale = resize_map(map_cor)

        map_myself = self.myself_obs_dict.draw_maps(center, range, scale)
        map_enemy = self.enemy_obs_dict.draw_maps(center, range, scale)
        map = np.concatenate([map_myself, map_enemy], axis=2)
        return [self.myself_obs_dict.extract_data(), self.enemy_obs_dict.extract_data(), map]

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