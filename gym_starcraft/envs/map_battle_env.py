import numpy as np
from functools import reduce
from gym import spaces
# from torchcraft_py import proto
import torchcraft.Constants as tcc
import gym_starcraft.utils as utils

import starcraft_env as sc
import math
# used to draw the map
import cv2

DISTANCE_FACTOR = 16
MYSELF_NUM = 5.
ENEMY_NUM = 5.
DATA_NUM = 10
MAP_SIZE = 72.
MYSELF_COLOR = 200
NORMALIZE = False


def compute_totoal_health(units_list):
    if len(units_list):
        health = reduce(lambda x,y: x+y, [unit.health for unit in units_list])
    else:
        health = 0
    return health

# the angle is defined from the north and clockwise
# angle ranges from -1  to 1
def compute_cos_value(unit_v1, v2):
    # the product sum
    v_product = np.sum(np.multiply(unit_v1, v2))
    cos_value = v_product/np.linalg.norm(v2)
    return cos_value

# I considers that in this situation, the neural network will learn the mapping rather
# than the relationship between positions and angles.

# this data needs to be regularized
# Not consider the air temporally
def extreme(cor, ext_cor):
    ext_cor[0] = min(ext_cor[0], cor[0])
    ext_cor[1] = min(ext_cor[1], cor[1])
    ext_cor[2] = max(ext_cor[2], cor[2])
    ext_cor[3] = max(ext_cor[3], cor[3])
    return ext_cor

def resize_map(map_cor):
    center = (int((map_cor[0] + map_cor[2])/2.), int((map_cor[1] + map_cor[3])/2.))
    range  = max(map_cor[2]-map_cor[0], map_cor[3]-map_cor[1])
#    print(range, map_cor)
    scale = range/MAP_SIZE #(I still want a circle rather than an oval)

    return center, range, scale

class data_unit(object):
    def __init__(self, unit, id):
        #print unit.groundRange, "Range"
        self.id = id
        self.health = unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.groundCD
        # the type is encoded to be one-hot, but is not considered temporally
        self.type = unit.type
        self.groundATK = unit.groundATK
        self.groundRange = unit.groundRange
        self.under_attack = unit.under_attack
        self.attacking = unit.attacking
        self.moving = unit.moving
        #TODO maintain a TOP-K list
        self.die = False
        self.scale = 1.


    def update_data(self, unit):
        self.health = unit.health
        self.x = unit.x
        self.y = unit.y
        self.shield = unit.shield
        self.attackCD = unit.groundCD
        self.under_attack = unit.under_attack
        self.attacking = unit.attacking
        self.moving = unit.moving
        #if self.health <= 0: # die
        self.die = False
        return [self.x - self.groundRange, self.y - self.groundRange,
                self.x + self.groundRange, self.y + self.groundRange]

    def extract_data(self):
        # type not included
        if self.die:
            # still communicate but hope skip this one. ( convenient for experience store and replay )
            # I am afraid there will be some memory leakage using the object.
            return [0 for _ in range(DATA_NUM)]
        data = [self.x, self.y, self.health, self.shield, self.attackCD, self.groundATK, self.groundRange/self.scale,
                self.under_attack, self.attacking, self.moving]
        assert(len(data) == DATA_NUM)
        return data, int(1-self.die)

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
            self.units_dict[i] = data_unit(unit, unit.id)
        # in a fixed order
        self.id_list = sorted(self.units_dict.keys())
        self.alive_num = -1
        print(self.id_list, "id list")

    def update(self, units):
        for id in self.id_list:
            self.units_dict[id].die = True
        map_cor = [10000, 10000, 0, 0]
        self.alive_num = len(units)
        for u in units:
            id = self.id_mapping[u.id]
            unit = self.units_dict[id]
            map_cor = extreme(unit.update_data(u), map_cor)
        return map_cor

    def extract_data(self):
        data_list = []
        mask_list = []
        for id in self.id_list:
            # zero or useful information.
            data, mask = self.units_dict[id].extract_data()
            mask_list.append(mask)
            data_list.append(data)
        return [np.array(data_list), np.array(mask)]

    def draw_maps(self, center, range, scale):
        map_size = int(MAP_SIZE)
        img = np.zeros((map_size, map_size,1), dtype=np.uint8)
        for id in self.id_list:
            if self.units_dict[id].die:
                continue
            unit = self.units_dict[id]
            radius = int(unit.groundRange/scale)
            id_center = (int((unit.x - center[0])/scale + MAP_SIZE/2.), int((unit.y - center[1])/scale + MAP_SIZE/2.))
            cv2.circle(img, id_center, radius, MYSELF_COLOR, 1) # changed to one at first.

            if NORMALIZE:
                unit.x = (unit.x - center[0])*2/range # [-1, 1]
                unit.y = (unit.y - center[1])*2/range
                unit.scale = scale
        return img

    # unit is a foreign unit ( not in this dict )
    def compute_closest_position(self, unit, angle):
        # unit vector
        theta = math.radians(angle * 180)
        unit_v1 = np.array([math.cos(theta), math.sin(theta)])
        target_id = self.compute_candidate(unit, unit_v1)
        return target_id

    def compute_candidate(self, unit, unit_v1):
        target_cos = -1
        target_id = None
        # not consider the same situation
        for id in self.id_list:
            if self.units_dict[id].die:
                continue
            # TODO if the value is normalized, pay attention to change the value here.
            target_unit = self.units_dict[id]
            v2 = np.array([target_unit.x - unit.x, target_unit.y - unit.y])
            cos = compute_cos_value(unit_v1, v2)
            if cos > target_cos:
                target_cos = cos
                target_id = target_unit.id
        # None or the enemy
        return target_id



class MapBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0,
                 self_play=False, max_episode_steps=2000):
        super(MapBattleEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)
        self.myself_health = None
        self.enemy_health = None
        self.delta_myself_health = 0
        self.delta_enemy_health = 0
        self.unit_action_nb = 3
        self.action_nb = 3 * MYSELF_NUM

    # multiple actions.
    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [-1.0, -1.0, -1.0] * int(MYSELF_NUM)
        action_high = [1.0, 1.0, 1.0] * int(MYSELF_NUM)
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
        assert (len(action) == self.action_nb)
        # 15 for 5 units
        for i in range(0, int(MYSELF_NUM)):
            # Remember to mask the loss of these actions.
            if self.myself_obs_dict.units_dict[i].die:
                continue
            unit = self.myself_obs_dict.units_dict[i]
            unit_action = action[i*self.unit_action_nb:(i+1) * self.unit_action_nb]
            cmds += self.take_action(unit_action, unit)
        return cmds

    def take_action(self, action, unit):
        # attack
        cmds = []
        if unit.id is None:
            return cmds
        if action[0] >= 0:
            enemy_id = self.enemy_obs_dict.compute_closest_position(unit, action[1])
            if enemy_id is None:
                return cmds
            # TODO: compute the enemy id based on its position ( I DON'T CARE THIS POINT )

            cmds.append([tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Attack_Unit, enemy_id])
        else:
            # Move action
            degree = action[1] * 180
            distance = (action[2] + 1) * DISTANCE_FACTOR  # at most 2*DISTANCE_FACTOR
            x2, y2 = utils.get_position2(degree, distance, unit.x, unit.y)
            cmds.append([tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, int(x2), int(y2)])
        return cmds

    def _make_observation(self):
        # used to compute the rewards.
        myself_health = compute_totoal_health(self.state.units[0])
        enemy_health = compute_totoal_health(self.state.units[1])
        self.delta_enemy_health = self.enemy_health - enemy_health
        self.delta_myself_health = self.myself_health - myself_health
        self.enemy_health = enemy_health
        self.myself_health = myself_health

        # the shape needs to be modified.
        # obs = np.zeros(self.observation_space.shape)

        # I'd like to represent the distance in that hot map.
        if self.myself_obs_dict is None:
            self.myself_obs_dict = data_unit_dict(self.state.units[0], 0)

        # enemy's flag is one
        if self.enemy_obs_dict is None:
            self.enemy_obs_dict = data_unit_dict(self.state.units[1], 1)

        map_cor1 = self.myself_obs_dict.update(self.state.units[0])
        map_cor2 = self.enemy_obs_dict.update(self.state.units[1])
        map_cor = extreme(map_cor1, map_cor2)
        center, range, scale = resize_map(map_cor)

        map_myself = self.myself_obs_dict.draw_maps(center, range, scale)
        map_enemy = self.enemy_obs_dict.draw_maps(center, range, scale)
        map = np.concatenate([map_myself, map_enemy], axis=2)
        # TODO normalzie the data in each unit corresponding to the map. PPPPPPPriority HIGH.
        return [self.myself_obs_dict.extract_data(), self.enemy_obs_dict.extract_data(), map]

    # return reward as a list.
    # I need to know the range at first.
    def _compute_reward(self):
        reward = self.delta_enemy_health/ENEMY_NUM - self.delta_myself_health/MYSELF_NUM
        return reward

    def reset_data(self):
        while len(self.state.units[0]) != MYSELF_NUM or len(self.state.units[1]) != ENEMY_NUM:
            #print("state has not been loaded completely", len(self.state.units[0]),len(self.state.units[1]))
            self.client.send([])
            self.state = self.client.recv()

        self.myself_health = MYSELF_NUM * 100
        self.enemy_health = ENEMY_NUM * 100
        self.delta_myself_health = 0
        self.delta_enemy_health = 0
        self.myself_obs_dict = None
        self.enemy_obs_dict = None

    def _check_done(self):
        if self.myself_obs_dict.alive_num == 0 or self.enemy_obs_dict.alive_num == 0:
            return True
        return False
