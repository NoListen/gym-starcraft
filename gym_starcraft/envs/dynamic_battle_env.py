import numpy as np
from functools import reduce
from gym import spaces
# from torchcraft_py import proto
import torchcraft.Constants as tcc
import gym_starcraft.utils as utils
import gym_starcraft.envs.starcraft_env as sc
import math
# used to draw the map
import cv2

DISTANCE_FACTOR = 16
MYSELF_NUM = 5
ENEMY_NUM = 5
DATA_NUM = 10
MAP_SIZE = 40
#NORMALIZE = True
NORMALIZE = False
MAX_RANGE = 100
HEALTH_SCALE = 20.
TIME_SCALE = 10.
COMPLICATE_ACTION = True#False
DYNAMIC = True # Dynamic means compression

# 96 by 96
# static map at first
CROP_LT = (60, 100)
CROP_RB = (140, 180)
# times relative to the map
SCALE = (max(CROP_RB[1] - CROP_LT[1], CROP_RB[0] - CROP_LT[0]))/float(MAP_SIZE)
# The range is kind of cheating.
# So data had better only include 1.health channel 3
#  2.shield, 3
# 3. enemy (flag). or player 3
# 4. type 3
# 5. Where is myself. 1

# one image with 13 channels.
# TODO: How one AI robot learns to learn his enemy, by something like transferring the viewpoints.
def pixel_coordinates(unit):
    x0 = unit.x - unit.pixel_size_x/10. - CROP_LT[0]
    y0 = unit.y - unit.pixel_size_y/10. - CROP_LT[1]
    x1 = x0 + unit.pixel_size_x/5.
    y1 = y0 + unit.pixel_size_y/5.
    return (int(x0/SCALE), int(y0/SCALE)), (int(x1/SCALE), int(y1/SCALE))


def get_map(map_type, unit_dict_list):
    # unit locations need to be drawed into different maps.
    if map_type == "unit_location":
        unit_num = reduce(lambda x,y: x+y, [d.num for d in unit_dict_list])
        map = [np.zeros((MAP_SIZE, MAP_SIZE, 1), dtype=np.uint8) for _ in range(unit_num)]
        for d in unit_dict_list:
            map = d.get_map(map_type, map, -int(unit_num))
            unit_num -= d.num
        # unit_location shape is about Myself_num,Map_size,Map_size,1
        map = np.array(map)
    else:
        # EASY FOR concatenate
        map = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        for dict in unit_dict_list:
            map = dict.get_map(map_type, map)
    return map



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
    scale = range/float(MAP_SIZE) #(I still want a circle rather than an oval)

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
        self.groundRange = unit.groundRange # this can be inferred from training
        self.under_attack = unit.under_attack
        self.attacking = unit.attacking
        self.moving = unit.moving
        #TODO maintain a TOP-K list
        self.die = False
        self.max_health = unit.max_health
        self.max_shield = unit.max_shield
        self.pixel_size_x = unit.pixel_size_x
        self.pixel_size_y = unit.pixel_size_y

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

    def in_map(self):
        if self.x > CROP_LT[0] and self.y > CROP_LT[1] and self.x < CROP_RB[0] and self.y < CROP_RB[1]:
            return True
        return False

    def extract_data(self, center=None, range=None, scale=None):
        # type not included
        if self.die:
            # still communicate but hope skip this one. ( convenient for experience store and replay )
            # I am afraid there will be some memory leakage using the object.
            return [0]*DATA_NUM, 0.

        if center != None and range != None and scale != None:
            data = [(self.x - center[0])*2./range, (self.y - center[1])*2./range, self.health/HEALTH_SCALE, self.shield/HEALTH_SCALE, self.attackCD/TIME_SCALE, self.groundATK/HEALTH_SCALE, 
                    self.groundRange*2./range, self.under_attack, self.attacking, self.moving]
        else:
            data = [self.x, self.y, self.health/HEALTH_SCALE, self.shield/HEALTH_SCALE, self.attackCD/TIME_SCALE, self.groundATK/HEALTH_SCALE, self.groundRange,
                    self.under_attack, self.attacking, self.moving]
        assert(len(data) == DATA_NUM)
        return data, 1.

    def extract_mask(self):
        if self.die:
            return 0.
        return 1.

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
        self.num = len(self.id_list)
        #print(self.id_list, "id list")

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

    def extract_data(self, center=None, range=None, scale=None):
        data_list = []
        mask_list = np.zeros(self.num, dtype="uint8")
        if DYNAMIC:
            mask_list[:self.alive_num] = 1
        # index
        i = 0
        for id in self.id_list:
            # zero or useful information.
            #return_stuff = self.units_dict[id].extract_data()
            #print(type(return_stuff), len(return_stuff))

            # ignore this mask
            data, mask = self.units_dict[id].extract_data(center, range, scale)
            data_list.append(data)
            if not DYNAMIC:
                mask_list[i] = mask
        return np.array(data_list), mask_list

    def in_map(self):
        for id in self.id_list:
            if not self.units_dict[id].in_map():
                return False
        return True


    def extract_mask(self):
        if DYNAMIC:
            mask_list = np.zeros(self.num, dtype="uint8")
            mask_list[:self.alive_num] = 1
            return mask_list

        mask_list = []
        for id in self.id_list:
            mask = self.units_dict[id].extract_mask()
            mask_list.append(mask)
        return np.array(mask_list)   

    def get_map(self, map_type, map, idx=None):
        #well, if the map is "unit_location", the color is only one.
        id_index = -1
        if map_type == "unit_location":
            assert(idx is not None)

        for id in self.id_list:
            id_index += 1
            if self.units_dict[id].die:
                if DYNAMIC:
                    id_index -= 1
                continue

            unit = self.units_dict[id]
            p1,p2 = pixel_coordinates(unit)
            # well , the relationship between the scale of the convolution input need to be considered again.
            if map_type=="unit_location":
                # maybe 128 can be better.
                cv2.rectangle(map[idx+id_index], p1, p2, 200, -1)
            elif map_type=="health":
                if unit.max_health:
                    # I think rgb allows more complicated operations
                    color = utils.hsv_to_rgb(unit.health * 120./unit.max_health, 100, 100)
                    cv2.rectangle(map, p1, p2, color, -1)
            elif map_type=="shield":
                if unit.max_shield:
                    color = utils.hsv_to_rgb(unit.max_shield * 120./unit.max_shield, 100, 100)
                    cv2.rectangle(map, p1, p2, color, -1)
            elif map_type=="type":
                color = utils.html_color_table[unit.type]
                cv2.rectangle(map, p1, p2, color, -1)
            elif map_type=="flag":
                color = utils.players_color_table[self.flag]
                cv2.rectangle(map , p1, p2, color, -1)
            else:
                print("Sorry, the type required can't be satisfied")
        return map

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



class DynamicBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0,
                 self_play=False, max_episode_steps=1000):
        self.map_types_table = ["unit_location", "health", "shield", "type", "flag"]
        """
        ul - unit location.
        au - number of alive units
        mask - [1, 1, 1, 0, 0] 3 alive among 5 units.
        s - situation including health, shield, type, flag  
        """
        #TODO: au is not returned
        self._observation_dtype = {'ul': "unint8", 's': "uint8", 'mask':"uint8", 'au': "uint8"}

        super(DynamicBattleEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)
        self.myself_health = None
        self.enemy_health = None
        self.delta_myself_health = 0
        self.delta_enemy_health = 0
        self.nb_unit_actions = 3
        # TODO test cv2.rectangle in only one channel (unit_location)
        self.range = 0

    # multiple actions.
    def _action_space(self):
        # attack or move, move_degree, move_distance
        action_low = [[-1.0, -1.0, -1.0] for _ in range(MYSELF_NUM)]
        action_high = [[1.0, 1.0, 1.0] for _ in range(MYSELF_NUM)]
        return spaces.Box(np.array(action_low), np.array(action_high))

    @property
    def observation_shape(self):
        obs_space = self.observation_space
        # temporally not know about one scalar.
        d = {k:obs_space[k].shape for k in obs_space.keys()}
        return d

    # encapsulate it to be safe from modification
    @property
    def observation_dtype(self):
        return self._observation_dtype

    def _observation_space(self):
        # unit location
        obs_space = {}

        ul_low = np.zeros((MYSELF_NUM, MAP_SIZE, MAP_SIZE, 1), dtype=np.uint8)
        ul_high = np.ones((ENEMY_NUM, MAP_SIZE, MAP_SIZE, 1), dtype=np.uint8)
        ul_obs_space = spaces.Box(np.array(ul_low), np.array(ul_high))
        obs_space["ul"] = ul_obs_space

        map_channels = reduce(lambda x,y: x+y, [utils.map_channels_table[mt] for mt in self.map_types_table if mt != 'unit_location'])
        print("All maps occupies %i channels" % map_channels)
        map_low = np.zeros((MAP_SIZE, MAP_SIZE, map_channels), dtype = np.uint8)
        map_high = map_low + 255
        map_obs_space = spaces.Box(np.array(map_low), np.array(map_high))
        obs_space["map"] = map_obs_space

        mask_low = np.zeros(MYSELF_NUM, dtype=np.uint8)
        mask_high = np.ones(MYSELF_NUM, dtype=np.uint8)
        mask_obs_space = spaces.Box(np.array(mask_low), np.array(mask_high))
        obs_space["mask"] = mask_obs_space

        au_obs_space = spaces.Box(np.array([0]), np.array([MYSELF_NUM]))
        obs_space["au"] = au_obs_space

        assert (obs_space.keys() == self._observation_dtype.keys())
        return obs_space

    def _make_commands(self, action):
        cmds = []
        assert(len(action) == MYSELF_NUM)
        assert(len(action[0]) == self.nb_unit_actions)
        if self.state is None or action is None:
            return cmds
        # 15 for 5 units
        # ui - unit index
        ui = 0
        for i in range(0, MYSELF_NUM):
            # Remember to mask the loss of these actions.
            if self.myself_obs_dict.units_dict[i].die:
                continue
            # choose the alive one
            unit = self.myself_obs_dict.units_dict[i]
            if DYNAMIC:
                # DYNAMIC means [a1, a2, a3, NULL, NULL]
                unit_action = action[ui]
                ui += 1
            else:
                # STATIC means [a1, NULL, a2, NULL, a3]
                unit_action = action[i]
            cmds += self.take_action(unit_action, unit)
        return cmds

    def take_action(self, action, unit):
        # attack
        cmds = []
        if unit.id is None:
            return cmds
        if action[0] >= 0:
            if COMPLICATE_ACTION:
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

        #TODO center and scale are useless temporally
        map_cor1 = self.myself_obs_dict.update(self.state.units[0])
        map_cor2 = self.enemy_obs_dict.update(self.state.units[1])
        map_cor = extreme(map_cor1, map_cor2)
        center, range, scale = resize_map(map_cor)

        assert("unit_location" in self.map_types_table)
        unit_dict_list = [self.myself_obs_dict, self.enemy_obs_dict]
        unit_locations = get_map('unit_location', [self.myself_obs_dict]) # 1
        maps = []
        for mt in self.map_types_table:
            if mt == 'unit_location':
                continue
            maps.append(get_map(mt, unit_dict_list))
        total_maps = np.concatenate(maps, axis=2) # 2
        self.range = range

        obs = {}
        obs["ul"] = unit_locations
        obs["s"] = total_maps
        obs["mask"] = self.myself_obs_dict.extract_mask() # shape(MYNUM,)
        obs["au"] = [self.myself_obs_dict.alive_num] # shape(1,)
        assert(obs.keys() == self._observation_dtype.keys())
        # TODO normalzie the data in each unit corresponding to the map. PPPPPPPriority HIGH.
        return obs

    # return reward as a list.
    # I need to know the range at first.
    def _compute_reward(self):
        # if self.range > MAX_RANGE or self.episode_steps == self.max_episode_steps:
        if not self.myself_obs_dict.in_map() or self.episode_steps == self.max_episode_steps:
                return -100./HEALTH_SCALE
        reward = self.delta_enemy_health/float(ENEMY_NUM) - self.delta_myself_health/float(MYSELF_NUM)
        reward = reward/HEALTH_SCALE
        return reward

    def reset_data(self):
        while len(self.state.units) == 0 or len(self.state.units[0]) != MYSELF_NUM or len(self.state.units[1]) != ENEMY_NUM:
            #print("state has not been loaded completely", len(self.state.units[0]),len(self.state.units[1]))
            self.client.send([])
            self.state = self.client.recv()
        self.range = 0
        self.myself_health = MYSELF_NUM * 100.
        self.enemy_health = ENEMY_NUM * 100.
        self.delta_myself_health = 0
        self.delta_enemy_health = 0
        self.myself_obs_dict = None
        self.enemy_obs_dict = None
        self.advanced_termination = True

    def _check_win(self):
        if self.myself_health/float(MYSELF_NUM) >= self.enemy_health/float(ENEMY_NUM):
            self.episode_wins += 1

    def _check_done(self):
        if self.myself_obs_dict.alive_num == 0 or self.enemy_obs_dict.alive_num == 0:
            self._check_win()
            return True
        if not self.myself_obs_dict.in_map() or self.episode_steps >= self.max_episode_steps:
            self._check_win()
            self.advanced_termination = True
            return True
        return False
