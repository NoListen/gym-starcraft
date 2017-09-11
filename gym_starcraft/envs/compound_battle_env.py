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

DISTANCE_FACTOR = 25
MYSELF_NUM = 3
ENEMY_NUM = 1
DATA_NUM = 8
MAP_SIZE = 40
HEALTH_SCALE = 20.
TIME_SCALE = 10.
COMPLICATE_ACTION = True#False
DYNAMIC = False # Dynamic means compression
TARGET_VAR = 1000000
UNIT_REWARD = True
K = 4
ENEMY_DATA_NUM = 2
MARGIN = 10
ENEMY_COMMAND = True
# 96 by 96
# static map at first
#CROP_LT = (10, 10)
CROP_LT = (60, 120)
#CROP_RB = (240, 240)
CROP_RB = (100, 140)
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
    if map_type == "unit_location" or "unit_density":
        unit_num = reduce(lambda x,y: x+y, [d.num for d in unit_dict_list])
        map = [np.zeros((MAP_SIZE, MAP_SIZE, 1), dtype=np.uint8) for _ in range(unit_num)]
        for d in unit_dict_list:
            map = d.get_map(map_type, map, -int(unit_num))
            unit_num -= d.num
        # unit_location shape is about Myself_num,Map_size,Map_size,1
        map = np.array(map)
        if map_type == "unit_density":
            map = np.sum(map, axis=0)
    else:
        # EASY FOR concatenate
        map = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)
        for dict in unit_dict_list:
            map = dict.get_map(map_type, map)
    return map


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
        self._delta_health = 0

    def update_data(self, unit):
        self._delta_health = self.health - unit.health
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

    def in_map(self):
        if self.x >= CROP_LT[0]-MARGIN and self.y >= CROP_LT[1]-MARGIN and self.x <= CROP_RB[0]+MARGIN and self.y <= CROP_RB[1]+MARGIN:
            return True
        return False

    @property
    def delta_health(self):
        return self._delta_health


    def extract_data(self, unit_dict):
        # type not included
        if self.die:
            # still communicate but hope skip this one. ( convenient for experience store and replay )
            # I am afraid there will be some memory leakage using the object.
            #return [100, 140, 0, 0, 0, self.groundATK/HEALTH_SCALE, self.groundRange, 0, 0, 0]+[100, 1, 0, 0]*ENEMY_NUM, 0.
            #return [0, 0, 0, self.groundATK/HEALTH_SCALE, self.groundRange, 0, 0, 0]+[100, 1, 0, 0]*ENEMY_NUM, 0.
            return [ 0,0, 0, self.groundATK/HEALTH_SCALE, self.groundRange, 0, 0, 0]+[-1, 1]*ENEMY_NUM, 0.

        #data = [self.x, self.y, self.health/HEALTH_SCALE, self.shield/HEALTH_SCALE, self.attackCD/TIME_SCALE, self.groundATK/HEALTH_SCALE, self.groundRange,
        data = [self.health/HEALTH_SCALE, self.shield/HEALTH_SCALE, self.attackCD/TIME_SCALE, self.groundATK/HEALTH_SCALE, self.groundRange,
                 self.under_attack, self.attacking, self.moving]
        assert(len(data) == DATA_NUM)
        data += unit_dict.get_distance_degree(self)
        #data = [self.x, self.y, self.health / HEALTH_SCALE, self.shield / HEALTH_SCALE, self.under_attack, self.attacking, self.moving]
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
        self.alive_num = len(units)
        for u in units:
            id = self.id_mapping[u.id]
            unit = self.units_dict[id]
            unit.update_data(u)

    def get_distance_degree(self, unit):
        dd = []
        for id in self.id_list:
            t = self.units_dict[id]
            distance = utils.get_distance(unit.x, unit.y, t.x, t.y)
            degree = utils.get_degree(unit.x, unit.y, t.x, t.y)/180.
            #dd += [distance, t.die, t.groundRange, degree]
            dd += [distance, degree]
        return dd
            

    def extract_data(self, unit_dict):
        mask_list = np.zeros(self.num, dtype="uint8")
        data_list = np.zeros((self.num, DATA_NUM+ENEMY_DATA_NUM*ENEMY_NUM), dtype="float32")
        i = 0
        for id in self.id_list:
            if DYNAMIC and self.units_dict[id].die:
                continue
            data_list[i], mask_list[i] = self.units_dict[id].extract_data(unit_dict)
            i += 1
        return data_list, mask_list

    def in_map(self):
        for id in self.id_list:
            unit = self.units_dict[id]
            if unit.die:
                continue

            if not self.units_dict[id].in_map():
                return False
        return True

    def extract_mask(self):
        mask_list = np.zeros(self.num, dtype="uint8")
        i = 0
        for id in self.id_list:
            unit = self.units_dict[id]
            # skip recording the data
            if unit.die and DYNAMIC:
                continue
            mask_list[i] = self.units_dict[id].extract_mask()
            # mask_list[i] = mask
            i += 1
        return mask_list

    def get_map(self, map_type, map, idx=None):
        #well, if the map is "unit_location", the color is only one.
        id_index = 0
        if map_type == "unit_location":
            assert(idx is not None)

        for id in self.id_list:
            if self.units_dict[id].die and DYNAMIC:
                continue

            unit = self.units_dict[id]
            p1,p2 = pixel_coordinates(unit)
            # well , the relationship between the scale of the convolution input need to be considered again.
            if map_type=="unit_location":
                # maybe 128 can be better.
                cv2.rectangle(map[idx+id_index], p1, p2, 1, -1)
            elif map_type=="unit_density":
                cv2.rectangle(map[idx+id_index], p1, p2, 1, -1)
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
                print(map_type)
                print("Sorry, the type required can't be satisfied")
            # increase the index
            id_index += 1
        return map

    # unit is a foreign unit ( not in this dict )
    def compute_closest_position(self, unit, action):
        # unit vector
        degree = action[1] * 180
        # TODO consider to change the target range.
        distance = (action[2] + 1) * unit.groundRange/2.  # at most 2*DISTANCE_FACTOR
        tx, ty = utils.get_position2(degree, distance, unit.x, unit.y)
        # TODO only consider those within the groundRange
        # limit the distance between target and target unit
        target_id = self.compute_candidate(tx, ty)
        return target_id

    def compute_candidate(self, tx, ty):
        target_id = None
        # set it to be a big positive integer
        d = 90000
        # not consider the same situation
        for id in self.id_list:
            if self.units_dict[id].die:
                continue
            # TODO if the value is normalized, pay attention to change the value here.
            target_unit = self.units_dict[id]
            td = (target_unit.x - tx)**2 + (target_unit.y - ty)**2
            if td < d and td < TARGET_VAR:
                target_id = target_unit.id
                d = td
        # None or the enemy
        return target_id

    def compute_single_unit_rewards(self, k, unit_dict):
        idx = 0
        rewards = np.zeros(self.num, dtype="float32")
        for id in self.id_list:
            unit = self.units_dict[id]
            if unit.die:
                if not DYNAMIC:
                    idx += 1
                continue
            rewards[idx] = utils.top_k_enemy_reward(k, unit, unit_dict)
            idx += 1
        return np.array(rewards)

    def compute_unit_rewards(self, k, unit_dict_list):
        idx = 0
        rewards = np.zeros(self.num, dtype="float32")
        for id in self.id_list:
            unit = self.units_dict[id]
            if unit.die:
                if not DYNAMIC:
                    idx += 1
                continue
                
            rewards[idx] = utils.unit_top_k_reward(k, unit, unit_dict_list)
            idx += 1
        return np.array(rewards)


# by default, use all the maps.
class CompoundBattleEnv(sc.StarCraftEnv):
    def __init__(self, server_ip, server_port, speed=0, frame_skip=0,
                 self_play=False, max_episode_steps=1000,
                 map_types_table=("unit_location", "unit_density", "unit_data", "health", "shield", "type", "flag")):
        self.map_types_table = map_types_table
        self.obs_cls = list(set(utils.obs_cls_table[k] for k in self.map_types_table)) + ["mask", "au"]
        """
        ul - unit location.
        ud - unit data
        au - number of alive units
        mask - [1, 1, 1, 0, 0] 3 alive among 5 units.
        s - situation including health, shield, type, flag, unit data
        """
        #TODO: au is not returned

        super(CompoundBattleEnv, self).__init__(server_ip, server_port, speed,
                                              frame_skip, self_play,
                                              max_episode_steps)
        self.myself_health = None
        self.enemy_health = None
        self.delta_myself_health = 0
        self.delta_enemy_health = 0
        self.nb_unit_actions = 3
        # TODO test cv2.rectangle in only one channel (unit_location)

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
        obs_space = self.observation_space
        d = {k:utils.obs_dtype[k] for k in obs_space.keys()}
        return d

    @property
    def reward_shape(self):
        if UNIT_REWARD:
            return (MYSELF_NUM,)
        else:
            return (1,)

    def _observation_space(self):
        # unit location
        obs_space = {}

        if "ul" in self.obs_cls:
            ul_low = np.zeros((MYSELF_NUM, MAP_SIZE, MAP_SIZE, 1), dtype=np.uint8)
            ul_high = np.ones((ENEMY_NUM, MAP_SIZE, MAP_SIZE, 1), dtype=np.uint8)
            ul_obs_space = spaces.Box(np.array(ul_low), np.array(ul_high))
            obs_space["ul"] = ul_obs_space

        if "s" in self.obs_cls:
            global_map_types = [mt for mt in self.map_types_table if ( mt != "unit_location" and mt != "unit_data")]
            map_channels = reduce(lambda x,y: x+y, [utils.map_channels_table[mt] for mt in global_map_types])
            print("All maps occupies %i channels" % map_channels)
            map_low = np.zeros((MAP_SIZE, MAP_SIZE, map_channels), dtype = np.uint8)
            map_high = map_low + 255
            map_obs_space = spaces.Box(np.array(map_low), np.array(map_high))
            obs_space["s"] = map_obs_space

        if "ud" in self.obs_cls:
            unit_data_low = np.zeros((MYSELF_NUM, DATA_NUM+ENEMY_NUM*ENEMY_DATA_NUM), dtype=np.float32)
            unit_data_low[:,-1] = -1.
            #unit_data_high = np.array([[400, 300, 100, 100, 100, 100, 100, 1.0, 1.0, 1.0]+[400, 1., 100,  1.]*ENEMY_NUM]*MYSELF_NUM, dtype=np.float32)
            unit_data_high = np.array([[ 100, 100, 100, 100, 100, 1.0, 1.0, 1.0]+[400, 1.]*ENEMY_NUM]*MYSELF_NUM, dtype=np.float32)
            unit_obs_space = spaces.Box(np.array(unit_data_low), np.array(unit_data_high))
            obs_space["ud"] = unit_obs_space

        mask_low = np.zeros(MYSELF_NUM, dtype=np.uint8)
        mask_high = np.ones(MYSELF_NUM, dtype=np.uint8)
        mask_obs_space = spaces.Box(np.array(mask_low), np.array(mask_high))
        obs_space["mask"] = mask_obs_space

        au_obs_space = spaces.Box(np.array(0), np.array(MYSELF_NUM))
        obs_space["au"] = au_obs_space

        assert (set(obs_space.keys()) == set(self.obs_cls))
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

        #if ENEMY_COMMAND:
        #    for i in range(0, ENEMY_NUM):
        #        unit = self.enemy_obs_dict.units_dict[i]
        #        if unit.die:
        #            continue
                #myself_id = self.myself_obs_dict.compute_closest_position(unit, [1, 0, 0])
                #if myself_id:
                #    print("Attack %i" % myself_id)
                # cmds.append([tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Attack_Unit, myself_id])
         #       cmds.append([tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, 100, 100])
        return cmds

    def take_action(self, action, unit):
        # attack
        cmds = []
        if unit.id is None:
            return cmds
        if action[0] >= 0:
            if COMPLICATE_ACTION:
                enemy_id = self.enemy_obs_dict.compute_closest_position(unit, action)
                if enemy_id is None:
                    return cmds
            # TODO: compute the enemy id based on its position ( I DON'T CARE THIS POINT )
                cmds.append([tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Attack_Unit, enemy_id])
        else:
            #print("MOVE", unit.id, uniy.x, unit.y)
            # Move action
            degree = action[1] * 180
            distance = (action[2] + 1) * DISTANCE_FACTOR  # at most 2*DISTANCE_FACTOR
            x2, y2 = utils.get_position2(degree, distance, unit.x, unit.y)
            x2 = min(max(int(x2), CROP_LT[0]), CROP_RB[0])
            y2 = min(max(int(y2), CROP_LT[1]), CROP_RB[1])
            cmds.append([tcc.command_unit_protected, unit.id, tcc.unitcommandtypes.Move, -1, int(x2), int(y2)])
        return cmds

    def _make_observation(self):

        # the shape needs to be modified.
        # obs = np.zeros(self.observation_space.shape)

        # I'd like to represent the distance in that hot map.
        if self.myself_obs_dict is None:
            self.myself_obs_dict = data_unit_dict(self.state.units[0], 0)

        # enemy's flag is one
        if self.enemy_obs_dict is None:
            self.enemy_obs_dict = data_unit_dict(self.state.units[1], 1)

        #TODO center and scale are useless temporally
        self.myself_obs_dict.update(self.state.units[0])
        self.enemy_obs_dict.update(self.state.units[1])

        #assert("unit_location" in self.map_types_table)
        unit_dict_list = [self.myself_obs_dict, self.enemy_obs_dict]
        #unit_locations = get_map('unit_location', [self.myself_obs_dict]) # 1
        maps = []

        obs = {}
        if "ul" in self.obs_cls:
            unit_locations = get_map('unit_location', [self.myself_obs_dict]) # 1
            obs["ul"] = unit_locations
        if "s" in self.obs_cls:
            for mt in self.map_types_table:
                if mt == 'unit_location' or mt == "unit_data":
                    continue
                maps.append(get_map(mt, unit_dict_list))
            total_maps = np.concatenate(maps, axis=2) # 2
            obs["s"] = total_maps
        if "ud" in self.obs_cls:
            obs["ud"], obs["mask"] = self.myself_obs_dict.extract_data(self.enemy_obs_dict)
        else:
            obs["mask"] = self.myself_obs_dict.extract_mask()
        #obs["au"] = [self.myself_obs_dict.alive_num] # shape(1,)
        if DYNAMIC:
            obs["au"] = self.myself_obs_dict.alive_num
        else:
            obs["au"] = MYSELF_NUM
        #assert(obs.keys() == self._observation_dtype.keys())
        # TODO normalzie the data in each unit corresponding to the map. PPPPPPPriority HIGH.
        return obs

    # return reward as a list.
    # I need to know the range at first.
    def _compute_reward(self):
        unit_dict_list = [self.myself_obs_dict, self.enemy_obs_dict]
        if UNIT_REWARD:
            #reward = self.myself_obs_dict.compute_single_unit_rewards(K, self.enemy_obs_dict)
            reward = self.myself_obs_dict.compute_unit_rewards(K, unit_dict_list)
        else:
            reward = utils.total_reward(unit_dict_list)
        #print("#",reward,"#")
        return reward/HEALTH_SCALE#-0.02

    def reset_data(self):
        while len(self.state.units) == 0 or len(self.state.units[0]) != MYSELF_NUM or len(self.state.units[1]) != ENEMY_NUM:
            #print("state has not been loaded completely", len(self.state.units[0]),len(self.state.units[1]))
            self.client.send([])
            self.state = self.client.recv()
        self.myself_obs_dict = None
        self.enemy_obs_dict = None
        self.advanced_termination = True

    def _check_win(self):
        if  self.enemy_obs_dict.alive_num > 0:
            return False
        self.episode_wins += 1
        return True

    def _check_done(self):
        if self.myself_obs_dict.alive_num == 0 or self.enemy_obs_dict.alive_num == 0:
            self._check_win()
            return True
        #if self.episode_steps >= self.max_episode_steps:
        if not self.myself_obs_dict.in_map() or self.episode_steps >= self.max_episode_steps:
            print("time done")
            self._check_win()
            self.advanced_termination = True
            return True
        return False
