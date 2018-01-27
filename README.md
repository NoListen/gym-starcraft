# Modification
It has been adapted to [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) and [BWAPI v 4.1.2](https://github.com/bwapi/bwapi/releases). It doesn't need torchcraft-py anymore because [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) supports python API.

# Compound Battel Env

The observations are returned as one dictionary.

- ul     unit location.
- ud     unit data
- au     number of alive units
- mask   [1, 1, 1, 0, 0] 3 alive among 5 units.
- s      2D scene map including health, shield, type, flag, unit data [used for convolution network]

Two types of rewards

- total reward for one camp (Set UNIT_REWARD to False)
- unit reward for each unit (Set UNIT_REWARD to True)

# BiCNet

It's compatitle with the algorithm in [multi-ddpg](https://github.com/NoListen/RL-forest/tree/master/RL_forest/ddpg_plant/multi_ddpg).

# gym-starcraft
Gym StarCraft is an environment bundle for OpenAI Gym. It is based on [Facebook's TorchCraft](https://github.com/TorchCraft/TorchCraft), which is a bridge between Torch and StarCraft for AI research.

## Installation

1. Install [OpenAI Gym](https://github.com/openai/gym) and its dependencies.

2. Install [TorchCraft](https://github.com/TorchCraft/TorchCraft) and its dependencies. You can skip the torch client part. 

3. Install [torchcraft-py](https://github.com/deepcraft/torchcraft-py) and its dependencies.

4. Install the package itself:
    ```
    git clone https://github.com/deepcraft/gym-starcraft.git
    cd gym-starcraft
    pip install -e .
    ```

## Usage
1. Start StarCraft server with BWAPI by Chaoslauncher.

2. Run examples:

    ```
    cd examples
    python random_agent.py --ip $server_ip --port $server_port 
    ```
    
    The `$server_ip` and `$server_port` are the ip and port of the server running StarCraft.   
    
