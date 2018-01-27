# Modification
It has been adapted to [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) and [BWAPI v 4.1.2](https://github.com/bwapi/bwapi/releases). It doesn't need torchcraft-py anymore because [Torchcraft v 1.3.0](https://github.com/TorchCraft/TorchCraft/releases) supports python API.

# Compound Battel Env

The observations are returned as one dictionary.
- **ul**    unit location.
- **ud**    unit data
- **au**    number of alive units
- **mask**  [1, 1, 1, 0, 0] 3 alive among 5 units.
- **s**     2D scene map including health, shield, type, flag, unit data [used for convolution network]

Two types of rewards

- total reward for one camp (Set UNIT_REWARD to False)
- unit reward for each unit (Set UNIT_REWARD to True)

# BiCNet

It's compatitle with the algorithm in [multi-ddpg](https://github.com/NoListen/RL-forest/tree/master/RL_forest/ddpg_plant/multi_ddpg).

# Instruction on Windows
1. install StarCraft
    * StarCraft (C:/ recommended)
    * BroodWar extension
2. install [BWAPI (v 4.1.2)](https://github.com/bwapi/bwapi)
3. intall TorchCraft（v 1.3.0）
    1. TorchCraft acts as a server on Windows
    2. Refer to the [installation](https://github.com/TorchCraft/TorchCraft/blob/master/docs/user/installation.md) where I chose “TorchCraft AIClient (DLL) for users”.

# Instruction on Ubuntu

```
mkdir RL
cd RL
git clone https://github.com/nolisten/RL-forest
git clone https://github.com/TorchCraft/TorchCraft
```

**install Torch**
```
sudo apt-get curl
curl -s https://raw.githubusercontent.com/torch/ezinstall/master/install-deps | bash
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch; ./install.sh
```
the bash file is used to activate torch
```
echo ". /home/larryeye/RL/torch/install/bin/torch-activate" > torch-activate.sh
```
check 
```
export LD_LIBRARY_PATH=/path/to/torch/pkg/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/path/to/torch/install/include:$LD_LIBRARY_PATH
```
in `.bashrc`

**install TorchCraft**
```
git clone https://github.com/torchcraft/torchcraft.git --recursive
cd torchcraft
luarocks make *.rockspec
```
note: if you meet zmq.h problem
```
sudo add-apt-repository ppa:chris-lea/zeromq
sudo apt-get update
sudo apt-get install libzmq3-dev
```
note: if you meet zstd problem
refer  to the [instruction](http://progur.com/2016/09/how-to-install-and-use-zstd-facebook.html)
version 1.3.0+ is recommended
```
  wget https://github.com/facebook/zstd/archive/v1.3.1.tar.gz
  tar -xzvf v1.3.1.tar.gz
  cd zstd-1.3.1
```
  note: add -fPIC behind both CFLAGS and CXXFLAGS in /lib/Makefile
```
  sudo make install
```
 
**install TorchCraft(py)**
```
cd py
pip install -e.
```


**install gym**
```
git clone https://github.com/openai/gym.git
cd gym
pip install -e .
```

**install gym-starcraft**
```
git clone https://github.com/nolisten/gym-starcraft （我在阿里的简单环境包装上做的拓展和修改）
cd gym-starcraft
pip install -e .
```


**some other packages**
```
pip3 install opencv-python
pip3 install tqdm
```


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
    
