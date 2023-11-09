# Examples
The various examples in the scope of SWARM project using Agentpy/Repast4py are listed below:
## `resource_allocation.py`
This is a basic resource allocation modeling example.
## `task_allocation.py`
This is a basic task allocation modeling example.
## `extended_allocator.py`
This is an extended Task Allocation example which allocates resources to various tasks using a greedy allocation algorithm. This example simulates having GPU, CPU and Disk resources available. It performs STFT computation for a GPU task, JAX computation for a CPU task and disk write for a Disk task.

## Installation
Requires `python3.10` or above. 

### Agentpy
Install the dependencies using the command:
```
sudo pip3 install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
sudo pip3 install -r agents/agent-py/requirements.txt
```
### Repast4py
It requires MPI and repast4py to be installed. Following steps can be used to install MPI and repast4py.
```
sudo apt install mpich
sudo env CC=mpicxx pip install repast4py
sudo pip3 install -r agents/repast4-py/requirements.txt
```

## Usage
- Extended allocator can be executed as below.
  - AgentPy
```
python3 -m agents.agent-py.extended_allocator --num_agents 500 --num_cpu 10000 --num_gpu 10000
```
  - Repast4Py
```
python3 -m agents.repast4-py.extended_allocator --num_agents 500 --num_cpu 10000 --num_gpu 10000
```
NOTE: For Repast4py, it may require to export the following environment variable.
```
export RDMAV_FORK_SAFE=1
```

## Performance Comparison
Steps for each of the runs below: 100

Number of Agents | AgentPy Time(s) | Repast4py Time(s)
---|---|---
100|  157.350244 | 160.388634
500|  | | 908.729727
1000| | | 1646.348326