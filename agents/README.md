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
sudo apt update -y
sudo apt-get -y install python3-pip
sudo pip3 install jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
sudo apt install mpich
sudo env CC=mpicxx pip3 install repast4py
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
500|  837.300738 | 908.729727
1000| 1704.914721 | 1750.930586


### Multi-Threaded/Processing environment
**Assumptions**:
- Resources available: 100 CPU tokens and 100 GPU tokens
- Each process requests one of the either kind of tokens

**Task Distribution:**
75% CPU tasks and 25% GPU tasks

** VM Profile **
16 cores, 128GB RAM

**Observations**:

Comparable performance observed in multi-threaded or multi-processing environment. 

Number of Agents | AgentPy Time(s) | Repast4py Time(s)
---|---|--- 
1000| 404.92798 | 414.211947
5000| 1786.703973 | 1588.871533
10000| 3370.322972 | 3242.823398
25000| 8871.172184 | 8448.093424
50000| Failed as VM ran out of memory for JAXB calculations | Failed as VM ran out of memory for JAXB calculations

#### Agentpv vs Repast4py
Features|Agentpy|Repast4Py
---|---|---
Last updated| Dec 2, 2022 <br>Github: https://github.com/jofmi/agentpy | Oct 31, 2023 <br>Github: https://github.com/Repast/repast4py
Performance | Comparable | Comparable
Metrics/Logging | No built-in support| Provides built in metric/log collection for each agent
Multi-Processing | No built-in support; user can extend using python `concurent.futures.multiprocessing` | MPI Support
Messaging | No built-in support; can be added using python libraries | MPI messaging framework is available
Hierarchical | No built-in support to define hierarchical agents | Inbuilt using MPI levels
Learning Cure | Easy to use, lightweight and flexible | Bit of learning curve but has comprehensive set ot tools for building, simulating and analyzing agent based models

**Recommendation**: Repast4py is the recommended option. Considering the performance numbers are similar, Repast4py provides inbuilt features such as metric collection, MPI support which would come in handy during simulations and experimentation. 