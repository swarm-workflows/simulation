# Repast4py
The various examples in the scope of SWARM project using Repast4py are listed below:
## `resource_allocation.py`
This is a basic resource allocation modeling example.
## `task_allocation.py`
This is a basic task allocation modeling example.
## `extended_allocator.py`
This is an extended Task Allocation example which allocates resources to various tasks using a greedy allocation algorithm. This example simulates having GPU, CPU and Disk resources available. It performs STFT computation for a GPU task, JAX computation for a CPU task and disk write for a Disk task.

## Pre-requisites
It requires MPI and repast4py to be installed. Following steps can be used to install MPI and repast4py.
```
sudo apt install mpich
sudo env CC=mpicxx pip install repast4py
sudo pip3 install -r requirements.txt
```

## Usage
Export the following environment variables:
```
export RDMAV_FORK_SAFE=1
```
Launch the example using the command:
```
python3 extended_allocator.py
```