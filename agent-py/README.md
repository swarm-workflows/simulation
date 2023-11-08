# Agentpy Examples
The various examples in the scope of SWARM project using Agentpy are listed below:
## `resource_allocation.py`
This is a basic resource allocation modeling example.
## `task_allocation.py`
This is a basic task allocation modeling example.
## `extended_allocator.py`
This is an extended Task Allocation example which allocates resources to various tasks using a greedy allocation algorithm. This example simulates having GPU, CPU and Disk resources available. It performs STFT computation for a GPU task, JAX computation for a CPU task and disk write for a Disk task.

## Pre-requisites
Requires `python3.10` or above. Install the dependencies using the command:
`pip3 install -r requirements.tx`

## Usage
- Extended allocator can be executed as below.
```
python3 extended_allocator.py
```
