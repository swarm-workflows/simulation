# Swarm Agent Simulation
This package contains a simulation for a SWARM Agent using the [repast4-py](https://repast.github.io/) framework.
It implements a basic distributed resource allocation algorithm based on the design shown below:
![Design Overview](./images/swarm-agent.png)

## Design Overview
- Each host runs a Resource Agent (RA) which has complete information of all the resources available on that host.
- RA picks a job from the Job Queue (Kafka topic) matching the resources available to RA. 
- RA finds an idle Child Agent (CA) if available or instantiates a new CA.
- RA allocates the resources to CA. 
- CA runs the job and moves to idle state and returns the resources back to RA on job completion.

## Usage
Swarm Agent on a host can be launched via:
`python -m simulation.host_controller`

Jobs can be added to the Job Queue via Job Producer:
`python job_producer.py`

## Limitations
This simulation tries to use Repast framework but doesn't benefit much from it and can be implemented without Repast.