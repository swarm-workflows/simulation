# Simulating Swarm Agents
Within this package lies a simulation for a SWARM Agent, utilizing the repast4-py framework. The simulation incorporates a fundamental distributed resource allocation algorithm, as illustrated below:

![Design Overview](./images/swarm-agent.png)

## Overview of Design
- Each host operates a Resource Agent (RA) equipped with comprehensive information about all available resources on that host.
- RA performs the following tasks:
  - The Agent with Rank 0 functions as the Leader Agent.
  - RA selects a job from the Job Queue (Kafka topic) that matches the resources available to RA.
  - RA identifies an idle Child Agent (CA), allocates resources to the CA, and marks the CA as running.
  - RA transmits resource and job information to the CA.
  - RA processes incoming Status messages from the CA to release any allocated resources and updates the CA's state.
- CA carries out the following actions:
  - The Agent with Rank > 0 operates as the Child Agent.
  - CA executes the assigned job.
  - CA sends a Status message back to RA containing state information.

![Agent Communication](./images/agent-comm.png)

## Usage
Launch the Swarm Agent on a host using the following command:
`mpiexec -n 4 python -m simulation.host_controller`

Add jobs to the Job Queue using the Job Producer:
`python job_producer.py`

## Tests
Test the MPI message exchange helper with the following command:
` mpiexec -n 2 python -m simulation.tests.message_tests`

## Limitations
While this simulation makes use of the Repast framework, it may not derive significant benefits from it and could be implemented without Repast.