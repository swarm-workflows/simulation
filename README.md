# swarm-agents
We did a comparative study of various Agent Based Modeling Simulation Systems. 
This repository contains the examples/tests we conducted as part of this anaylysis.

## What are we looking for in the various modeling systems?
- Portable and Extensible
- Scalable
- Secure
- Supports plugins - describing agent behavior
- Topology creation (Inter swarm communication)
- Inbuilt Network stack communication
- Easy integration with Python code

## What are the options available?
We explored following frameworks:
- [Agent-py](https://agentpy.readthedocs.io/en/latest/) 				
- [Mesa](https://mesa.readthedocs.io/en/stable/overview.html)
- [OpenBMB/AgentVerse](https://github.com/OpenBMB/AgentVerse)
- [AutoGen](https://microsoft.github.io/autogen/)
- [GAMA](https://gama-platform.org/)
- [RePast](https://repast.github.io/)
- [Flame](https://flamegpu.com/)

## Agents
Overview of the features for the agents explored are listed below. 
Based on the analysis, AgentPy and Repast4py are the two recommended candidates that met the 
project requirements and were explored in depth. More details for AgentPy and Repast4Py can be found in the respective directories. 

### Agent-py (Recommended)
- Built-in support for creating agents, environments, and experiments.
- Interactive computing with IPython and Jupyter.
- Provides parameter sampling, Monte Carlo experiments, stochastic processes, parallel computing, and sensitivity analysis.
- Supports topologies: Spatial grid, continuous space, network
- Experiment class that supports multiple iterations, parameter samples, randomization, and parallel processing
- Tools for data arrangement and sensitivity analysis
- Very easy to use but does not have inbuilt features and relies on the implementation for features like scalability or inter agent communication

### Mesa
- Apache2 licensed agent-based modeling (or ABM) framework in Python.
- Built-in support for creating agents, environments, and experiments.
- Its goal is to be the Python 3-based counterpart to NetLogo, Repast, or MASON.
- Supports most of the features of Agent-Py except parameter sampling and limited analysis tools.
- Similar to AgentPy but has wider community support and moderate to high scalability.

### AgentVerse
- AgentVerse is a framework for simulating the collaboration of language agents for enabling LLM applications
- It requires OpenAI API key to operate (not free)
- It uses the agents to call OpenAI API to create conversation in different scenarios (classroom, prisoner-dilemma, etc)
- It does not support communication among the agents
- Support a basic observation, planning, reflection architecture for the agent/environment which is extensible

### AutoGen
- Developed by Microsoft
- Enables next-gen LLM applications based on multi-agent conversations.
- Diverse conversation patterns for complex workflows
- Provides enhanced LLM inference
- Conversational Agents; requires API Key for gpt-4
- Can be integrated with any open source LLMs as well
- Has multi-agent communication modules [1], each agent simulates one speaker

### GAMA
- Java Based open source modeling and simulation environment for creating spatially explicit agent-based simulations
- Can be invoked from Python
- Ability to connect to databases and supports multi paradigm  modeling.
- Agent behavior can be defined in GAML file.
- Provides a graphical interface for creating agent-based models - to define agents, environments, and interactions using a visual representation.
- Active Community of users and developers.
- Supports large-scale, high-performance simulations.
- Bit of learning curve w.r.t interface and GAML

### Repast
- Repast is one of the most established and widely used agent-based modeling frameworks.
- It is implemented in Java and offers support for developing agent-based models in both Java and Python.
- Repast has a strong community and a long history of use in academia and industry.
- It is versatile and can be used for various types of agent-based modeling applications.
- Uses mpi4py.MPI.*comm for inter agent communication.
- Active community support and scalable.

#### Repast-Simphony (Java)
- **Agent-Based Modeling**: Simulate interactions among autonomous agents.
- **Graphical Interface**: User-friendly design for easy model creation.
- **Spatial Modeling**: Represent spatial relationships in models.
- **Custom Behaviors**: Define unique agent rules and characteristics.
- **Experiment Design**: Set up experiments to explore model behavior.
- **Data Visualization**: Create visuals to interpret simulation results.
- **Parallel Computing**: Run simulations across multiple processors.
- **Community Support**: Active community and extensive documentation.

#### Repast4py (Python) (Recommended)
- **Python-Based Modeling**: Develop agent-based models using Python.
- **Agent Customization**: Define agent attributes and behaviors.
- **Spatial Modeling**: Simulate spatial interactions and environments.
- **Data Visualization**: Visualize and analyze simulation results in Python.
- **Integration with Repast**: Complements Repast Simphony for Python users.
- **Community Resources**: Access documentation and support from a user community.
- **Custom Behaviors**: Create custom agent actions for tailored models.
- **MPI support**.

#### RePast-HPC (C++)
- **High-Performance Computing**: Utilize parallel and distributed computing.
- **Large-Scale Simulations**: Run simulations on clusters and multi-core systems.
- **Batch Mode**: Conduct extensive experiments without a graphical interface.
- **Hybrid Modeling**: Combine agent-based modeling with other approaches.
- **Data Analysis**: Perform in-depth analysis of simulation outputs.
- **Community and Support**: Engage with a helpful user community and resources.
- **Cross-Platform Compatibility**: Works across various operating systems.

### Comparison
Feature | Agent-Py | Mesa | AgentVerse | Autogen | Gama | Repast 
--- | --- | --- | --- |--- |--- |--- 
Language | Python | Python | Python | Python | GAMA(Domain Specific language) | Java/Python/C++
Modeling Interface |  Python | Python/GUI |API | API | Graphical |Graphical/API
Spatial Modeling | Supported | Supported | Implementation dependent | Supported | Supported | Supported
Custom Behaviors | Supported | Supported | Implementation dependent | Supported | Supported | Supported
Parallel Computing | Implementation dependent | Supported | Implementation dependent | Implementation dependent | Limited | Supported
Data Visualization |  Visualization tools | Visualization tools | Implementation dependent | Implementation dependent | Real time, Visualization tools | Visualization tools
Experiment Design | Supported | Supported | Supported | Supported | Supported | Supported
Community Support |Dependent on adoption |Active community |Limited |Active community |Active community |Active community 
Data Analysis | Tools available | Tools available| Limited| Tools available
Extensibility | Custom libraries, extensions |Custom libraries, extensions|Basic, left to implementation|Custom libraries, extensions|Custom libraries, extensions|Custom libraries, extensions
Learning Curve |Beginner to Intermediate |Moderate |Moderate |Moderate |Moderate |Moderate
Scalability |Varies, implementation dependent| Moderate to High |Limited to Moderate ||Moderate |Moderate |High

