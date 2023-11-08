from datetime import datetime

from agentpy import Agent, Model, Space
import random

# Define a resource agent
class Resource(Agent):
    def setup(self):
        self.value = random.randint(1, 10)

# Define a resource allocation agent
class ResourceAllocator(Agent):
    def setup(self):
        self.resources = []

    def request_resource(self, model, resource):
        if resource in model.resources and resource not in self.resources:
            self.resources.append(resource)
            model.resources.remove(resource)
            return True
        return False

    def release_resource(self, model, resource):
        if resource in self.resources:
            self.resources.remove(resource)
            model.resources.append(resource)
            return True
        return False

# Define a resource allocation model
class ResourceAllocationModel(Model):
    def setup(self):
        self.space = Space(self, (10, 10))
        self.resources = [Resource(self) for _ in range(100000)]
        self.space.add_agents(self.resources)
        self.agents = [ResourceAllocator(self) for _ in range(5000)]

    def step(self):
        for agent in self.agents:
            if not agent.resources:
                resource = random.choice(self.resources)
                agent.request_resource(self, resource)
            else:
                resource = random.choice(agent.resources)
                agent.release_resource(self, resource)


if __name__ == '__main__':
    # Create a model and run it
    model = ResourceAllocationModel()

    begin = datetime.now()
    model.run(steps=10)

    # Access agent and resource data
    for agent in model.agents:
        print(f"After the run: Agent {agent.id} has resources: {[r.value for r in agent.resources]}")

    print(f"Total time taken for the run: {(datetime.now()-begin).total_seconds()} seconds")
