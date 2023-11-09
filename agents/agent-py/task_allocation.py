import agentpy as ap
import random

# Define a simple task class
from agentpy import Space


class Task(ap.Agent):
    def setup(self):
        self.priority = random.randint(1, 1000)

# Define a resource allocation agent
class ResourceAllocator(ap.Agent):
    def setup(self):
        pass

    def allocate_task(self):
        if self.model.tasks:
            # Find the task with the highest priority
            highest_priority_task = max(self.model.tasks, key=lambda task: task.priority)
            if self.id not in self.model.allocated_tasks:
                self.model.allocated_tasks[self.id] = []
            self.model.allocated_tasks[self.id].append(highest_priority_task)
            self.model.tasks.remove(highest_priority_task)


# Define the simulation model
class ResourceAllocationModel(ap.Model):
    def setup(self):
        self.space = Space(self, (10, 10))
        self.tasks = [Task(self) for _ in range(1000)]
        self.space.add_agents(self.tasks)
        self.allocated_tasks = {}  #  Store allocated tasks
        self.resources = [ResourceAllocator(self) for _ in range(10)]
        self.space.add_agents(self.resources)

    def step(self):
        for resource in self.resources:
            resource.allocate_task()


if __name__ == '__main__':

    # Run the simulation
    model = ResourceAllocationModel()
    model.run(steps=100)

    # Display the results
    for i, resource in enumerate(model.resources):
        print(f"Resource {i} allocated tasks: {[task.id for task in resource.model.allocated_tasks[resource.id]]}")

