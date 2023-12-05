import agentpy as ap
import random
import concurrent.futures

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
        return "SUCCESS"


class TaskExecutor:
    @staticmethod
    def run(task: ResourceAllocator):
        task.allocate_task()


# Define the simulation model
class ResourceAllocationModel(ap.Model):
    MAX_THREADS = 10

    def setup(self):
        self.space = Space(self, (10, 10))
        self.tasks = [Task(self) for _ in range(1000)]
        self.space.add_agents(self.tasks)
        self.allocated_tasks = {}  #  Store allocated tasks
        self.resources = [ResourceAllocator(self) for _ in range(10)]
        self.space.add_agents(self.resources)

    def step(self):
        self.process_pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.MAX_THREADS)
        futures = []
        for resource in self.resources:
            #resource.allocate_task()
            future = self.process_pool.submit(TaskExecutor.run, resource)
            futures.append(future)

        concurrent.futures.wait(futures)
        for future in futures:
            future.result()


if __name__ == '__main__':

    # Run the simulation
    model = ResourceAllocationModel()
    model.run(steps=100)

    # Display the results
    for i, resource in enumerate(model.resources):
        print(f"Resource {i} allocated tasks: {[task.id for task in resource.model.allocated_tasks[resource.id]]}")

