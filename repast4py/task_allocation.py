from mpi4py import MPI
from repast4py import core, random, space, schedule, logging, parameters
import random as rr


# Define a resource agent
class Task(core.Agent):
    TYPE = 0

    def __init__(self, agent_id, rank):
        super().__init__(agent_id, Task.TYPE, rank)
        self.priority = rr.randint(1, 1000)


# Define a ResourceAgent
class ResourceAllocator(core.Agent):
    TYPE = 0

    def __init__(self, model, agent_id, rank):
        super().__init__(agent_id, ResourceAllocator.TYPE, rank)
        self.model = model

    def allocate_task(self):
        if self.model.tasks:
            # Find the task with the highest priority
            highest_priority_task = max(self.model.tasks, key=lambda task: task.priority)
            if self.id not in self.model.allocated_tasks:
                self.model.allocated_tasks[self.id] = []
            self.model.allocated_tasks[self.id].append(highest_priority_task)
            self.model.tasks.remove(highest_priority_task)


# Define a ResourceAllocationModel
class ResourceAllocationModel:
    def __init__(self):
        comm = MPI.COMM_WORLD
        # create the schedule
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(100)

        box = space.BoundingBox(xmin=0, xextent=100, ymin=0, yextent=200, zmin=0, zextent=0)
        self.space = space.SharedCSpace('ResourceSpace', bounds=box, borders=space.BorderType.Sticky,
                                        occupancy=space.OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)

        self.tasks = [Task(x, rr.randint(1, 1000)) for x in range(1000)]
        for x in self.tasks:
            self.space.add(x)

        self.allocated_tasks = {}

        self.resources = [ResourceAllocator(self, x, rr.randint(1, 10)) for x in range(10)]
        for x in self.resources:
            self.space.add(x)

    def step(self):
        for resource in self.resources:
            resource.allocate_task()

    def start(self):
        self.runner.execute()


if __name__ == '__main__':

    # Run the simulation
    model = ResourceAllocationModel()
    model.start()

    # Display the results
    for i, resource in enumerate(model.resources):
        print(f"Resource {i} allocated tasks: {[task.id for task in resource.model.allocated_tasks[resource.id]]}")

