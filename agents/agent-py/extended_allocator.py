import argparse
from datetime import datetime
import agentpy as ap
import random

from ..common.resource import Resource, ResourceType, ResourceTypeToActionMap


# Define a Task Agent that can request different types of resources and has a priority
class TaskAgent(ap.Agent):
    def setup(self, resource_type: ResourceType):
        self.priority = random.randint(1, 100)  # Higher number means higher priority
        self.requested_resource_type = resource_type
        self.done = False

    def perform_task(self):
        start = datetime.now()
        action = ResourceTypeToActionMap[self.requested_resource_type]()
        action.perform(batch_size=64)
        self.release_resource()
        #print(f"Task: {self.id} took time: {(datetime.now()-start).total_seconds()} to complete")

    def has_resources(self):
        return self.allocated_resource is not None

    def is_done(self):
        return self.done

    def allocate_resource(self, resource: Resource):
        if self.done or self.requested_resource_type != resource.resource_type:
            return
        self.allocated_resource = resource
        resource.allocate(self)

    def release_resource(self):
        self.allocated_resource.release()
        self.done = True


# Define the main Model
class ResourceAllocationModel(ap.Model):
    def __init__(self, parameters=None, _run_id=None, num_cpu: int = 100, num_gpu: int = 50,
                 num_nvme: int = 10, num_agents: int = 100, **kwargs):
        super(ResourceAllocationModel, self).__init__(parameters, _run_id)

        # Create resources
        self.resources = {
            ResourceType.CPU: [Resource(ResourceType.CPU) for _ in range(num_cpu)],
            ResourceType.GPU: [Resource(ResourceType.GPU) for _ in range(num_gpu)]
        }

        # Create tasks with priorities
        self.tasks = [TaskAgent(self, random.choice(list(ResourceType))) for _ in range(num_agents)]

    def get_available_resource(self, resource_type: ResourceType):
        for resource in self.resources[resource_type]:
            if resource.is_available:
                return resource
        return None  # If no available resource is found

    def setup(self):
        pass

    def step(self):

        # Sort tasks by priority (higher first)
        # Allocate tasks using a greedy algorithm based on priority

        allocated_tasks = []

        sorted_tasks = sorted(self.tasks, key=lambda t: t.priority, reverse=True)
        for task in sorted_tasks:
            if task.is_done():
                continue

            available_resource = self.get_available_resource(resource_type=task.requested_resource_type)
            if available_resource:
                task.allocate_resource(resource=available_resource)
                allocated_tasks.append(task)

        for t in allocated_tasks:
            t.perform_task()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resource Allocation Model Parameters')
    parser.add_argument('--num_agents', type=int, default=100, help='Number of task agents')
    parser.add_argument('--num_gpu', type=int, default=50, help='Number of GPU resources')
    parser.add_argument('--num_cpu', type=int, default=100, help='Number of CPU resources')
    parser.add_argument('--num_nvme', type=int, default=10, help='Number of NVMe resources')
    parser.add_argument('--steps', type=int, default=100, help='Number of steps to run the model')

    args = parser.parse_args()
    model = ResourceAllocationModel(num_cpu=args.num_cpu, num_gpu=args.num_gpu, num_nvme=args.num_nvme,
                                    num_agents=args.num_agents)
    results = model.run(steps=args.steps)
    # Results analysis here
