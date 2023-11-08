from mpi4py import MPI
from repast4py import core, random, space, schedule, logging, parameters
import random as rr


# Define a resource agent
class Resource(core.Agent):
    TYPE = 0

    def __init__(self, agent_id, rank):
        super().__init__(agent_id, Resource.TYPE, rank)
        self.value = rr.randint(1, 10)


# Define a ResourceAgent
class ResourceAgent(core.Agent):
    TYPE = 0

    def __init__(self, agent_id, rank):
        super().__init__(agent_id, ResourceAgent.TYPE, rank)
        self.resources = []

    def request_resource(self, model, resource: Resource):
        if resource in model.resources and resource not in self.resources:
            self.resources.append(resource)
            model.resources.remove(resource)
            return True
        return False

    def release_resource(self, model, resource: Resource):
        if resource in self.resources:
            self.resources.remove(resource)
            model.resources.append(resource)
            return True
        return False


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

        self.resources = [Resource(x, rr.randint(1, 100000)) for x in range(100000)]
        self.agents = []

        for i in range(10):
            agent = ResourceAgent(i, rr.randint(1, 10))
            self.space.add(agent)
            self.agents.append(agent)  # Add the agent to the list first
            pt = space.ContinuousPoint(rr.randint(1, 100), rr.randint(1, 200))
            self.space.move(agent, pt)  # Then, move the agent to the space

    def step(self):
        for agent in self.agents:
            if not agent.resources:
                resource = rr.choice(self.resources)
                agent.request_resource(self, resource)
            else:
                resource = rr.choice(agent.resources)
                agent.release_resource(self, resource)

    def start(self):
        self.runner.execute()

if __name__ == '__main__':
    # Create a model and run it
    model = ResourceAllocationModel()
    model.start()

# Access agent and resource data
    for agent in model.agents:
        print(f"After the run: Agent {agent.id} has resources: {[r.value for r in agent.resources]}")