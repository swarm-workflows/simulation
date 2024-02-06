import enum
import time
from mpi4py import MPI
from repast4py.network import UndirectedSharedNetwork
from repast4py.schedule import Schedule, PriorityType

from simulation.job_queue import JobQueue, Job
from simulation.base_agent import BaseAgent, AgentType
from simulation.task_executor import TaskExecutor


class AgentState(enum.Enum):
    RUNNING = enum.auto()
    IDLE = enum.auto()


class ResourceAgent(BaseAgent):
    def __init__(self, agent_id: int, agent_type: AgentType, rank: int, resources: dict):
        super().__init__(agent_id, agent_type, rank)
        self.resources = resources
        self.msgs_rcvd_cnt = 0
        self.msgs_sent_cnt = 0
        self.jobs_executed = 0
        self.child_agents = []  # List to keep track of child agents
        self.state = AgentState.IDLE
        self.job = None

    def compute_child_agent_resources(self, requested_resources: dict):
        # Get the requested resources
        cpus_requested = requested_resources.get("cpus", 0)
        nics_requested = requested_resources.get("nics", 0)
        gpus_requested = requested_resources.get("gpus", 0)

        # Check if requested resources are greater than available resources
        if cpus_requested > len(self.resources.get("cpus", [])) or \
                nics_requested > len(self.resources.get("nics", [])) or \
                gpus_requested > len(self.resources.get("gpus", [])):
            return None  # or handle the case according to your requirements

        # Allocate requested resources
        requested_resources = {
            "cpus": self.resources.get("cpus", [])[:cpus_requested],
            "nics": self.resources.get("nics", [])[:nics_requested],
            "gpus": self.resources.get("gpus", [])[:gpus_requested]
        }

        # Update available resources
        self.resources["cpus"] = self.resources.get("cpus", [])[cpus_requested:]
        self.resources["nics"] = self.resources.get("nics", [])[nics_requested:]
        self.resources["gpus"] = self.resources.get("gpus", [])[gpus_requested:]

        return requested_resources

    def allocate_resources(self, job: Job):
        child_resources = self.compute_child_agent_resources(requested_resources=job.get_resources())
        child_agent = ResourceAgent(len(self.child_agents), AgentType.Resource, self.uid_rank, child_resources)
        child_agent.job = job
        self.child_agents.append(child_agent)
        return child_agent

    def find_and_allocate_jobs(self):
        # Recover resources back from Idle agents
        # Iterate through child agents and recover resources from IDLE agents
        for child_agent in self.child_agents:
            if child_agent.state == AgentState.IDLE:
                # Add recovered resources back to the parent agent
                self.resources["cpus"].extend(child_agent.resources.get("cpus", []))
                self.resources["nics"].extend(child_agent.resources.get("nics", []))
                self.resources["gpus"].extend(child_agent.resources.get("gpus", []))

                # Remove the idle child agent from the list
                self.child_agents.remove(child_agent)
                print(f"Recovered resources from IDLE agent: {child_agent}")

        # Find a Matching Job
        job_found = JobQueue.get().find_matching_job(agent_resources=self.resources)
        if job_found is not None:
            try:
                print(f"Trying to allocate job: {job_found}")
                self.allocate_resources(job_found)
                JobQueue.get().schedule_job(job_found)
            except Exception as e:
                print(f"Error allocating job: {job_found}, Error: {e}")

    def execute_job(self):
        if self.job is not None and self.state == AgentState.IDLE:
            self.state = AgentState.RUNNING
            print(f"Executing job: {self.job}")

            try:
                total_execution_time = 0.0
                # Run the job commands sequentially
                for command in self.job.get_commands():
                    start_time = time.time()

                    return_code = TaskExecutor.run_command(command, cpus=self.resources.get("cpus", []))

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    total_execution_time += elapsed_time

                    print(f"Command: {command} executed with return code: {return_code}, "
                          f"Elapsed Time: {elapsed_time:.2f} seconds")

                JobQueue.get().mark_job_completed(self.job)
                print(f"{self.job} completed, Total Execution Time: {total_execution_time:.2f} seconds")
                print()
                print()

                self.jobs_executed += 1
            except Exception as e:
                print(f"Error executing job: {self.job}, Error: {e}")
            self.job = None
            self.state = AgentState.IDLE

    def step(self):
        self.find_and_allocate_jobs()
        for c in self.child_agents:
            print(f"Executing child: {c}")
            c.execute_job()


if __name__ == '__main__':

    scripts_dir = "/root/swarm-agents/agents/jobs/"

    num_agents = 5
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a job queue
    job_queue = JobQueue.get()

    # Add jobs to the queue
    job1 = Job(resources={"cpus": 1, "gpus": 0, "nic_cards": 0}, commands=[f"python3 -m agents.jobs.cpu_computation"])
    job2 = Job(resources={"cpus": 2, "gpus": 0, "nic_cards": 0}, commands=[f"python3 -m agents.jobs.cpu_computation"])
    job3 = Job(resources={"cpus": 2, "gpus": 0, "nic_cards": 1}, commands=[f"python3 {scripts_dir}/server.py localhost 12345 1"])
    job4 = Job(resources={"cpus": 2, "gpus": 0, "nic_cards": 1}, commands=[f"python3 {scripts_dir}/client.py localhost 12345"])
    job_queue.add_job(job1)
    job_queue.add_job(job2)
    if rank == 0:
        job_queue.add_job(job3)
    else:
        job_queue.add_job(job4)

    resources = [{"cpus": [1, 2], "gpus": [], "nic_cards": ["localhost"]},
                 {"cpus": [1, 2], "gpus": [], "nic_cards": ["localhost"]},
                 {"cpus": [1, 2], "gpus": [], "nic_cards": ["localhost"]},
                 {"cpus": [1, 2], "gpus": [], "nic_cards": ["localhost"]},
                 {"cpus": [1, 2], "gpus": [], "nic_cards": ["localhost"]}]
    agents = [ResourceAgent(i, AgentType.Resource, rank, resources[i]) for i in range(rank, 5, size)]
    network = UndirectedSharedNetwork('resource_agent_nw', comm)
    network.add_nodes(agents=agents)

    scheduler = Schedule()
    for agent in agents:
        scheduler.schedule_repeating_event(1.0, 1.0, lambda agent=agent: agent.step(),
                                           priority_type=PriorityType.RANDOM)

    steps = 10
    for _ in range(steps):
        scheduler.execute()

    comm.Barrier()

    # Print the state of jobs in the queues
    print("Pending Jobs:", [str(job) for job in job_queue.pending_queue])
    print("Scheduled Jobs:", [str(job) for job in job_queue.scheduled_queue])
    print("Completed Jobs:", [str(job) for job in job_queue.completed_queue])

