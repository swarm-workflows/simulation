import enum
import time
from mpi4py import MPI
from repast4py.network import UndirectedSharedNetwork
from repast4py.schedule import Schedule, PriorityType

from simulation.job_queue import JobQueue, Job
from simulation.base_agent import BaseAgent, AgentType
from simulation.message import MessageHelper
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
        self.child_agents = {}  # List to keep track of child agents
        self.state = AgentState.IDLE

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

    def execute_job(self, job: dict):
        if self.state == AgentState.IDLE:
            self.state = AgentState.RUNNING
            job_obj = Job(resources=job.get("resources"),
                          commands=job.get("commands"))
            print(f"{self}  executing job: {job_obj}")

            try:
                total_execution_time = 0.0
                # Run the job commands sequentially
                for command in job_obj.get_commands():
                    start_time = time.time()

                    return_code = TaskExecutor.run_command(command, cpus=self.resources.get("cpus", []))

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    total_execution_time += elapsed_time

                    print(f"Command: {command} executed with return code: {return_code}, "
                          f"Elapsed Time: {elapsed_time:.2f} seconds")

                print(f"Job completed, Total Execution Time: {total_execution_time:.2f} seconds")
                print()
                print()

                self.jobs_executed += 1
            except Exception as e:
                print(f"Error executing job: {job}, Error: {e}")
            self.state = AgentState.IDLE

    def redeem_resources_from_child_agent(self, incoming_resources: dict):
        print(f"Redeeming CA resources: {incoming_resources}")
        self.resources["cpus"].extend(incoming_resources.get("cpus", []))
        self.resources["nics"].extend(incoming_resources.get("nics", []))
        self.resources["gpus"].extend(incoming_resources.get("gpus", []))

    def find_idle_child_agent(self):
        for info in self.child_agents.values():
            if info["state"] == AgentState.IDLE:
                return info

    def step(self):
        #print(f"IN {self} STEP START")
        if self.local_rank == 0:
            # Resource Agent Status update Messages
            if MPI.COMM_WORLD.Iprobe(source=MPI.ANY_SOURCE, tag=1):
                ra_info = MessageHelper.receive_message(source=MPI.ANY_SOURCE, tag=1)
                self.msgs_rcvd_cnt += 1
                child_resources = ra_info.get("resources")
                if len(child_resources):
                    self.redeem_resources_from_child_agent(incoming_resources=child_resources)
                ra_info.pop("resources")
                # Save Child Agent info in child agents
                self.child_agents[ra_info.get("rank")] = ra_info

            idle_child_agent = self.find_idle_child_agent()
            if idle_child_agent:
                job_found = JobQueue.get().find_matching_job(agent_resources=self.resources)
                if job_found:
                    child_agent_resources = self.compute_child_agent_resources(requested_resources=
                                                                               job_found.get_resources())

                    print(f"Allocated {child_agent_resources} to CA {idle_child_agent} to execute {job_found}")
                    data = {"job": job_found.to_dict(),
                            "resources": child_agent_resources}
                    JobQueue.get().schedule_job(job_found)
                    dest_rank = idle_child_agent.get("rank")
                    MessageHelper.send_message(data=data, destination=dest_rank, tag=0)
                    self.msgs_sent_cnt += 1
                    # Mark child agent as RUNNING
                    self.child_agents[dest_rank]["state"] = AgentState.RUNNING
        else:
            # Listen for any jobs to execute from Leader
            if MPI.COMM_WORLD.Iprobe(source=0, tag=0):
                incoming_msg = MessageHelper.receive_message(source=0, tag=0)
                self.msgs_rcvd_cnt += 1
                job = incoming_msg.get("job")
                self.resources = incoming_msg.get("resources")
                self.execute_job(job=job)
                # TODO inform leader of job status

            # Inform Leader of the Child Agent Rank and state
            data = {
                "agent_id": self.id,
                "rank": self.local_rank,
                "state": self.state,
                "resources": self.resources
            }
            MessageHelper.send_message(data=data, destination=0, tag=1)
            self.msgs_sent_cnt += 1
            self.resources.clear()
        #print(f"IN {self} STEP STOP")


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

