import time
from mpi4py import MPI
from repast4py.network import UndirectedSharedNetwork
from repast4py.schedule import Schedule, PriorityType

from simulation.job_queue import JobQueue, Job
from simulation.base_agent import BaseAgent, AgentType
from simulation.task_executor import TaskExecutor


class ResourceAgent(BaseAgent):
    def __init__(self, agent_id: int, agent_type: AgentType, rank: int, resources: dict):
        super().__init__(agent_id, agent_type, rank)
        self.resources = resources
        self.msgs_rcvd_cnt = 0
        self.msgs_sent_cnt = 0
        self.jobs_executed = 0
        self.child_agents = []  # List to keep track of child agents

    def allocate_resources(self, job: Job):
        child_agent = ResourceAgent(len(self.child_agents), AgentType.Resource, self.uid_rank, job.resources)
        self.child_agents.append(child_agent)
        return child_agent

    def find_and_execute_job(self):
        job_found = JobQueue.get().find_matching_job(agent_resources=self.resources)
        if job_found is not None:
            JobQueue.get().schedule_job(job_found)
            print(f"Executing job: {job_found}")

            try:
                total_execution_time = 0.0

                # Check if child agents need to be spawned
                if job_found.resources != self.resources:
                    child_agent = self.allocate_resources(job_found)
                    child_agent.find_and_execute_job()  # Execute the job using the child agent
                else:
                    # Run the job commands sequentially
                    for command in job_found.get_commands():
                        start_time = time.time()

                        return_code = TaskExecutor.run_command(command, cpus=self.resources.get("cpus", []))

                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        total_execution_time += elapsed_time

                        print(f"Command: {command} executed with return code: {return_code}, "
                              f"Elapsed Time: {elapsed_time:.2f} seconds")

                    JobQueue.get().mark_job_completed(job_found)
                    print(f"{job_found} completed, Total Execution Time: {total_execution_time:.2f} seconds")
                    print()
                    print()

                    self.jobs_executed += 1
            except Exception as e:
                print(f"Error executing job: {job_found}, Error: {e}")

    def step(self):
        self.find_and_execute_job()
        # Check and collect resources from child agents
        for child_agent in self.child_agents:
            child_agent.find_and_execute_job()  # Execute jobs using child agents
            # Collect resources back from child agent (assuming all resources go back to the parent)
            self.resources = child_agent.resources


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
        print(f"KOMAL --- Rank: 0 {job_queue}")
    else:
        job_queue.add_job(job4)
        print(f"KOMAL --- Rank: 1 {job_queue}")

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

