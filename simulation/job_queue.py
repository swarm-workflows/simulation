import enum
from typing import List
import threading
from enum import Enum


class JobState(Enum):
    PENDING = enum.auto()
    SCHEDULED = enum.auto()
    DONE = enum.auto()

    def __str__(self):
        return self.name


class Job:
    def __init__(self, resources: dict, commands: List[str]):
        self.resources = resources
        self.commands = commands
        self.state = JobState.PENDING

    def get_commands(self) -> List[str]:
        return self.commands

    def get_resources(self) -> dict:
        return self.resources

    def get_state(self) -> JobState:
        return self.state

    def set_state(self, state: JobState):
        self.state = state

    def __str__(self):
        return f"[Job: {self.commands} with Resources: {self.resources}]"

    def to_dict(self):
        return {"resources": self.resources,
                "commands": self.commands,
                "state": self.state}


class JobQueue:
    _instance = None  # Class variable to store the singleton instance
    _lock = threading.Lock()  # Lock for thread safety

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super(JobQueue, cls).__new__(cls)
                cls._instance.pending_queue = []
                cls._instance.scheduled_queue = []
                cls._instance.completed_queue = []
        return cls._instance

    @staticmethod
    def get():
        return JobQueue()

    def add_job(self, job: Job):
        self.pending_queue.append(job)

    def schedule_job(self, job: Job):
        self.pending_queue.remove(job)
        self.scheduled_queue.append(job)
        job.set_state(JobState.SCHEDULED)

    def find_matching_job(self, agent_resources: dict) -> Job:
        for job in self.pending_queue:
            if self.check_resource_availability(job, agent_resources):
                return job

    def check_resource_availability(self, job, agent_resources: dict) -> bool:
        for resource, required_amount in job.get_resources().items():
            if resource not in agent_resources or len(agent_resources[resource]) < required_amount:
                return False
        return True

    def mark_job_completed(self, job: Job):
        self.scheduled_queue.remove(job)
        self.completed_queue.append(job)
        job.set_state(JobState.DONE)


if __name__ == '__main__':
    # Example usage:

    # Create a job queue (Singleton instance) using the static get() method
    job_queue = JobQueue.get()

    # Add jobs to the pending queue
    job1 = Job(resources={"cpus": 4, "gpus": 1, "nic_cards": 2}, commands=["python script1.py"])
    job2 = Job(resources={"cpus": 2, "gpus": 0, "nic_cards": 1}, commands=["python script2.py"])
    job3 = Job(resources={"cpus": 8, "gpus": 2, "nic_cards": 1}, commands=["sh shell_script.sh"])
    job_queue.add_job(job1)
    job_queue.add_job(job2)
    job_queue.add_job(job3)

    # Example agent resources
    agent_resources = {"cpus": 6, "gpus": 1, "nic_cards": 3}

    # Find a matching job for the agent
    matching_job = job_queue.find_matching_job(agent_resources)

    if matching_job:
        print(f"Found a matching job: {matching_job}")
        # Schedule the job
        job_queue.schedule_job(matching_job)
        print(f"Scheduled job: {matching_job}")
    else:
        print("No matching job found for the agent.")

    job_queue.mark_job_completed(matching_job)

    # Print the state of jobs in the queues
    print("Pending Jobs:", [str(job) for job in job_queue.pending_queue])
    print("Scheduled Jobs:", [str(job) for job in job_queue.scheduled_queue])
    print("Completed Jobs:", [str(job) for job in job_queue.completed_queue])
