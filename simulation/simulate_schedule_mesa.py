#!/usr/bin/env python
import mesa
from typing import List, NamedTuple, Tuple, Any


class Message(NamedTuple):
    src_uid: int
    content: Any


class AbstractMessagingAgent(mesa.Agent):
    def __init__(self, uid, model):
        super().__init__(uid, model)

        self._in_messages = []
        self._out_messages = []

    def send_message(self, dest_uid, message):
        self._out_messages.append((dest_uid, message))

    def get_received_messages(self) -> List[Message]:
        return self._in_messages

    def get_out_messages(self) -> List[Tuple[int, Any]]:
        return self._out_messages

    def clean_out_messages(self):
        self._out_messages = []

    def clean_in_messages(self):
        self._in_messages = []

    def receive_message(self, sender_uid, content):
        self._in_messages.append(Message(sender_uid, content))

    def is_running(self):
        return (len(self._in_messages) + len(self._out_messages)) > 0


class Job(NamedTuple):
    uid: int
    duration: int
    cpu_req: int


class RunningJob(NamedTuple):
    remaining: int
    job: Job


class QueueAgent(AbstractMessagingAgent):
    def __init__(self, uid, model, jobs: List[Job]):
        super().__init__(uid, model)
        self.jobs_by_id = {job.uid: job for job in jobs}

    def step(self):
        for sender, avail_capacity in self.get_received_messages():
            job = next((job for job in self.jobs_by_id.values() if job.cpu_req <= avail_capacity), None)
            if job is not None:
                self.send_message(sender, job)
                del self.jobs_by_id[job.uid]

    def is_running(self):
        return len(self.jobs_by_id) > 0


class WorkerAgent(AbstractMessagingAgent):
    def __init__(self, uid, model, capacity):
        super().__init__(uid, model)

        self.capacity = capacity
        self._running_tasks = {}
        self.step_completed_jobs = []
        self.wait = False

    def step(self):
        self.step_completed_jobs = []
        for _, job in self.get_received_messages():
            assert job.uid not in self._running_tasks
            self._running_tasks[job.uid] = RunningJob(job.duration, job)
            self.capacity -= job.cpu_req
            assert self.capacity >= 0

        for job_uid in list(self._running_tasks.keys()):
            running = self._running_tasks[job_uid]
            running = RunningJob(running.remaining - 1, running.job)
            self._running_tasks[job_uid] = running
            if self._running_tasks[job_uid].remaining == 0:
                # job complete
                self.capacity += running.job.cpu_req
                del self._running_tasks[job_uid]
                self.step_completed_jobs.append(running.job)

        if self.capacity > 0 and not self.wait:
            self.send_message(0, self.capacity)
            self.wait = True
        elif self.wait:
            self.wait = False

    def is_running(self):
        return super().is_running() or (len(self._running_tasks) > 0)

    def get_complete_jobs(self):
        return self.step_completed_jobs


class Model(mesa.Model):
    COMPLETE_JOBS_TABLE = 'complete_jobs'

    def __init__(self, node_caps: List[int], jobs: List[Job]):
        super().__init__()
        self.schedule = mesa.time.RandomActivation(self)
        self.agents_by_id = {}

        # queue
        self.agents_by_id[0] = queue = QueueAgent(0, self, jobs)
        self.schedule.add(queue)

        # nodes
        for i, node_cap in enumerate(node_caps):
            self.agents_by_id[i + 1] = WorkerAgent(i + 1, self, node_cap)
            self.schedule.add(self.agents_by_id[i + 1])

        self.datacollector = mesa.DataCollector(
            tables={
                Model.COMPLETE_JOBS_TABLE: ['completion', 'node'] + list(Job._fields)
            }
        )

    def step(self):
        self.schedule.step()

        # pass messages
        for agent in self.agents_by_id.values():
            agent.clean_in_messages()

        for sender_id, sender_agent in self.agents_by_id.items():
            for dest_id, msg in sender_agent.get_out_messages():
                self.agents_by_id[dest_id].receive_message(sender_id, msg)
            sender_agent.clean_out_messages()

        # log complete jobs
        for node in self.get_agents_of_type(WorkerAgent):
            for job in node.get_complete_jobs():
                row = job._asdict()
                row['completion'] = self.schedule.steps
                row['node'] = node.unique_id
                self.datacollector.add_table_row(Model.COMPLETE_JOBS_TABLE, row)

        self.running = any(agent.is_running() for agent in self.agents_by_id.values())


if __name__ == '__main__':
    m = Model(
        node_caps=[1] * 4,
        jobs=[Job(i, 2, 1) for i in range(10)])
    m.run_model()
    m.datacollector.get_table_dataframe(Model.COMPLETE_JOBS_TABLE).to_csv('result.csv')
