#!/usr/bin/env python3
import csv
import simpy
from typing import Any, List, NamedTuple

class MessageBroker:
    def __init__(self, env: simpy.Environment, n_queues: int):
        self.queues = [simpy.Store(env) for _ in range(n_queues)]

    def send(self, dest: int, msg: Any):
        self.queues[dest].put(msg)

    def get_queue(self, queue: int) -> simpy.Store:
        return self.queues[queue]

class Job(NamedTuple):
    uid: int
    duration: int
    cpu_req: int

class NoMatchingJobs:
    pass

class RunningJob(NamedTuple):
    ends_at: int
    job: Job


def queue_agent(env: simpy.Environment, broker: MessageBroker, uid: int, jobs: List[Job]):
    jobs_by_id = {job.uid: job for job in jobs}
    while True:
        sender, avail_capacity = yield broker.get_queue(uid).get()
        job = next((job for job in jobs_by_id.values() if job.cpu_req <= avail_capacity), None)
        if job is not None:
            broker.send(sender, job)
            del jobs_by_id[job.uid]
        elif len(jobs_by_id) > 0:
            broker.send(sender, NoMatchingJobs())
            
def worker_agent(env: simpy.Environment, broker: MessageBroker, uid: int, capacity: int, log_store: List[List[int]]):
    running_tasks = {}
    sent = False
    received_no_matching = False
    while True:
        if capacity > 0 and not sent:
            broker.send(0, (uid, capacity))
            sent = True

        # wait for response or completion
        with broker.get_queue(uid).get() as wait_msg:
            if len(running_tasks) > 0:
                min_end_time = min([rj.ends_at for rj in running_tasks.values()])
                min_wait_time = max(0, min_end_time - env.now)
                yield wait_msg | env.timeout(min_wait_time)
            else:
                yield wait_msg
        
            if wait_msg.triggered:
                msg = wait_msg.value
                if isinstance(msg, Job):
                    running_tasks[msg.uid] = RunningJob(env.now + msg.duration, msg)
                    capacity -= msg.cpu_req
                    sent = False
                elif isinstance(msg, NoMatchingJobs):
                    received_no_matching = True # postpone flag cleanup

        released = False
        for running_job in list(running_tasks.values()):
            if running_job.ends_at <= env.now:
                del running_tasks[running_job.job.uid]
                capacity += running_job.job.cpu_req
                log_store.append([running_job.ends_at, uid, running_job.job.uid, running_job.job.duration, running_job.job.cpu_req])
                released = True
        if released and received_no_matching:
            received_no_matching = False
            sent = False

def main(node_caps: List[int], jobs: List[Job]):
    env = simpy.Environment()
    log_store = []
    broker = MessageBroker(env, 1 + len(node_caps))
    env.process(queue_agent(env, broker, 0, jobs))
    for i, cap in enumerate(node_caps):
        env.process(worker_agent(env, broker, i + 1, cap, log_store))
    env.run()

    with open('result_simpy.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['completion', 'node', 'uid', 'duration', 'cpu_req'])
        for row in log_store:
            writer.writerow(row)

if __name__ == '__main__':
    main(node_caps=[2] * 1,
        jobs=[Job(i, 2, 1) for i in range(10)])
