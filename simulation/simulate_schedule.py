#!/usr/bin/env python
import csv
import numpy as np
from mpi4py import MPI
from repast4py.network import read_network
from repast4py import core
from repast4py import context as ctx
from repast4py import schedule, logging
from typing import Tuple, Dict, List, Iterable, Self, NamedTuple

class Job(NamedTuple):
    uid: int
    duration: int
    cpu_req: int

class RunningJob(NamedTuple):
    remaining: int
    cpu_req: int

class Offer(NamedTuple):
    job_uid: int
    node_uid: int

def jobs_from_csv(fn: str) -> List[Job]:
    res = []
    with open(fn, 'r') as f:
        for row in csv.DictReader(f):
            res.append(Job(
                uid=int(row['uid']),
                duration=int(row['duration']),
                cpu_req=int(row['cpu_req'])))
    return res

class QueueAgent(core.Agent):
    TYPE = 0

    def __init__(self, local_id: int, rank: int, jobs: Iterable[Job], allocations: Dict[int, Job] = None):
        super().__init__(id=local_id, type=QueueAgent.TYPE, rank=rank)
        self.jobs_by_uid = {job.uid: job for job in jobs}
        self.allocations = allocations or {}

    @property
    def available_jobs(self) -> Iterable[Job]:
        return self.jobs_by_uid.values()

    @property
    def available_jobs_len(self) -> int:
        return len(self.jobs_by_uid)

    def process_offers(self, offers: List[Offer]):
        # XXX: extend
        offers = filter(lambda o: o is not None, offers)
        offers = sorted(offers, key=lambda o: o.node_uid)

        self.allocations = {}
        for offer in offers:
            if offer is not None and offer.job_uid in self.jobs_by_uid:
                assert offer.node_uid not in self.allocations
                self.allocations[offer.node_uid] = self.jobs_by_uid[offer.job_uid]
                del self.jobs_by_uid[offer.job_uid]

    def save(self) -> Tuple:
        return (self.uid, list(self.jobs_by_uid.values()), self.allocations)

    @staticmethod
    def restore(data: Tuple) -> Self:
        uid = data[0]
        return QueueAgent(uid[0], uid[2], data[1], data[2])

    def update(self, jobs: Iterable[Job], allocations: Dict[int, Job]):
        self.jobs_by_uid = {job.uid: job for job in jobs}
        self.allocations = allocations

class NodeAgent(core.Agent):
    TYPE = 1
    def __init__(self, local_id: int, rank: int, capacity: int, running_tasks: Dict[int, RunningJob] = None, offer: Offer = None):
        super().__init__(id=local_id, type=NodeAgent.TYPE, rank=rank)
        self.capacity = capacity
        self.offer = offer
        self._running_tasks = running_tasks or {}

    def make_offer(self, queue: QueueAgent):
        self.offer = None
        for job in queue.available_jobs:
            if job.cpu_req <= self.capacity:
                self.offer = Offer(node_uid=self.uid, job_uid=job.uid)
                return

    def get_allocation(self, queue: QueueAgent, logger_fn):
        if self.uid in queue.allocations:
            job = queue.allocations[self.uid]
            assert job.uid not in self._running_tasks
            self._running_tasks[job.uid] = RunningJob(job.duration, job.cpu_req)
            self.capacity -= job.cpu_req
            assert self.capacity >= 0
            logger_fn(job)

    def process(self):
        for job in list(self._running_tasks.keys()):
            running_job = self._running_tasks[job]
            self._running_tasks[job] = RunningJob(running_job.remaining - 1, running_job.cpu_req)
            if self._running_tasks[job].remaining == 0:
                # job complete, release resources
                self.capacity += running_job.cpu_req
                del self._running_tasks[job]

    def save(self) -> Tuple:
        return (self.uid, self.capacity, self._running_tasks, self.offer)

    @staticmethod
    def restore(data: Tuple) -> Self:
        uid = data[0]
        return NodeAgent(uid[0], uid[2], data[1], data[2], data[3])

    def update(self, capacity: int, running_tasks: Dict[int, List[int]], offer: Offer):
        self.capacity = capacity
        self._running_tasks = running_tasks
        self.offer = offer

    @property
    def running_jobs_count(self):
        return len(self._running_tasks)

def restore_agent(data: Tuple) -> core.Agent:
    uid = data[0]
    match uid[1]:
        case QueueAgent.TYPE:
            return QueueAgent.restore(data)
        case NodeAgent.TYPE:
            return NodeAgent.restore(data)
        case _:
            raise RuntimeError('Unknown agent type')

def create_agent(uid, agent_type, rank, **kwargs):
    match agent_type:
        case QueueAgent.TYPE:
            jobs = jobs_from_csv(kwargs['initial_jobs'])
            return QueueAgent(uid, rank, jobs)
        case NodeAgent.TYPE:
            return NodeAgent(uid, rank, int(kwargs['cpus']))
        case _:
            raise RuntimeError(f'Unknown agent type {agent_type}')

class Model:
    def __init__(self, comm: MPI.Intracomm, schedule_log_file='logs/schedule.log', network='network.txt'):
        self.comm = comm
        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        # self.runner.schedule_stop(22)
        self.runner.schedule_end_event(self.at_end)

        self.context = ctx.SharedContext(comm)
        read_network(network, self.context, create_agent, restore_agent)
        self.net = self.context.get_projection('swarm_network')

        # initialize the logging
        self.node_logger = logging.TabularLogger(comm, schedule_log_file, ['tick', 'agent_id', 'job_id', 'duration', 'cpu_req', 'completion_time'])

    def local_agents(self, agent_type):
        if self.context.contains_type(agent_type):
            return self.context.agents(agent_type)
        return []

    def step(self):
        # reset counters
        tasks_queued = 0
        tasks_running = 0
        tick = self.runner.schedule.tick

        for node in self.local_agents(NodeAgent.TYPE):
            for queue in self.net.graph.neighbors(node):
                node.make_offer(queue)

        # Make offers visible to queue
        self.context.synchronize(restore_agent)

        for queue in self.local_agents(QueueAgent.TYPE):
            offers = [node.offer for node in self.net.graph.neighbors(queue)]
            queue.process_offers(offers)
            tasks_queued += int(queue.available_jobs_len)

        # Make queue decision visible to nodes
        self.context.synchronize(restore_agent)

        for node in self.local_agents(NodeAgent.TYPE):
            def _log(job: Job):
                self.node_logger.log_row(tick, node.id, job.uid, job.duration, job.cpu_req, job.duration + tick)

            for queue in self.net.graph.neighbors(node):
                node.get_allocation(queue, _log)
            node.process()
            tasks_running += node.running_jobs_count

        self.node_logger.write()
        self.check_stop_condition(tasks_running + tasks_queued)

    def check_stop_condition(self, local_tasks: int):
        my_cnt = np.zeros(1, dtype=int)
        cnt_sum = np.zeros(1, dtype=int)
        my_cnt[0] = local_tasks
        self.comm.Allreduce(my_cnt, cnt_sum, MPI.SUM)
        if cnt_sum[0] == 0:
                self.runner.stop()

    def at_end(self):
        self.node_logger.close()

    def start(self):
        self.runner.execute()

def run():
    model = Model(MPI.COMM_WORLD)
    model.start()

if __name__ == '__main__':
    run()
