#!/usr/bin/env python
import argparse
import csv
import numpy as np
from mpi4py import MPI
from repast4py.network import read_network
from repast4py import core
from repast4py import context as ctx
from repast4py import schedule, logging
from typing import Tuple, Dict, List, Iterable, Self, NamedTuple, Callable

class Job(NamedTuple):
    uid: int
    duration: int
    cpu_req: int

class RunningJob(NamedTuple):
    remaining: int
    job: Job

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

def load_disconnections(fn: str) -> Dict[Tuple[int, int], int]:
    res = {}
    with open(fn, 'r') as f:
        for row in csv.DictReader(f):
            key = (int(row['tick']), int(row['node']))
            assert key not in res
            res[key] = int(row['duration'])
    return res

class QueueAgent(core.Agent):
    TYPE = 0

    def __init__(self, local_id: int, rank: int, jobs: Iterable[Job], allocations: Dict[int, Job] = None):
        super().__init__(id=local_id, type=QueueAgent.TYPE, rank=rank)
        self.jobs_by_uid = {}
        self.allocations = allocations or {}
        self._add_jobs(jobs)

    def available_jobs(self) -> Iterable[Job]:
        return self.jobs_by_uid.values()

    @property
    def available_jobs_len(self) -> int:
        return len(self.jobs_by_uid)

    def get_allocation(self, worker_id: int) -> Job | None:
        return self.allocations.get(worker_id)

    def _add_jobs(self, jobs: Iterable[Job]) -> None:
        for j in jobs:
            self.jobs_by_uid[j.uid] = j

    def process_nodes(self, nodes: Iterable['NodeAgent']):
        offers = []
        for node in nodes:
            if node.is_damaged():
                self._add_jobs(node.get_currently_running())
            elif (offer := node.get_offer()) is not None:
                offers.append(offer)

        # XXX: extend
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
    def __init__(self, local_id: int, rank: int, capacity: int, running_tasks: Dict[int, RunningJob] = None, offer: Offer = None, unavail_timeout: int = 0):
        super().__init__(id=local_id, type=NodeAgent.TYPE, rank=rank)
        self.capacity = capacity
        self.offer = offer
        self._running_tasks = running_tasks or {}
        self.unavail_timeout = unavail_timeout

    def simulate_disconnection(self, timeout):
        if self.unavail_timeout == 0:
            self.unavail_timeout = timeout

    def is_damaged(self) -> bool:
        return self.unavail_timeout > 0

    def get_currently_running(self) -> List[Job]:
        return [rj.job for rj in self._running_tasks.values()]

    def get_offer(self) -> Offer | None:
        return self.offer

    def make_offer(self, queue: QueueAgent):
        self.offer = None
        if self.is_damaged():
            return
        for job in queue.available_jobs():
            if job.cpu_req <= self.capacity:
                self.offer = Offer(node_uid=self.uid, job_uid=job.uid)
                return

    def get_allocation(self, queue: QueueAgent):
        job = queue.get_allocation(self.uid)
        if job is None:
            return

        assert not self.is_damaged()
        assert job.uid not in self._running_tasks
        self._running_tasks[job.uid] = RunningJob(job.duration, job)
        self.capacity -= job.cpu_req
        assert self.capacity >= 0

    def process(self, logger_fn: Callable[Job, None]):
        if self.is_damaged():
            self._running_tasks = {}
            self.unavail_timeout -= 1
            return

        for job in list(self._running_tasks.keys()):
            running_job = self._running_tasks[job]
            self._running_tasks[job] = RunningJob(running_job.remaining - 1, running_job.job)
            if self._running_tasks[job].remaining == 0:
                # job complete, release resources
                logger_fn(running_job.job)
                self.capacity += running_job.job.cpu_req
                del self._running_tasks[job]

    def save(self) -> Tuple:
        return (self.uid, self.capacity, self._running_tasks, self.offer, self.unavail_timeout)

    @staticmethod
    def restore(data: Tuple) -> Self:
        uid = data[0]
        return NodeAgent(uid[0], uid[2], data[1], data[2], data[3], data[4])

    def update(self, capacity: int, running_tasks: Dict[int, List[int]], offer: Offer, unavail_timeout: int):
        self.capacity = capacity
        self._running_tasks = running_tasks
        self.offer = offer
        self.unavail_timeout = unavail_timeout

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
    def __init__(
        self,
        comm: MPI.Intracomm,
        schedule_log_file: str = 'logs/schedule.log',
        network: str ='network.txt',
        disconnection_schedule: str = None
        ):
        self.comm = comm
        self.disconnection_schedule = load_disconnections(disconnection_schedule) if disconnection_schedule is not None else {}
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
            if (tick, node.uid[0]) in self.disconnection_schedule:
                duration = self.disconnection_schedule[(tick, node.uid[0])]
                node.simulate_disconnection(duration)
            for queue in self.net.graph.neighbors(node):
                node.make_offer(queue)

        # Make offers visible to queue
        self.context.synchronize(restore_agent)

        for queue in self.local_agents(QueueAgent.TYPE):
            queue.process_nodes(self.net.graph.neighbors(queue))
            tasks_queued += int(queue.available_jobs_len)

        # Make queue decision visible to nodes
        self.context.synchronize(restore_agent)

        for node in self.local_agents(NodeAgent.TYPE):
            def _log(job: Job):
                self.node_logger.log_row(tick - job.duration, node.id, job.uid, job.duration, job.cpu_req, tick)

            for queue in self.net.graph.neighbors(node):
                node.get_allocation(queue)
            node.process(_log)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default='logs/schedule.log')
    parser.add_argument('--network', default='network.txt')
    parser.add_argument('--disconnection-schedule')
    args = parser.parse_args()
    model = Model(MPI.COMM_WORLD, schedule_log_file=args.log, network=args.network, disconnection_schedule=args.disconnection_schedule)
    model.start()

if __name__ == '__main__':
    run()
