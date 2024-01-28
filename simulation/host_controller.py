import threading
import time
import psutil
from confluent_kafka import Consumer, KafkaError
from mpi4py import MPI
from repast4py.network import UndirectedSharedNetwork
from repast4py.schedule import Schedule

from simulation.job_queue import JobQueue, Job
from simulation.base_agent import AgentType
from simulation.resource_agent import ResourceAgent


class HostController:
    def __init__(self, kafka_bootstrap_servers, kafka_topic, group_id='my_group'):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.group_id = group_id
        self.scheduler = Schedule()
        self.comm = MPI.COMM_WORLD
        self.network = UndirectedSharedNetwork('resource_agent_nw', self.comm)

    def get_system_resources(self):
        cpu_count = psutil.cpu_count(logical=False)
        nic_count = len(psutil.net_if_addrs())
        gpu_count = 0

        result = {"cpus": list(range(1, cpu_count + 1)), "nics": list(range(1, nic_count + 1)), "gpus": list(range(1, gpu_count + 1))}
        return result

    def read_kafka_jobs(self):
        conf = {
            'bootstrap.servers': self.kafka_bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }

        consumer = Consumer(conf)
        consumer.subscribe([self.kafka_topic])

        try:
            while True:
                msg = consumer.poll(1.0)

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(msg.error())
                        break

                job_info = msg.value().decode('utf-8')
                job_data = self.parse_job_info(job_info)
                self.populate_job_queue(job_data)
        finally:
            consumer.close()

    def parse_job_info(self, job_info):
        job_data = {
            "resources": {"cpus": 1, "gpus": 0, "nics": 0},
            "commands": [f"python3 -m agents.jobs.cpu_computation"]
        }
        return job_data

    def populate_job_queue(self, job_data):
        job_queue = JobQueue.get()
        job = Job(resources=job_data["resources"], commands=job_data["commands"])
        job_queue.add_job(job)

    def start(self):
        kafka_thread = threading.Thread(target=self.read_kafka_jobs, daemon=True)
        kafka_thread.start()

        system_resources = self.get_system_resources()
        resource_agent = ResourceAgent(0, AgentType.Resource, 0, system_resources)

        self.network.add(agent=resource_agent)

        self.scheduler.schedule_repeating_event(1.0, 1.0, lambda agent=resource_agent: agent.step())

        steps = 10
        for _ in range(steps):
            self.scheduler.execute()
            time.sleep(1.0)


if __name__ == '__main__':
    kafka_bootstrap_servers = 'localhost:9092'
    kafka_topic = 'host-1'

    host_controller = HostController(kafka_bootstrap_servers, kafka_topic)
    host_controller.start()