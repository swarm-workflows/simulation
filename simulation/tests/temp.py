from mpi4py import MPI
import repast4py.core as core
import repast4py.network as rpn
import repast4py.space as rps
import random
from repast4py.network import UndirectedSharedNetwork
from repast4py.schedule import Schedule, PriorityType
import networkx as nx
import matplotlib.pyplot as plt


class DataPacket:
    def __init__(self, size, content):
        self.size = size
        self.content = content


class ISP(core.Agent):
    def __init__(self, a_id, rank):
        super().__init__(a_id, 0, rank)
        self.a_id = a_id
        self.rank = rank
        self.data_traffic = 0
        self.received_messages = 0
        self.sent_packets = 0

    def step(self, model):
        if model.comm.Iprobe(source=MPI.ANY_SOURCE):
            data_packet = model.comm.recv(source=MPI.ANY_SOURCE)
            self.receive_packet(data_packet, model)
        self.send_packet(model)

    def pos(self):
        return (random.random(), random.random())

    def send_packet(self, model):
        neighbors = list(model.network.graph.neighbors(self))
        if neighbors:
            neighbors_load = {}
            for neighbor in neighbors:
                neighbor_mpi_rank = model.neighbor_rank_map.get(neighbor.a_id, None)
                if neighbor_mpi_rank is None:
                    print(f"Warning: Missing rank info for neighbor {neighbor.a_id}. Skipping.")
                    continue
                model.comm.isend(('request_load', self.rank), dest=neighbor_mpi_rank)

            for neighbor in neighbors:
                received_messages_count = model.comm.recv(source=neighbor.rank)
                neighbors_load[neighbor] = received_messages_count

            selected_neighbor = min(neighbors_load, key=neighbors_load.get)

            # print(f"[Debug] ISP {self.a_id} on rank {model.rank} sending packet to {selected_neighbor.a_id}")
            new_packet = DataPacket(size=1, content="Hello neighbor")
            model.send_message(selected_neighbor.rank, new_packet)
            self.data_traffic += new_packet.size
            self.sent_packets += 1
        else:
            # print(f"[Debug] ISP {self.a_id} on rank {model.rank} has no neighbors to send to.")
            pass

    def receive_packet(self, data_packet, model):
        # print(f"[Debug] ISP {self.a_id} on rank {model.rank} received packet.")
        if isinstance(data_packet, tuple) and data_packet[0] == 'request_load':
            sender_rank = data_packet[1]
            model.comm.send(self.received_messages, dest=sender_rank)
        else:
            self.received_messages += 1


class ISPModel:
    def __init__(self, params):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.network = UndirectedSharedNetwork('network_name', self.comm)

        self.agents = [ISP(i, self.rank) for i in range(self.rank, params['num_agents'], self.size)]
        for agent in self.agents:
            self.network.add(agent)

        self.neighbor_rank_map = {}

        self.initialize_random_topology(params['num_agents'])

        self.steps = params.get('steps', 10)

    def initialize_random_topology(self, num_agents):
        for agent in self.agents:
            num_connections = random.randint(1, min(3, num_agents - 1))

            possible_partners = [a for a in self.agents if a != agent]
            connections = random.sample(possible_partners, num_connections)

            for partner in connections:
                if not self.network.graph.has_edge(agent, partner):
                    self.network.add_edge(agent, partner)

    def gather_statistics(self):
        local_stats = [(agent.a_id, agent.sent_packets, agent.received_messages) for agent in self.agents]

        all_stats = self.comm.gather(local_stats, root=0)

        if self.rank == 0:
            all_stats_flat = [stat for sublist in all_stats for stat in sublist]
            all_stats_sorted = sorted(all_stats_flat, key=lambda x: x[0])
            for a_id, sent, received in all_stats_sorted:
                print(f"ISP {a_id}: Sent Packets = {sent}, Received Packets = {received}")

    def setup_network(self):
        for agent in self.agents:
            self.neighbor_rank_map[agent.a_id] = agent.rank

        self.comm.Barrier()

        # if self.rank == 0:
        #     print(f"[Debug] neighbor_rank_map: {self.neighbor_rank_map}")

    def run(self):
        self.setup_network()
        scheduler = Schedule()

        for agent in self.agents:
            scheduler.schedule_repeating_event(1.0, 1.0, lambda agent=agent: agent.step(self),
                                               priority_type=PriorityType.RANDOM)

        for _ in range(self.steps):
            scheduler.execute()
            self.comm.Barrier()
            # You can draw the network at each step by uncommenting the next line
            # self.draw_network()

        self.gather_statistics()
        # Or draw the network at the end of the simulation
        self.draw_network()

    def send_message(self, recipient_rank, message):
        # print(f"[Debug] Message sent from ISP {self.rank} to ISP with rank {recipient_rank}")
        self.comm.send(message, dest=recipient_rank)

    def receive_messages(self):
        while self.comm.Iprobe(source=MPI.ANY_SOURCE):
            message = self.comm.recv(source=MPI.ANY_SOURCE)
            # print(f"[Debug] Message received by ISP {self.rank}")

    def draw_network(self):
        # Create a NetworkX graph from the repast4py network
        G = nx.Graph()
        positions = {}
        labels = {}
        for agent in self.agents:
            G.add_node(agent.a_id)
            positions[agent.a_id] = agent.pos()
            labels[agent.a_id] = f"{agent.a_id}\nS:{agent.sent_packets}\nR:{agent.received_messages}"
            for neighbor in self.network.graph.neighbors(agent):
                if agent.a_id < neighbor.a_id:  # To ensure each edge is added only once
                    G.add_edge(agent.a_id, neighbor.a_id)

        # Draw the network
        plt.figure(figsize=(6, 6))
        nx.draw(G, pos=positions, with_labels=False, node_size=500, node_color="skyblue", alpha=0.7)
        nx.draw_networkx_labels(G, positions, labels=labels)
        plt.title("Network of ISPs with Sent (S) and Received (R) Packets")
        plt.show()


parameters = {'steps': 10, 'num_agents': 10}

if __name__ == '__main__':
    model = ISPModel(parameters)

    # if model.rank == 0:
    #     print("[Debug] Initial network state:")
    #     for agent in model.agents:
    #         print(f"  Agent {agent.a_id} connected to {[neighbor.a_id for neighbor in model.network.graph.neighbors(agent)]}")

    model.run()