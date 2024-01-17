from mpi4py import MPI
from repast4py.network import UndirectedSharedNetwork


class Model:
    def __init__(self, parameters: dict):
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