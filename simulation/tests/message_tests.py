from mpi4py import MPI

from simulation.message import MessageHelper

if __name__ == '__main__':
    comm = MPI.COMM_WORLD  # Get the MPI communicator
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes

    if rank == 0:
        # Code executed by the root process (rank 0)
        data = {'message': 'Hello from the root process!'}
        print(f"Seding: {data}")
        #comm.send(data, dest=1, tag=0)
        MessageHelper.send_message(data, destination=1, tag=0)
    elif rank == 1:
        # Code executed by the second process (rank 1)
        #data = comm.recv(source=0, tag=0)
        data = MessageHelper.receive_message(source=0, tag=0)
        print(f"Received message: {data['message']}")

    # Ensure all processes have finished before exiting
    comm.Barrier()
