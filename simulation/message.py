import logging

from mpi4py import MPI


class MessageHelper:
    @staticmethod
    def send_message(data: dict, destination: int, tag: int):
        """
        Send message from a source to destination; messages can be filtered based on the tags
        @param data - dictionary containing the message to be sent
        @param destination - destination process rank
        @param tag - message type used for filtering
        """
        logging.info(f"Sending Message: {data} sent to {destination}")
        MPI.COMM_WORLD.send(data, dest=destination, tag=tag)

    @staticmethod
    def receive_message(source: int, tag: int) -> dict:
        """
        Receive message from a source; messages can be filtered based on the tags
        @param source - source process rank
        @param tag - message type used for filtering
        """
        data = MPI.COMM_WORLD.recv(source=source, tag=tag)
        logging.info(f"Received Message: {data} from {source}")
        return data
