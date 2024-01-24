import socket
import sys


def start_client(server_ip: str, server_port: int, source_ip: str):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to a specific source IP address (if provided)
    if source_ip:
        client_socket.bind((source_ip, 0))  # Use a random port for the source

    # Connect to the server
    server_address = (server_ip, server_port)
    client_socket.connect(server_address)
    print(f"Connected to {server_address}")

    # Send a message to the server
    message = "Hello from the client!"
    client_socket.send(message.encode('utf-8'))
    print(f"Sent message to server: {message}")

    # Receive and print the response from the server
    response = client_socket.recv(1024).decode('utf-8')
    print(f"Received response from server: {response}")

    # Close the connection with the server
    client_socket.close()


if __name__ == "__main__":
    # Check if all required arguments are provided
    print(sys.argv)

    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python client.py <server_ip> <server_port> <source_ip (optional)>")
        sys.exit(1)

    # Parse the command-line arguments
    server_ip = sys.argv[1]
    server_port = int(sys.argv[2])
    source_ip = sys.argv[3] if len(sys.argv) == 4 else None

    start_client(server_ip, server_port, source_ip)
