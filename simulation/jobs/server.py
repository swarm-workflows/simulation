import socket
import sys


def start_server(ip_address: str, port: int, max_connections: int):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to the specified address and port
    server_address = (ip_address, port)
    server_socket.bind(server_address)

    # Listen for incoming connections
    server_socket.listen(max_connections)
    print(f"Server listening on {server_address}, waiting for {max_connections} connections...")

    connection_count = 0

    while connection_count < max_connections:
        # Wait for a connection
        print("Waiting for a connection...")
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        # Receive and print the message from the client
        message = client_socket.recv(1024).decode('utf-8')
        print(f"Received message from client: {message}")

        # Send a response back to the client
        response = "Hello from the server!"
        client_socket.send(response.encode('utf-8'))

        # Close the connection with the client
        client_socket.close()

        connection_count += 1

    # Close the server socket after all connections are handled
    server_socket.close()


if __name__ == "__main__":
    # Check if all required arguments are provided
    if len(sys.argv) != 4:
        print("Usage: python server.py <ip_address> <port> <max_connections>")
        sys.exit(1)

    # Parse the command-line arguments
    ip_address = sys.argv[1]
    port = int(sys.argv[2])
    max_connections = int(sys.argv[3])

    start_server(ip_address, port, max_connections)
