
import socket

if __name__ == "__main__":
    ip = "10.42.0.1"
    # ip = "127.0.0.1"
    port = 5003

    connection, client_address = None, None
    while True:
        if connection is None or client_address is None:
            serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            serverSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            serverSocket.bind((ip, port))

            print(f'Listening to connections on {ip}:{port}')
            serverSocket.listen()

            connection, client_address = serverSocket.accept()
            print(f"Accepted a connection request from {client_address[0]}:{client_address[1]}")

        encoded_data = connection.recv(1024)
        data = encoded_data.decode()

        if len(data) == 0:
            print('Closed pipe')
            connection.close()
            connection = None
        print(data)
