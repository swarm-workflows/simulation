import socket
import psutil


class Utils:
    @staticmethod
    def get_ip_addresses():
        # Get network interface information
        net_if_info = psutil.net_if_addrs()

        # Create a list to store IP addresses
        ip_addresses = []

        # Iterate through network interfaces
        for interface, addresses in net_if_info.items():
            for address in addresses:
                if address.family == socket.AF_INET:  # Check for IPv4 addresses
                    ip_addresses.append(address.address)
        return ip_addresses

    @staticmethod
    def get_system_resources():
        cpu_count = psutil.cpu_count(logical=False)

        gpu_count = 0

        result = {
            "cpus": list(range(1, cpu_count + 1)),
            "nics": Utils.get_ip_addresses(),
            "gpus": list(range(1, gpu_count + 1))
        }
        return result
