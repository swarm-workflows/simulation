import subprocess
from typing import List

import psutil


class TaskExecutor:
    @staticmethod
    def set_cpu_affinity(process, cpus: List[int]):
        try:
            process_pid = process.pid
            cpu_affinity = cpus
            psutil.Process(process_pid).cpu_affinity(cpu_affinity)
        except Exception as e:
            print(f"Error setting CPU affinity for process {process}: {e}")

    @staticmethod
    def run_command(command, cpus):
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True
            )

            # Set CPU affinity for the subprocess
            #TaskExecutor.set_cpu_affinity(process, cpus)

            # Wait for the subprocess to complete
            process.wait()

            return process.returncode
        except subprocess.CalledProcessError as e:
            print(f"Error executing command: {command}, Error: {e}")
            return None
