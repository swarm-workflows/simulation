import time
from datetime import datetime

from action import Action


class DiskAction(Action):
    def perform(self, *args, **kwargs):
        start = datetime.now()
        time.sleep(5)
        #print(f"Disk operation: took: {(datetime.now() - start).total_seconds()}")
