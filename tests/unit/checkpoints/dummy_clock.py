#!/bin/env python3

import time
import csv
import os
from pathlib import Path


# log time every second
def my_clock():
    log_file_path = Path(__file__).parent / "time_log.csv"
    with open(log_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        while True:
            writer.writerow([time.strftime("%H:%M:%S")])
            file.flush()
            os.fsync(file.fileno())

            print(time.strftime("%H:%M:%S"))
            time.sleep(1)


if __name__ == "__main__":
    my_clock()
