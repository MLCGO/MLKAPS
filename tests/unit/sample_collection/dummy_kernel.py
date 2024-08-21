#!/bin/env python3

import sys
import os

def main():

    if len(sys.argv) != 2:
        print("Usage: python dummy_kernel.py id")
        sys.exit(1)

    if os.environ.get("DO_SLEEP") == "1":
        import time

        time.sleep(1)

    n_output = 1

    if os.environ.get("N_OUTPUT") is not None:
        n_output = int(os.environ.get("N_OUTPUT"))
        print(n_output)
        
    print("Hello, World!")

    if n_output < 1:
        return
    
    id = int(sys.argv[1])
    print(id, end="")

    for i in range(n_output - 1):
        print(f",", id / (i + 2), end="")


if __name__ == "__main__":
    main()
