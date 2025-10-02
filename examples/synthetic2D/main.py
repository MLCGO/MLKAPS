#!/usr/bin/env python3
import sys
from math import cos, sin


def main(cdict: dict | None = None):

    if cdict is None:
        if len(sys.argv) < 4:
            exit(1)

        x = float(sys.argv[1])
        y = float(sys.argv[2])
        b = float(sys.argv[3])
    else:
        x = cdict["x"]
        y = cdict["y"]
        b = cdict["b"]

    res = b * sin(x) ** 2 * x * cos(y) ** 2 * 1 / (10 + x * y)

    if cdict is None:
        print(res)
    else:
        cdict["r"] = res
        return cdict


if __name__ == "__main__":
    main()
