#!/usr/bin/env python3

import psutil
import sys

if len(sys.argv) != 3:
    print("Usage: python3 templatize.py <input_file> <output_file>")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

ncores = len(psutil.Process().cpu_affinity())

template = "config_mlkaps_template.json"

match = [("@max_threads", str(ncores))]

with open(template, "r") as i:
    content = i.read()

for m in match:
    content = content.replace(m[0], m[1])

with open("config_mlkaps.json", "w") as o:
    o.write(content)
