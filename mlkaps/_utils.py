"""
    Copyright (C) 2020-2024 Intel Corporation
    Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
    Copyright (C) 2024-  MLKAPS contributors
    SPDX-License-Identifier: BSD-3-Clause
"""

import logging
import pathlib
import os
import re

_log_format = "(%(asctime)s) [%(levelname)s] %(message)s"


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format=_log_format,
        handlers=[
            logging.StreamHandler(),
        ],
        force=True,  # Force the configuration to be applied even if logging has already been configured
    )


def add_file_logger(output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "mlkaps_output.log"
    logging.info(f"Creating log file at {log_path}")

    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(_log_format))
    logging.getLogger().addHandler(fh)


def ensure_not_root(args):
    if os.getuid() != 0 or args.allow_root:
        return

    logging.critical(
        "MLKAPS was run with root permissions. This is not allowed due to security concerns, exiting (1)"
    )
    exit(1)


_ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def filter_ansi_code(text):
    return _ansi_escape.sub("", text)
