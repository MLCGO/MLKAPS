# Copyright (C) 2020-2024 Intel Corporation
# Copyright (C) 2022-2024 University of Versailles Saint-Quentin-en-Yvelines
# Copyright (C) 2024-  MLKAPS contributors
# SPDX-License-Identifier: BSD-3-Clause
#

import json
import os
import subprocess


class TestBuilderHelper:

    def setup(self, srcpath=None, buildpath=None, build=False):

        if srcpath and not os.path.exists(srcpath):
            raise ValueError("Invalid src path")

        if srcpath and not os.path.isdir(srcpath):
            raise ValueError("src path must be a directory")

        self.srcpath = os.path.abspath(srcpath)

        if not buildpath:
            buildpath = self.srcpath
        elif not os.path.isdirectory(buildpath):
            raise ValueError("buildpath path must be a directory")

        self.buildpath = os.path.abspath(buildpath)

        if build:
            self.run_makefile()

    def run_makefile(self):
        if not self.srcpath:
            raise ValueError("Cannot call makefile: the source folder was not set")
        if not self.buildpath:
            raise ValueError("Cannot call makefile: the buildfolder was not set")
        os.makedirs(self.buildpath, exist_ok=True)
        subprocess.run(self.srcpath + "/compiler.sh", cwd=self.buildpath)

    def load_configuration(self, relative_json_path, output_path) -> [dict, str]:
        json_path = self.srcpath + "/" + relative_json_path
        with open(json_path, "r") as f:
            config = json.load(f)

        config["EXPERIMENT"]["output_path"] = str(output_path)
        return config, json_path
