import os

from mlkaps.MLKaps import run_from_config
import pytest

class Test2D:

    def _run_helper(self, tmp_path, builder_helper, config_name):
        curr_path = os.path.abspath(os.path.dirname(__file__))
        builder_helper.setup(srcpath=curr_path, build=False)

        json, _ = builder_helper.load_configuration(config_name, tmp_path)
        run_from_config([], json, builder_helper.srcpath, tmp_path)

    @pytest.mark.parametrize("file", ["hvs.json", "hvsr.json", "multilevel_hvsr.json", "random.json", "ga_adaptive.json", "ga_adaptive_optuna.json", "optuna_model.json"])
    def test_config(self, tmp_path, builder_helper, file):
        self._run_helper(tmp_path, builder_helper, file)