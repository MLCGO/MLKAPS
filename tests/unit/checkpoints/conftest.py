# conftest.py
import pytest


@pytest.fixture
def remote_recovery_data():
    # to run with srun, set to True and give the name of the target machine
    return [False, "name of target machine"]
