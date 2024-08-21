import builder_helper as helper
import pytest


@pytest.fixture(scope="class")
def builder_helper():
    return helper.TestBuilderHelper()
