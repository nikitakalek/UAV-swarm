import pytest

from swarmsim.scenario import Scenario


def make_scenario(**kw) -> Scenario:
    base = {
        "name": "test",
        "dt": 0.5,
        "t_end": 3600,
        "world": {"size": [1000, 1000], "targets": {"count": 10}},
        "agents": {"count": 4},
        "mission": {"n_cells": 9},
    }
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    return Scenario.model_validate(base)


@pytest.fixture
def scenario_factory():
    return make_scenario
