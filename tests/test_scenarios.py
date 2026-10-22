"""M2-7: схема сценария и шаблонные сценарии репозитория."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from swarmsim.scenario import Scenario

ROOT = Path(__file__).resolve().parents[1]
SCEN = sorted((ROOT / "scenarios").glob("*.yaml"))


def test_at_least_three_scenarios_present():
    names = {p.stem for p in SCEN}
    assert {"nominal", "failures", "comms_loss"} <= names


@pytest.mark.parametrize("path", SCEN, ids=lambda p: p.stem)
def test_repo_scenarios_valid_and_reproducible(path):
    pytest.importorskip("yaml")
    from swarmsim.sim import run

    sc = Scenario.from_yaml(path)
    a, b = run(sc, 1), run(sc, 1)
    assert a.metrics_hash() == b.metrics_hash()
    assert a.metrics["valid"]


def test_schema_rejects_bad_input(scenario_factory):
    with pytest.raises(ValidationError):
        scenario_factory(events=[{"t": 10, "type": "agent_failure", "agent": 99}])
    with pytest.raises(ValidationError):
        scenario_factory(events=[{"t": 10, "type": "meteor"}])
    with pytest.raises(ValidationError):
        scenario_factory(agents={"count": 0})
    with pytest.raises(ValidationError):
        scenario_factory(unknown_field=1)
    with pytest.raises(ValidationError):
        scenario_factory(schema_version=2)


def test_scenario_describes_area_n_targets_events(scenario_factory):
    sc = scenario_factory(events=[{"t": 5, "type": "comms_loss", "duration": 10}])
    assert sc.world.bounds_polygon() and sc.agents.count == 4 and sc.world.targets.count == 10
    assert sc.events[0].type == "comms_loss"
