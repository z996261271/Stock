import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from professional_quant.core.context import RunContext  # noqa: E402
from professional_quant.core.events import ORDER_PLAN_BUILT, SIGNAL_GENERATED  # noqa: E402
from professional_quant.core.mod import ModManager  # noqa: E402


class CountingMod:
    name = "counting"

    def __init__(self) -> None:
        self.events = []

    def install(self, context: RunContext) -> None:
        context.event_bus.subscribe("*", self.events.append)


def test_run_context_event_bus_and_mod_manager_capture_lifecycle():
    context = RunContext(run_id="run_1", strategy="demo", mode="paper")
    mod = CountingMod()
    installed = ModManager([mod]).install_all(context)

    context.emit(SIGNAL_GENERATED, {"signals": 2})
    context.emit(ORDER_PLAN_BUILT, {"orders": 1})
    context.add_artifact("state_report", "reports/paper/state.json")
    manifest = context.manifest()

    assert installed == ["counting"]
    assert [event.name for event in mod.events] == [SIGNAL_GENERATED, ORDER_PLAN_BUILT]
    assert manifest["metadata"]["installed_mods"] == ["counting"]
    assert manifest["artifacts"]["state_report"] == "reports/paper/state.json"
    assert [event["name"] for event in manifest["events"]] == [SIGNAL_GENERATED, ORDER_PLAN_BUILT]


if __name__ == "__main__":
    test_run_context_event_bus_and_mod_manager_capture_lifecycle()

