from __future__ import annotations

import json

from inference import _log


def test_start_step_end_log_snapshots(capsys):
    _log("[START]", {"task_id": 1, "scenario_id": "demo_scenario", "seed": 7})
    _log(
        "[STEP]",
        {
            "task_id": 1,
            "step": 1,
            "action": {"pr_decision": "REJECT"},
            "reward": 0.75,
            "done": False,
        },
    )
    _log(
        "[END]",
        {
            "task_id": 1,
            "steps": 6,
            "total_reward": 3.5,
            "episode_score": 0.81,
        },
    )

    lines = capsys.readouterr().out.strip().splitlines()

    assert lines[0] == '[START] {"task_id":1,"scenario_id":"demo_scenario","seed":7}'
    assert (
        lines[1]
        == '[STEP] {"task_id":1,"step":1,"action":{"pr_decision":"REJECT"},"reward":0.75,"done":false}'
    )
    assert lines[2] == '[END] {"task_id":1,"steps":6,"total_reward":3.5,"episode_score":0.81}'

    for line in lines:
        tag, payload = line.split(" ", 1)
        assert tag in {"[START]", "[STEP]", "[END]"}
        decoded = json.loads(payload)
        assert isinstance(decoded, dict)
