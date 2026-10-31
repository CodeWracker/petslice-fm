import json

import torch

from petslice.training.runner import restore_history, save_checkpoint


def test_checkpoint_replaces_atomically_and_round_trips(tmp_path):
    path = tmp_path / "last.pt"
    save_checkpoint(path, {"epoch": 3, "rng": torch.get_rng_state()})
    save_checkpoint(path, {"epoch": 4, "rng": torch.get_rng_state()})
    assert torch.load(path)["epoch"] == 4
    assert [p.name for p in tmp_path.iterdir()] == ["last.pt"]


def test_history_drops_epochs_after_the_checkpoint(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text("".join(json.dumps({"epoch": e}) + "\n" for e in range(4)))
    kept = restore_history(path, completed_epochs=2)
    assert [json.loads(line)["epoch"] for line in kept] == [0, 1]


def test_history_is_empty_on_a_fresh_run(tmp_path):
    assert restore_history(tmp_path / "history.jsonl", 0) == []
