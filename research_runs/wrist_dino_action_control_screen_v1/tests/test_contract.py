from collections import Counter

from common import config, load_records


def test_frozen_candidate_and_data_grid():
    cfg = config()
    assert len(cfg["candidates"]) == 27
    assert len(cfg["endpoints"]) == 2
    records = load_records()
    counts = Counter((row["pilot_split"], row["task_id"]) for row in records)
    assert len(records) == 40 and set(counts.values()) == {2}
