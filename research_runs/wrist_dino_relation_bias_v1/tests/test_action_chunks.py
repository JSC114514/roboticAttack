"""Actual Parquet boundaries, strict chunk validation and the real clean bank."""
import numpy as np
import pandas as pd
import pytest

from common import decode_action_chunk, source_run, validate_trace_actions


def test_parquet_nested_action_slots_keep_values_and_order(tmp_path):
    expected=np.arange(56,dtype=np.float64).reshape(8,7)/37.
    path=tmp_path/"actions.parquet"
    pd.DataFrame({"predicted_environment_action_chunk":[expected.tolist()]}).to_parquet(path,index=False)
    value=pd.read_parquet(path).iloc[0].predicted_environment_action_chunk
    assert value.shape==(8,) and value.dtype==object
    with pytest.raises((TypeError,ValueError)):
        np.asarray(value,dtype=np.float32).reshape(8,7)
    result=decode_action_chunk(value,"round_trip")
    assert result.dtype==np.float32 and result.shape==(8,7)
    assert np.array_equal(result,expected.astype(np.float32))
    assert np.array_equal(decode_action_chunk(expected),result)
    assert np.array_equal(decode_action_chunk(expected.tolist()),result)


@pytest.mark.parametrize("value",[
    np.zeros((7,8)),np.zeros(56),np.zeros((7,7)),
    [[0.]*7]*7+[[0.]*6],np.full((8,7),np.nan),np.full((8,7),np.inf),
    np.full((8,7),1e50),None,
])
def test_bad_chunks_fail_in_cpu_with_context(value):
    with pytest.raises(RuntimeError,match="Invalid saved action chunk.*test_episode"):
        decode_action_chunk(value,"test_episode")


@pytest.mark.parametrize("executed",[0,9,1.5,np.nan])
def test_cpu_preflight_rejects_invalid_execution_count(executed):
    frame=pd.DataFrame({"predicted_environment_action_chunk":[np.zeros((8,7))],"executed_slots":[executed]})
    with pytest.raises(RuntimeError,match="Invalid executed_slots"):
        validate_trace_actions(frame)


def test_all_real_clean_chunks_pass_and_match_float32_slot_values():
    traces=pd.read_parquet(source_run()/"results/LONG4_TRACES.parquet",
                           filters=[("condition","==","clean")],
                           columns=["suite","task_id","state_index","decision_block","executed_slots",
                                    "predicted_environment_action_chunk"])
    assert len(traces)==8062
    report=validate_trace_actions(traces)
    assert report["status"]=="PASS" and report["chunks_checked"]==8062
    for value in traces.predicted_environment_action_chunk:
        decoded=decode_action_chunk(value)
        for slot in range(8):
            assert np.array_equal(decoded[slot],np.asarray(value[slot],dtype=np.float32))
