"""Reuse must be explicitly audited and must preserve exact artifact hashes."""
import pytest
import common
import json
from cache_reuse import resource_migration_matches

def test_a800_reuse_rejects_scientific_or_memory_budget_changes():
    proof=common.read_json(common.ROOT/"reports/CACHE_REUSE_SOURCE_e4387959_A800.json")
    original=proof["resource_migration_originals"]["config.json"]
    current=common.cfg()
    assert resource_migration_matches("config.json",original,json.dumps(current))
    current["epochs"]+=1
    assert not resource_migration_matches("config.json",original,json.dumps(current))
    current=common.cfg();current["resources"]["train"]["host_memory_gib"]=32
    assert not resource_migration_matches("config.json",original,json.dumps(current))

def test_a800_reuse_only_allows_exact_scheduler_migration():
    proof=common.read_json(common.ROOT/"reports/CACHE_REUSE_SOURCE_e4387959_A800.json")
    for relative,original in proof["resource_migration_originals"].items():
        if relative=="config.json":continue
        current=(common.ROOT/relative).read_text()
        assert resource_migration_matches(relative,original,current)
        assert not resource_migration_matches(relative,original,current+"\nexit 0\n")

def test_only_audited_cache_receipts_can_cross_source_revisions(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path);monkeypatch.setattr(common,"signature",lambda:"current")
    relative="data/cache/train/libero_object/task_00.json"
    payload=tmp_path/relative.replace(".json",".pt");payload.parent.mkdir(parents=True)
    payload.write_bytes(b"original frozen features")
    receipt=tmp_path/relative
    common.atomic_json({"status":"COMPLETE","signature":"old", "artifacts":{
        payload.relative_to(tmp_path).as_posix():common.sha(payload)}},receipt)
    with pytest.raises(RuntimeError,match="another protocol"):common.complete(receipt)
    manifest={"target_signature":"current","receipts":{relative:{"origin_signature":"old","sha256":common.sha(receipt)}}}
    common.atomic_json(manifest,tmp_path/"reports/CACHE_REUSE.json")
    assert common.complete(receipt)
    assert common.cache_shard_origin_matches({"signature":"old"},receipt)
    assert not common.cache_shard_origin_matches({"signature":"unrelated"},receipt)
    payload.write_bytes(b"tampered features")
    with pytest.raises(RuntimeError,match="artifact changed"):common.complete(receipt)

def test_manifest_cannot_authorize_a_different_target_revision(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path);monkeypatch.setattr(common,"signature",lambda:"current")
    receipt=tmp_path/"data/cache/train/s/task_00.json"
    common.atomic_json({"status":"COMPLETE","signature":"old","artifacts":{}},receipt)
    common.atomic_json({"target_signature":"wrong","receipts":{
        receipt.relative_to(tmp_path).as_posix():{"origin_signature":"old","sha256":common.sha(receipt)}}},
        tmp_path/"reports/CACHE_REUSE.json")
    with pytest.raises(RuntimeError,match="another protocol"):common.complete(receipt)
