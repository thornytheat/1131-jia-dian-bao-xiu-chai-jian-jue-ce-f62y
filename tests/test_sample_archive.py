from dataclasses import replace

import pytest

from warrantycost.repair_cohorts.service import (
    archive_shift, latest_writebacks, training_gate,
    NEW_SAMPLE_ADOPTED, UNPUBLISHED_SAMPLE_REPLACED, PUBLISHED_SAMPLE_RETAINED,
    REPEATED_WRITEBACK_IGNORED, SUPERSEDED_WITHIN_BATCH,
    OPEN_OR_INVALID_REPAIR, LATE_WRITEBACK_CROSSING_SPLIT, AUTHORIZATION_UNPUBLISHED,
)
from warrantycost.repair_cohorts.contracts import RepairLog
from warrantycost.pipeline import run


def work(doc_id, day, *, product="洗碗机", closed=None, posted=None, price=24000, lost=0, firm=1):
    return RepairLog.parse(dict(work_order=doc_id, family=product, symptom="不加热", age_month=10,
                                service_minutes=35, accepted=day, closed=closed or day + 1,
                                authorized_published=posted if posted is not None else day + 2,
                                solved=1 - lost, cost_cents=price, firm=firm))


def decisions_of(archive):
    return {item.work_order: item.decision for item in archive.decisions}


def test_published_sample_keeps_old_version_on_rewrite():
    # TRAP-60
    base = [work("P-1", 10, price=20000), work("D-1", 12, price=21000)]
    rewrite_published = work("P-1", 10, price=29900)
    rewrite_draft = work("D-1", 12, price=21500)
    archive = archive_shift(base, {"P-1"}, [rewrite_published, rewrite_draft], 30,
                            base_version=3, min_series=6)
    by_order = {row.work_order: row for row in archive.records}
    assert by_order["P-1"].cost_cents == 20000
    assert by_order["D-1"].cost_cents == 21500
    assert decisions_of(archive) == {"P-1": PUBLISHED_SAMPLE_RETAINED,
                                     "D-1": UNPUBLISHED_SAMPLE_REPLACED}
    assert archive.version == 4


def test_new_completion_is_adopted_and_identical_rewrite_is_ignored():
    base = [work("A-1", 10)]
    fresh = work("A-2", 14)
    archive = archive_shift(base, set(), [fresh, work("A-1", 10)], 30, base_version=3, min_series=6)
    assert decisions_of(archive) == {"A-1": REPEATED_WRITEBACK_IGNORED, "A-2": NEW_SAMPLE_ADOPTED}
    assert [row.work_order for row in archive.records] == ["A-1", "A-2"]


def test_batch_superseded_writebacks_are_deterministic():
    # TRAP-61
    early = work("B-1", 20, closed=22, posted=23, price=25000)
    later = work("B-1", 20, closed=24, posted=25, price=25500)
    archive = archive_shift([], set(), [early, later], 30, base_version=3, min_series=6)
    again = archive_shift([], set(), [later, early], 30, base_version=3, min_series=6)
    assert archive == again
    assert archive.records[0].closed == 24
    assert sorted(item.decision for item in archive.decisions) == [NEW_SAMPLE_ADOPTED, SUPERSEDED_WITHIN_BATCH]
    superseded = [item for item in archive.decisions if item.decision == SUPERSEDED_WITHIN_BATCH]
    assert len(superseded) == 1 and superseded[0].work_order == "B-1"


def test_unclosed_unpublished_and_late_writebacks_never_enter_training():
    base = [work("C-1", 10)]
    batch = [work("C-2", 20, firm=0),
             work("C-3", 20, closed=30, posted=31),
             work("C-4", 20, closed=28, posted=30),
             work("C-5", 20, closed=28, posted=29)]
    archive = archive_shift(base, set(), batch, 30, base_version=3, min_series=6)
    reasons = {order: reason for order, _family, reason in archive.excluded}
    assert reasons == {"C-2": OPEN_OR_INVALID_REPAIR,
                       "C-3": LATE_WRITEBACK_CROSSING_SPLIT,
                       "C-4": AUTHORIZATION_UNPUBLISHED}
    assert [row.work_order for row in archive.training] == ["C-1", "C-5"]
    assert len(archive.records) == 5


def test_gate_boundaries_follow_the_split_day():
    assert training_gate(work("G-1", 20, closed=29, posted=29), 30) is None
    assert training_gate(work("G-2", 20, closed=30, posted=30), 30) == LATE_WRITEBACK_CROSSING_SPLIT
    assert training_gate(work("G-3", 20, closed=29, posted=30), 30) == AUTHORIZATION_UNPUBLISHED
    assert training_gate(work("G-4", 20, firm=0), 30) == OPEN_OR_INVALID_REPAIR


def test_same_batch_rerun_keeps_version_sample_count_and_threshold():
    base = [work("R-1", 10), work("R-2", 11)]
    batch = [work("R-3", 12), work("R-2", 11, price=19900)]
    first = archive_shift(base, set(), batch, 30, base_version=3, min_series=6)
    rerun = archive_shift(base, set(), list(reversed(batch)), 30, base_version=3, min_series=6)
    assert (first.version, first.sample_count, first.min_series, first.digest) == \
           (rerun.version, rerun.sample_count, rerun.min_series, rerun.digest) == (4, 3, 6, first.digest)
    replay = archive_shift(first.records, set(), batch, 30, base_version=first.version, min_series=6)
    assert replay.version == first.version
    assert replay.digest == first.digest
    assert replay.sample_count == first.sample_count
    assert set(decisions_of(replay).values()) == {REPEATED_WRITEBACK_IGNORED}


def test_empty_batch_mints_no_new_version():
    base = [work("E-1", 10)]
    archive = archive_shift(base, set(), [], 30, base_version=3, min_series=6)
    assert archive.version == 3
    assert archive.sample_count == 1
    assert archive.decisions == ()


def test_refit_threshold_below_model_floor_is_rejected():
    with pytest.raises(ValueError, match="REFIT_THRESHOLD"):
        archive_shift([work("T-1", 10)], set(), [], 30, base_version=3, min_series=5)


def test_conflicting_base_archive_is_rejected():
    with pytest.raises(ValueError, match="BASE_ARCHIVE_CONFLICT"):
        archive_shift([work("T-1", 10), work("T-1", 10, price=22000)], set(), [], 30,
                      base_version=3, min_series=6)


def test_sparse_family_falls_back_to_previous_model_version():
    # TRAP-62
    result = run(9)
    series = {row["产品组"]: row for row in result["stages"][2]["records"] if "授权模型" in row}
    assert series["消毒柜"]["授权模型"] == "v3 沿用上一版"
    assert series["消毒柜"]["可训练样本"] < series["消毒柜"]["重拟合门槛"]
    assert series["洗碗机"]["授权模型"] == "v4 重拟合"
    scored = {row["保修单"]: row for row in result["stages"][3]["records"] if "保修单" in row}
    assert scored["CS-087-020"]["模型版本"] == "v3 沿用上一版"


def test_archive_version_row_is_stable_across_rerun_and_capacity():
    first, again, other = run(4), run(4), run(40)
    rows = []
    for result in (first, again, other):
        row, = [item for item in result["stages"][2]["records"] if "入档版号" in item]
        rows.append(row)
    assert rows[0] == rows[1] == rows[2]
    assert rows[0]["入档版号"] == rows[0]["上一版号"] + 1
    assert rows[0]["可训练样本数"] == first["metrics"]["训练样本"]
    assert first["metrics"]["入档版号"] == rows[0]["入档版号"]
