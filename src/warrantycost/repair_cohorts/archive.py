from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from json import dumps

NEW_SAMPLE_ADOPTED = "NEW_SAMPLE_ADOPTED"
UNPUBLISHED_SAMPLE_REPLACED = "UNPUBLISHED_SAMPLE_REPLACED"
PUBLISHED_SAMPLE_RETAINED = "PUBLISHED_SAMPLE_RETAINED"
REPEATED_WRITEBACK_IGNORED = "REPEATED_WRITEBACK_IGNORED"
SUPERSEDED_WITHIN_BATCH = "SUPERSEDED_WITHIN_BATCH"

OPEN_OR_INVALID_REPAIR = "OPEN_OR_INVALID_REPAIR"
LATE_WRITEBACK_CROSSING_SPLIT = "LATE_WRITEBACK_CROSSING_SPLIT"
AUTHORIZATION_UNPUBLISHED = "AUTHORIZATION_UNPUBLISHED"

MODEL_FLOOR = 6


@dataclass(frozen=True, slots=True)
class Arbitration:
    work_order: str
    family: str
    decision: str

    def evidence(self) -> dict:
        return {"work_order": self.work_order, "family": self.family, "decision": self.decision}


@dataclass(frozen=True, slots=True)
class ArchiveVersion:
    version: int
    base_version: int
    digest: str
    records: tuple
    training: tuple
    excluded: tuple
    min_series: int
    series_counts: tuple
    decisions: tuple

    @property
    def sample_count(self) -> int:
        return len(self.training)


def _writeback_key(row):
    return (row.closed, row.authorized_published, row.accepted, row.cost_cents,
            row.service_minutes, row.age_month, row.solved, row.firm, row.family, row.symptom)


def latest_writebacks(records):
    groups = defaultdict(set)
    for row in records:
        groups[row.work_order].add(row)
    winners = []
    superseded = []
    for work_order in sorted(groups):
        ordered = sorted(groups[work_order], key=_writeback_key)
        winners.append(ordered[-1])
        superseded.extend(Arbitration(work_order, row.family, SUPERSEDED_WITHIN_BATCH) for row in ordered[:-1])
    return winners, superseded


def training_gate(record, on_day):
    if not record.firm or record.closed < record.accepted:
        return OPEN_OR_INVALID_REPAIR
    if record.closed >= on_day:
        return LATE_WRITEBACK_CROSSING_SPLIT
    if record.authorized_published >= on_day:
        return AUTHORIZATION_UNPUBLISHED
    return None


def archive_shift(base_records, published, writebacks, on_day, *, base_version, min_series):
    if min_series < MODEL_FLOOR:
        raise ValueError("REFIT_THRESHOLD_BELOW_MODEL_FLOOR")
    base = tuple(sorted(base_records, key=lambda row: row.work_order))
    if len({row.work_order for row in base}) != len(base):
        raise ValueError("BASE_ARCHIVE_CONFLICT")
    locked = frozenset(published)
    base_by_order = {row.work_order: row for row in base}
    winners, superseded = latest_writebacks(writebacks)
    decisions = list(superseded)
    merged = dict(base_by_order)
    for row in winners:
        prior = base_by_order.get(row.work_order)
        if prior is None:
            merged[row.work_order] = row
            decision = NEW_SAMPLE_ADOPTED
        elif prior == row:
            decision = REPEATED_WRITEBACK_IGNORED
        elif row.work_order in locked:
            decision = PUBLISHED_SAMPLE_RETAINED
        else:
            merged[row.work_order] = row
            decision = UNPUBLISHED_SAMPLE_REPLACED
        decisions.append(Arbitration(row.work_order, row.family, decision))
    records = tuple(sorted(merged.values(), key=lambda row: row.work_order))
    version = base_version + (records != base)
    training = []
    excluded = []
    for row in records:
        reason = training_gate(row, on_day)
        if reason is None:
            training.append(row)
        else:
            excluded.append((row.work_order, row.family, reason))
    counts = defaultdict(int)
    for row in training:
        counts[row.family] += 1
    digest = sha256(dumps([row.evidence() for row in records], ensure_ascii=False,
                          sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    return ArchiveVersion(version=version, base_version=base_version, digest=digest, records=records,
                          training=tuple(training), excluded=tuple(excluded), min_series=min_series,
                          series_counts=tuple(sorted(counts.items())),
                          decisions=tuple(sorted(decisions, key=lambda item: (item.work_order, item.decision))))
