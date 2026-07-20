# Project status

Last updated: 2026-07-19

## Current mission

Mission 005 — load and audit PTB-XL 100 Hz signals.

- Issue: `#13` — `[DATA] Load and audit PTB-XL 100 Hz signals`.
- Branch: `data/13-load-audit-lr-signals`.
- Contract: `docs/context/missions/005_load_audit_lr_signals.md`.

## Current step

Implementation, synthetic validation and the complete real-data audit pass.
The final local quality suite passes. Review and publish the versioned diff for
independent CI.

## Next actions

1. Review the versioned scope and confirm raw WFDB files remain ignored.
2. Commit, push and open a pull request closing issue `#13`.
3. Merge only after Python quality and GitGuardian pass, then close context.

## Mission 005 evidence

- Official `records100/`: 43,598 files; every SHA-256 verified.
- Real cohort audit: 21,388 loaded, zero missing and zero invalid.
- All signals: `(1000, 12)`, 100 Hz and finite numeric values.
- One consistent header order: `I, II, III, AVR, AVL, AVF, V1, V2, V3, V4,
  V5, V6`.
- Repeated audit report SHA-256:
  `8c5c5deed8d77eba5381ccb6e93e295b8d2c96a73d75014678f8c1218fc5040f`.
- Metadata, label and cohort hashes remain unchanged.
- No normalization, filtering, resampling, PyTorch or model decision was added.

## Last completed mission

Mission 004 — define the five-superclass modeling cohort:

- Issue `#10` closed.
- Pull request `#11` squash-merged as `b78d7ea`.
- Python quality and GitGuardian passed on the PR.
- Post-merge Quality workflow passed on `main`.
- Local and remote implementation branches were removed.

## Mission 004 evidence

- Explicit inclusion rule: at least one of the five target superclasses.
- Exact partition: 21,799 input = 21,388 included + 411 excluded.
- All-zero records remain unchanged in the master label table.
- Included patients: 18,617; excluded-only patients: 252.
- Train: 17,084 records / 14,823 patients.
- Validation: 2,146 records / 1,917 patients.
- Test: 2,158 records / 1,877 patients.
- Zero patient overlap after cohort filtering.
- No label, identity, fold or split value is recalculated or changed.
- Cohort, exclusions and report reproduce identical bytes.
- Processed tables remain ignored; the deterministic report is versioned.
- Full local suite passed: 44 tests, Ruff lint and Ruff format.
- No signal file, class balancing, threshold or model decision entered the
  mission.

## Stable repository foundation

- Reproducible Python 3.11 environment managed by uv.
- PTB-XL v1.0.3 identity, metadata, folds and patient isolation verified.
- Official five-superclass labels reproducibly constructed and audited.
- Initial modeling cohort explicitly defined without test-driven selection.
- GitHub Actions and GitGuardian green on `main`.
- Durable decisions live in `docs/context/DECISIONS.md`; completed contracts
  remain under `docs/context/missions/`.
