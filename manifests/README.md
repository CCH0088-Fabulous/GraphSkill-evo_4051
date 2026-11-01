# Fixed experiment identities

Dataset preparation writes `split_manifest.json` alongside normalized data. Keep its
exact contents with the experiment, and copy it into each run's artifacts. It records
source metadata, the seed and selection rule, ordered example IDs, counts, and file
SHA-256 hashes. `repro.datasets.manifests.verify_manifest` verifies these records.

Published shared protocol v1 is preserved in `experiment-v1/manifest.json`, with
its former activation record in `experiment-v1-activation.json`. The current
course configurations do not enforce that historical protocol's local model path,
source files or environment. Those archival snapshots describe the prior acceptance;
they do not certify a current run. Version directories refuse overwrite.

The original course data manifests are separately retained in `data-v1/`. A course
run verifies its dataset files and compares IDs, partitions, source revisions and
checksums with the corresponding reference; the upstream checkout path may differ.
See `docs/frozen-settings-v1.md` for historical acceptance and truncation/retry rules.

The course profile defaults to at most 100 train, 30 validation, and 50 test examples
per dataset. GPU-hour planning is unrelated to this profile and is not enforced here.
The profile can be disabled explicitly for separately documented experiments.

SearchQA uses the upstream GraphSkillEvo split IDs. Public HotpotQA preparation uses
a **custom split**, disclosed with `course_split_verified=false`: official training
examples supply disjoint local train and validation subsets; labeled official dev
examples supply the local held-out test subset. Do not describe this as a supplied
course split. Switch to `--source course` when actual course partition files exist.

Default Parquet downloads use a pinned Hub revision and preserve original file
sizes and SHA-256 hashes under `source.raw_files`. Resumable raw files are kept in
`data/raw`; completed normalized outputs are not replaced by a retry.

Optional Dataset Viewer downloads record the Hub revision before and after retrieval and
hash the retrieved bytes. The Viewer API serves current data, so it is not a
revision-pinned historical API. Retain the materialized files and hashes. SearchQA
also supports revision-pinned HF streaming when stronger rebuild guarantees are
needed. Frozen outputs are never overwritten without an explicit `--overwrite`.
