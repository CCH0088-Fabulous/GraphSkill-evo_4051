# Data preparation

The official repositories contain code and manifests rather than bundled model
weights. Preparation below downloads only dataset records, never model weights.

```powershell
python scripts/prepare_searchqa.py
python scripts/prepare_hotpotqa.py
```

Both commands read their dataset revision, cache location, seed, counts and course
output directory from the root `config.yaml`. Defaults are 100 train / 30 validation /
50 local held-out test examples. `--train`, `--val`, `--test`, `--seed`, `--revision`
and other explicit options override those values before inspecting outcomes.
Data outputs contain `{train,val,test}.jsonl` and `split_manifest.json`.

The default `--download-mode parquet` resolves and records a fixed dataset commit,
downloads only the required public dataset configuration, and caches raw files under
`data/raw`. Downloads resume `.part` files and retry transient errors. Completed raw
files are reused after size/SHA-256 verification. SearchQA raw files total about
325 MB; HotpotQA distractor raw files total about 359 MB. No models are downloaded.
Normalized outputs remain immutable unless `--overwrite` is supplied explicitly.
The course configurations compare the rebuilt manifests with `manifests/data-v1/`
for exact IDs, partitions, revisions and source/data checksums, while allowing a
different local upstream checkout path.

SearchQA uses `third_party/GraphSkillEvo/data/searchqa_id_split` as its partition
recipe, matching each manifest ID to `key` in
[lucadiliello/searchqa](https://huggingface.co/datasets/lucadiliello/searchqa).
Use `--raw path/to/source.jsonl` for local source data, or `--raw` with a directory
of source Parquet shards. Use `--revision COMMIT_SHA` to rebuild from a named
revision; `--download-mode streaming` is also available.

HotpotQA is sourced from the official
[hotpotqa/hotpot](https://github.com/hotpotqa/hotpot) dataset, using its
[Hugging Face dataset](https://huggingface.co/datasets/hotpotqa/hotpot_qa) in
`distractor` mode. The custom local split draws train and
validation from disjoint official training records, and draws local test from
official labeled dev. This is recorded as `course_split_verified=false`. The
public official test file has no labels and is not silently used for local EM/F1.

For public HotpotQA, row offsets are selected with a fixed seed before record
inspection. The same offsets are preserved across the Parquet and optional Viewer
backends; a transient download failure never resamples identities. Hub revision,
raw file hashes, offsets, normalized file hashes, and final IDs are all retained.
The optional `--download-mode viewer` fetches small cached slices with retries;
the Parquet backend avoids Viewer query timeouts and request-rate limits.

To use the official JSON files linked from GitHub, pass `--raw-train` and `--raw-val`
(the latter is the official dev file in public mode). To preserve supplied course
boundaries, use `--source course --raw-train TRAIN --raw-val VAL --raw-test TEST`.

Normalized records preserve original source fields under `metadata.source_record`.
Only question and context belong in model task inputs; reference answers and
supporting facts must not be included as task evidence. Evaluation code is in
`src/repro/evaluation` and is the member C integration boundary.

## Professor's stratified HotpotQA protocol

The professor's root-level [hotpot_split.md](../hotpot_split.md) permits constructing
fixed splits from official distractor data. The user now specifies 300 training
examples (100 per difficulty), 150 validation examples (50 per difficulty,
excluding training IDs), and 1000 held-out examples from official dev, replacing
the earlier provisional count of 200. The original instruction is retained
unchanged; the manifest records the user-selected count. The overall size remains
provisional relative to the original PDF limits. The user has explicitly resumed
development experiments on v2; the old interrupted v1 batch stays unchanged.

```powershell
python scripts/prepare_hotpotqa.py --stratified
python -m repro validate-data --config config.yaml --preset course --preset hotpotqa.provisional
```

This path verifies the already cached revision-pinned Parquet files and their
download receipts, then samples by original difficulty and original ID with data
seed 42. It does not download anything. The new normalized data lives under
`data/processed/hotpotqa-stratified-v2/`. Reference manifest and submitted
`train_ids`, `val_ids`, `test_ids` live under `manifests/hotpotqa-stratified-v2/`.
Existing outputs are verified instead of overwritten. The v1 stratified split
with 200 test questions remains unchanged, as do the original unstratified
100/30/50 split and its results. V2 retains the exact train/validation data;
its 1000-question test set contains all 200 previous test IDs.

Historical manifests retain the instruction's original filename. Preparation and
development preflight verify the current configured instruction by its exact
SHA-256, allowing a filename change without rewriting historical manifests or
resampling IDs. Any content or other source-metadata change is still rejected.

Experiment seeds 0, 42 and 127 share this one fixed split. Test records are reserved
for evaluation after team method finalization; development experiments never send
them to the model. Solver inputs retain every supplied context paragraph and omit
reference answers and supporting-fact labels.
