# Repository memory

Last updated: 2026-11-02 (Asia/Shanghai).

This file preserves context for future work. Read it together with `AGENTS.md`.
Current user instructions and verified repository state take precedence over stale notes.
Keep configuration values in `config.yaml`; keep measurements in their original run artifacts.

## User preferences

- Communicate primarily in Chinese, with direct explanations and clear completion status.
- Keep Python modules focused on one responsibility. CLI parses and dispatches; experiment modules compose components; model I/O, data, scoring, storage and visualization stay separate.
- Keep the repository concise for the teacher's code interview: retain basic checks and real experiment entry points, consolidate GGUF, and remove superseded debugging tools. README commands use `python` in the user's activated environment.
- Use native Windows by default. Do not install or enable WSL. Existing Linux is an optional route.
- Do not download model weights without an explicit request. Bootstrap and serving must use existing local weights; non-weight assets require an explicit allowlist.
- During long GPU experiments, waiting is acceptable. Use GPT-6 luna for model-run monitoring when delegation is available; keep monitoring read-only and avoid frequent empty progress updates.
- HotpotQA v2 development comparison is complete. The held-out test was authorized, but the latest user instruction explicitly stops experiments and requests analysis of Linux and parallel inference only. Do not restart any service, test run, install packages, or begin migration until a new action request. Preserve test outputs and do not feed cases/results back to evolution, selectors, prompts, tuning or skill selection. Root config stays test-disabled.
- After modifying code, configuration or documentation, finish the user-facing response with a suggested Git commit message. This preference does not request an automatic commit or push. For a purely analytical answer, omit the commit suggestion.

## Project and source of truth

- The user is member A of the GraphSkillEvo course project. `nlp-project5.pdf` defines course requirements; `group_div.md` defines team responsibilities and planned milestones.
- Member A owns reproduction infrastructure and original baselines. Member B owns the improved failure selector and ablation. Member C owns scoring and analysis. An unavailable improved selector must fail explicitly rather than fall back to the original selector.
- `config.yaml` is the shared settings entry point: model/service, decoding, data preparation, dataset paths, experiment presets and budgets. `configs/*.yaml` mostly select named presets for backward-compatible commands.
- Example selection: `python -m repro show-config --config config.yaml --preset course --preset searchqa.course`.
- Environment overrides and `.env` can change local model settings; inspect effective settings before running. Credentials belong in the ignored `.env`, not shared YAML.
- Source locks, dependency locks and fixed data manifests remain separate because their own tools verify them. Keep `third_party/GraphSkillEvo` at `third_party/upstream.lock.json`; put adaptations in `src/repro/integration`, not upstream source.
- Historical experiment v1 is archived, not automatically enforced. The user explicitly removed its global path/environment lock so teammates can use different machines. Do not restore that implicit lock. Fixed data IDs, revisions, raw-data checksums and partition boundaries remain protected.
- Current Transformers implementation uses BF16 / SDPA on logical `cuda:0`; these are implementation constants, not selectable YAML options. Hardware differences must be recorded and must support the configured workload.

## Local environment and model

- Working directory: `E:\GraphSkill-evo_4051`; PowerShell; project interpreter `.venv\Scripts\python.exe` (Python 3.13.11 at last verification).
- The current complete model is user-provided `E:\Qwen3.5-9B`, selected by `model.local_path`. The repository's `models/Qwen3.5-9B` contains earlier non-weight assets and is not the complete model directory.
- GGUF is an optional backend selected by the `local.gguf` preset. Its single entry point is `scripts/gguf.py inspect|install|serve`; see `docs/gguf.md`. The installer verifies the existing runtime lock without rewriting it or fetching weights. It is not the model used for the completed formal comparisons.
- Model upstream download revision remains unverified; completed runs contain full weight/tokenizer/config fingerprints. Do not substitute the non-weight cache revision as proof of the complete model's source.

## Experimental boundaries

- Every generation request, including initialization, failures and retries, must pass through the metered client. Missing usage remains unknown rather than zero.
- Evolution receives training and validation examples only. Test records may be checked for integrity, but held-out test questions must not inform optimization or method development.
- Keep the original, improved and ablation conditions on matching model settings, execution/generation limits and call/token budgets. Report actual usage as well as limits.
- Truncated QA outputs retain the raw response, have an empty scored answer and EM/F1 zero; no automatic continuation or token-limit increase. Truncated optimizer outputs are rejected and retried within the existing attempt budget.
- Preserve completed runs and failed attempts. New runs get unique output directories. Synthetic/offline checks are not research results.
- `scripts/run_formal.py` snapshots the effective plan once, then uses copies for each condition with one temporary model service. Its default route retains six old course comparisons and two selected-skill repeats. `--provisional-hotpotqa --seeds 0 42 127` instead runs new stratified HotpotQA, isolating each seed's three A conditions and selected-skill repeat; test inference stays disabled.

## Completed formal validation batch

- Batch: `outputs/formal/20261101T135059023401Z-163af6/`, completed 2026-11-01. All eight subruns completed; each original optimizer completed three rounds.
- Seed: 42 only. Each dataset has fixed 100 train / 30 validation / 50 held-out test examples. The test subsets have not been evaluated by the model.
- SearchQA EM/F1 percentages: no skill 50.00/64.33; initial skill 66.67/72.52; selected GraphSkillEvo skill 66.67/72.52. Selection returned the initial skill `g0000_i0000`.
- HotpotQA EM/F1 percentages: no skill 83.33/88.33; initial skill 86.67/91.56; selected GraphSkillEvo skill 90.00/92.22. Selection returned round-two `g0002_i0000`, from `crossover_other`.
- Actual usage: 830 requests, 1,586,121 input tokens, 111,625 output tokens, about 64m17s wall time, device-wide sampled peak 25,809 MiB. HTTP failures and unknown/pending usage were zero. Five truncated predictions were correctly scored zero. The temporary server was closed after completion; verify current service state before any new launch.
- `docs/formal-results.md` is the readable report. Batch `summary.json`, `audit-results.json` and `results.json` hold evidence. `handoff.md` has portable relative links; `handoff.json` retains original absolute provenance paths. The original selected skill and repeated-evaluation archive can have different raw hashes because of CRLF/LF normalization; decoded text was verified equal.
- The validation set was used for selection. Same-seed `selected_val` is a repeated evaluation, not an independent seed replicate. HotpotQA's gain is one net correct answer on 30 validation examples; do not claim established held-out generalization or original-paper numerical reproduction.
- Last code checks after adding the formal runner: 208 tests passed and Ruff passed. These are historical checks, not a substitute for checking later changes.

## Repository cleanup

- Removed obsolete architecture probes, setup-report generation, non-weight asset preparation, redundant bootstrap scripts, and the unverified Linux/vLLM shell route. Installation now follows `requirements.txt`; native service diagnostics use the active interpreter instead of a fixed environment folder.
- Retired v1 publication/strict verification commands, their runtime hooks and acceptance-only presets. Pilot can use one to three rounds through the normal override while preserving real-data and no-test checks. Current settings and truncation/retry rules remain in the shared configuration and runtime.
- Moved original pilot/v1 reports to `docs/archive/`; removed the duplicate team-plan document in favor of root `group_div.md`. Historical manifests and completed outputs remain original evidence. Old configuration snapshots may contain retired fields; use current documented configuration entry points for new runs.
- Cleanup verification: 202 tests passed, Ruff and dependency checks passed; both synthetic full-loop smoke checks and both course data validations passed. The consolidated GGUF inspector validated the existing local file without loading it. A hash receipt in `.cache/verification/cleanup-preserved-artifacts.json` verified 5,979 original evidence files unchanged; only mutable `manifests/README.md` documentation was excluded.

## Outstanding work and recommendations

- PDF Project 5 explicitly allows a model such as Qwen3.5-9B. It requires fixed-test EM/F1/latency and preservation of the provided data partitions; it does not prescribe a minimum number of random seeds.
- The professor supplied `AIAA_4051_Hotpot_QA_Data_in_Evolving_Graph-Structured_Skills_for_LLM_Agents_(1).md`. It explicitly permits constructing project splits from official distractor train/dev, stratifying train/val by easy/medium/hard, retaining all context, and submitting original IDs. The former question about another private course dataset is resolved. Its Test sampling/last paragraph say 200 but table Total says 1000. The user first authorized 300/150/200, then explicitly paused experiments and specified test=1000. Current settings are 300/150/1000; retain the original professor file and record the user-specified resolution instead of editing it. The overall size versus PDF 100/30/50 cap remains provisional.
- C still needs the complete development failure inventory and diagnoses; B still needs the improved selector and focused ablation. Team method finalization precedes final held-out testing. B/C independently reproducing a small run from README has not yet been evidenced.
- Experiment seeds `[0,42,127]` are fixed. HotpotQA v2 now has all 12 subruns completed, with three original evolution rounds per seed; old v1 remains interrupted. SearchQA's fixed data and prior single-seed result are unchanged. Root YAML comments record the seed set; `experiment.seed` remains the single-run integer with default 42.
- For multi-seed execution, vary `experiment.seed`, not data-selection seeds or fixed IDs. Reuse the existing seed-42 batch only after verifying compatible model/data/protocol/method semantics; protocol changes require a fresh matched set. Run all A conditions at each seed; B's improvement/ablation should use the same seed list and budgets. Report every seed, means/sample standard deviations and paired differences; repeated use of 30 examples does not create 90 independent examples.
- Extra seeds do not resolve validation selection bias, sample uncertainty, or the unresolved size instructions. Keep the new larger user-authorized experiment explicitly provisional; the original `course` profile retains its PDF limits.

## Provisional stratified HotpotQA

- Current data: `data/processed/hotpotqa-stratified-v2/`, with 300 train (100 each difficulty), 150 val (50 each difficulty), and 1000 local held-out test from official distractor dev. Data sampling seed 42, pinned Hub revision `1908d6afbbead072334abe2965f91bd2709910ab`; ID-only SHA-256 ranking is independent of input ordering and model outcomes. All three original-ID sets are disjoint. The old v1 200-test split is retained unchanged; v2 train/val IDs and file bytes are identical, and its test set contains all old 200 IDs.
- Current submitted `train_ids`, `val_ids`, `test_ids` and data reference are in `manifests/hotpotqa-stratified-v2/`; v1 archives remain historical. Preparation uses only checksum-verified local Parquet cache and refuses overwrites; repeating it verifies existing files. The original professor document's SHA-256 and `1000_user_specified` resolution are recorded in v2.
- Shared settings are `presets.hotpotqa.provisional` plus new `preparation.hotpotqa_*` fields in root YAML. Three rounds, two new candidates, train batch 15, failure feedback 3, model/decoding/truncation/retry rules remain unchanged. Validation grew fivefold, so per-condition hard ceilings are 3,000 calls / 10,000,000 output tokens, shared by all conditions and future matching B comparisons.
- Code checks before launch: 235 tests passed, Ruff passed, actual data/preflight verified 100/100/100 train and 50/50/50 val plus 200 test, repeated preparation verified without replacing files. Original data/manifests and completed runs remain preserved. New development inference never sends the 200 test questions.
- Historical launch on 2026-11-02 at 04:30:41 UTC: `outputs/provisional-hotpotqa/20261102T043041541371Z-15a503/`, seeds `[0,42,127]`, PID 41364, owned server PID 40808. Both verified owned processes were stopped at the user's pause request; monitoring also stopped. GPU memory returned to about 0.9 GiB. Launch receipt remains `.cache/experiments/hotpotqa-20261102T043041189Z.launch.json`; `pause.json` records the interruption. Do not automatically resume or treat the batch as complete.
- Paused batch: seed0/no_skill and initial_skill completed (150 validation predictions each); original is incomplete, with one started call lacking completion/usage. Preserve that unknown usage and all existing logs, not zero-filled estimates. No test inference occurred. Current v2 needs a new run if the user requests resumption; interrupted optimizer state has no implemented in-place resume.
- After the test-count change: 237 tests and Ruff passed, current data/preflight passed, and v1 evidence remains unchanged. `.cache/verification/pre-test1000-evidence.json` contains the preservation receipt for 8,768 original manifest/output files, including the paused batch.
- Resumption preflight: effective Qwen3.5-9B settings, 300/150/1000 fixed manifests/ID lists, difficulty balance, instruction hash and disabled test inference all verified; 74 focused tests passed. Model directory, pinned upstream, CUDA and dependencies passed doctor; no service was occupying the model port before launch. All conditions are rerun in a new batch on v2, with fresh outputs and one shared temporary service.
- Completed v2 batch: `outputs/provisional-hotpotqa/20261102T050827834273Z-ae48d3/`, 2026-11-02 approximately 13:08 to 18:27 Asia/Shanghai, all 12 subruns completed. Temporary service closed automatically; PIDs 45328/11240 are historical. Launch receipt remains `.cache/experiments/hotpotqa-v2-20261102T050827423Z.launch.json`. The old paused batch and its unknown pending usage are unchanged.
- Current report: `docs/hotpotqa-v2-results.md`. Machine-readable audit, per-seed CSV, portable handoff and recomputation script: `outputs/reports/hotpotqa-v2/20261102T105531089060Z-76f4bd/`. All 32,767 source batch files were hashed and verified unchanged during aggregation. Report outputs are ignored by Git; handoff needs the full raw batch and report.
- Validation mean/sample SD in percent: no_skill EM 69.7778/1.0184, F1 80.7690/0.9421; initial EM 68.0000/1.3333, F1 78.8835/1.7322; selected EM 70.8889/1.0184, F1 82.3563/1.3876. Selected versus initial: +2.8889 EM and +3.4728 F1 percentage points; versus no_skill +1.1111/+1.5873. Initial EM is below no_skill for every seed; do not omit that finding. Selected skills: seed0 g0001_i0001 mutation_graph; seed42 g0002_i0000 crossover_other; seed127 g0003_i0000 mutation_other.
- Completed-batch audit: 4,647 calls (27 optimizer, 4,620 target), 9,647,908 input and 523,542 output tokens; zero failed/pending/unknown usage or HTTP retries, server POST count matches ledger. Four truncated QA predictions scored zero correctly; selected_val has none. Wall time 19,120.749 seconds (5h18m41s); device-wide sampled peak 27,359 MiB (26.72 GiB), not exclusive model memory or kernel time. All 12 model/data/source/environment snapshots match, every target ID belongs to train/val, and score recomputation agrees with saved results.
- Development conclusion: adapted original-method workflow and three-seed development baseline reproduction succeeded. Improvements over no_skill are small, validation was used for selection, and the same 150 examples are reused across seeds. The development report alone does not establish held-out generalization, statistical significance, full course completion, or original-paper numerical matching. The development batch never queried the test split; the separately authorized A test evaluation is tracked below. B's selector/ablation and C's failure/mechanism analysis remain team work.

## Frozen member-A held-out evaluation

- The user requested test evaluation after reviewing the development summary. Scope is all 1000 fixed test IDs, all three seeds and A's three conditions: nine runs, 9000 question evaluations. B's improved selector and ablation are not implemented or included; they must not use these test cases/failures/results as development feedback.
- Entry point: `python scripts/run_formal.py --test-from-batch outputs/provisional-hotpotqa/20261102T050827834273Z-ae48d3`. Test mode rejects configuration, parameter and seed overrides; it uses completed source plans, verifies all source subruns and model/data/skill identities, then snapshots initial/selected skills and data before launching one owned model service. No optimizer is called.
- Model, decoding, retry/truncation and budget limits remain those of development (3000 calls / 10,000,000 output tokens per condition). With max_retries=1, the 1000-example test fits these limits without raising them. Current archive checks confirmed the source plan, summaries, model identities and archived skills match the development report's checksums.
- Prelaunch checks: 254 full tests and Ruff passed, frozen-source preflight matched all selected skills and model fingerprint. Current held-out batch: `outputs/heldout-hotpotqa/20261102T114602167274Z-34728d/`, launched 2026-11-02 at 19:45 Asia/Shanghai, main PID 34732 and owned model server PID 45496 at launch. Receipt: `.cache/experiments/hotpotqa-test-20261102T114553034Z.launch.json`, with corresponding parent stdout/stderr logs. Verify live process state on subsequent turns rather than assuming PID identity persists.
- Test batch is now paused at the user's explicit request. Both identity-verified processes 34732 and 45496 were stopped; port8000 is closed and GPU memory returned to about 1 GiB. GPT-6 luna monitoring also stopped. `pause.json` records completed seed0/no_skill (1000 predictions), partial initial_skill (965 predictions), and one pending request with unknown usage; do not invent its result or count usage as zero. Other seven runs did not start. All original logs/frozen inputs remain unchanged; no automatic restart.
- Read-only performance diagnosis: server stderr lines6-9 confirm missing `causal_conv1d` and `flash-linear-attention`, causing Qwen3.5's conv/delta-rule operations to fall back to slower reference PyTorch paths. Model has24 linear-attention layers and8 full-attention layers; SDPA does not accelerate the missing linear-attention kernels. Current serving is one locked, single-input generation stream; client iterates sequentially. Completed no_skill test averaged4.236s/request,121.6 output tokens, about28.7 output tokens per request-second including prefill; non-request gaps totaled only7.3s of4243.4s wall. Exact kernel/CPU/memory bottleneck proportions were not profiled.

## Git and handoff

- Remote: `https://github.com/CCH0088-Fabulous/GraphSkill-evo_4051.git`. Remote history was restored; verify the current branch and staging state before Git actions.
- `.gitattributes` uses LF for ordinary text and PowerShell, CRLF for batch scripts, and preserves exact bytes for archived manifests, split manifests, locks and binary PDFs.
- `.gitignore` excludes environments, secrets, caches, datasets, weights, restorable upstream checkouts and experiment outputs. Raw results are not included in a normal Git push; copy a complete batch directory separately for B/C.
- Keep this memory concise as the project progresses: update current status and point to detailed reports instead of copying logs or storing secrets.
