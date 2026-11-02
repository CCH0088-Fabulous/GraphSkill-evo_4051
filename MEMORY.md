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
- `scripts/run_formal.py` snapshots both course configurations once, then uses copies for each condition with one temporary model service. It runs six member-A comparisons and two same-seed selected-skill validation evaluations, with test inference disabled.

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
- Current HotpotQA is a self-fixed official-public distractor subset with `course_split_verified=false`. Course alignment is unconfirmed, not proven wrong. Have C confirm equivalence or acceptance before investing in more repetitions.
- C still needs the complete development failure inventory and diagnoses; B still needs the improved selector and focused ablation. Team method finalization precedes final held-out testing. B/C independently reproducing a small run from README has not yet been evidenced.
- The user confirmed the experiment seed set `[0, 42, 127]` on 2026-11-02, replacing the prior suggested set. Seed 42 is complete; 0 and 127 remain unrun. This request changes the plan and asks for pre-experiment analysis; it does not request starting GPU experiments. Root YAML comments record the seed set; `experiment.seed` remains the single-run integer with default 42.
- For multi-seed execution, vary `experiment.seed`, not data-selection seeds or fixed IDs. Reuse the existing seed-42 batch only after verifying compatible model/data/protocol/method semantics; protocol changes require a fresh matched set. Run all A conditions at each seed; B's improvement/ablation should use the same seed list and budgets. Report every seed, means/sample standard deviations and paired differences; repeated use of 30 examples does not create 90 independent examples.
- Extra seeds do not resolve validation selection bias, small sample uncertainty, or unconfirmed course data alignment. Stay within the PDF's dataset and evolution caps.

## Git and handoff

- Remote: `https://github.com/CCH0088-Fabulous/GraphSkill-evo_4051.git`. Remote history was restored; verify the current branch and staging state before Git actions.
- `.gitattributes` uses LF for ordinary text and PowerShell, CRLF for batch scripts, and preserves exact bytes for archived manifests, split manifests, locks and binary PDFs.
- `.gitignore` excludes environments, secrets, caches, datasets, weights, restorable upstream checkouts and experiment outputs. Raw results are not included in a normal Git push; copy a complete batch directory separately for B/C.
- Keep this memory concise as the project progresses: update current status and point to detailed reports instead of copying logs or storing secrets.
