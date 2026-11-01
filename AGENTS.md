# Project conventions

- Keep Python modules focused on one responsibility. CLI parses and dispatches; experiments compose components; data, model I/O, scoring, storage, and visualization stay separate.
- Keep `third_party/GraphSkillEvo` at the commit in its lock file. Put documented shared adaptations in `src/repro/integration`.
- Default to native Windows. Do not install or enable WSL. Linux scripts are an optional route for an existing Linux system.
- Bootstrap and model-serving scripts must not fetch model weights. Downloading weights requires a user request covering them; non-weight tokenizer/config assets use an explicit allowlist.
- Preserve frozen IDs, source revisions, raw-data checksums, and partition boundaries. Evolution must not receive held-out test examples.
- Every model request, including initialization, revision, failure and retry, must pass through the metered client. Unknown usage stays explicitly unknown.
- Label synthetic data and offline responses clearly; they are infrastructure checks, not research results.
- Member B owns failure selection; member C owns scoring and analysis. An unavailable improved selector must fail instead of silently using the original.
- Use focused tests for changed experiment boundaries and run relevant existing checks. Do not edit upstream source or overwrite completed runs to make a check pass.
