# Synthetic fixtures

These hand-written examples are infrastructure fixtures, **not SearchQA or HotpotQA
benchmark samples**. Every ID begins with `synthetic-`, every record has
`metadata.synthetic=true`, and every context is marked `[SYNTHETIC]`.

Each fixture dataset has six training and three validation records. The empty test
partition is intentional: offline infrastructure validation must not be confused with
held-out research results. Mock-server outputs and their metrics are synthetic too.
