# ABC_IUEH research implementation

Read README.md, docs/data_sources.md and docs/protocol.md before changing data,
splits, evaluation or publication behavior. Keep official constraints, team
choices and unresolved statements distinct. The 20-sample / 100 Hz / 100 ms
contradiction remains unresolved; preserve the 400 -> 20 graph interface.

Use wxyz unit quaternions. Split by user before windowing. Never train, fit
scalers, select models or pretrain on development/final users. Windows must stay
inside one gap-free recording segment. Final holdout evaluation requires a
locked configuration and is not part of pipeline smoke tests.

No raw/processed human recordings, local access attestations, private manifests,
credentials, model weights or environment dumps belong in Git. Data license is
separate from code licensing. Do not download BOXRR until its recipient review
requirements are resolved. Do not infer challenge eligibility from a permissive
license or a successful alternative-dataset pilot.

For the current paper goal, the user's newer instruction allows gpt-6.1-sol
or gpt-6-luna with xhigh reasoning, chosen by workload, and fork_turns="none".
At most coordinator + three workers; no nested delegation. Keep a recorded
role/ownership/acceptance contract and use independent research/code review.
