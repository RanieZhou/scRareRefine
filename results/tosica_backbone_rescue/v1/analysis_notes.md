# TOSICA-backbone rescue notes

- Closed ledger successful runs: 32/32; failures: 0.
- Runs requested in this invocation: 32.
- TOSICA was trained only on labeled training cells and supplied its native 48-dimensional CLS latent.
- No scANVI latent, probability, or predicted label entered training, prototypes, rank selection, tau calibration, or test rescue.
- Rare-F1 wins/ties/losses after fixed rescue: 15/15/2.
- Empirical alpha violations: 2.
- This is a seed-42, eight-dataset portability screen and does not establish universal backbone independence.
