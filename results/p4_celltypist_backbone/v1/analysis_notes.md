# P4 CellTypist-backbone notes

- Completed 9/9 prespecified runs.
- CellTypist was trained only on labeled training cells; its native multiclass decision scores supplied the prototype space.
- Rank and conformal tau used validation only; test labels were used only for frozen final metrics.
- F1 wins/ties/losses: 5/1/3.
- Empirical alpha violations: 0; abstentions: 1.
- This is a single-seed, three-dataset portability screen, not evidence of universal backbone independence.
