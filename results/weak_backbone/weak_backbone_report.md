# Weak-Backbone Rescue Demo

Scope: kNN base predictions on the same scANVI latent space, followed by the unchanged validation-calibrated scRareRefine rescue.

Scarce-region summary:

| region | variant | n | f1_mean | recall_mean | precision_mean | rare_fp_rate_max | rescue_ffr_max | n_abstain | wins | ties | losses | worst_delta | best_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SCARCE | kNN | 54 | 0.7248 | 0.6506 | 0.9911 | 0.001221 | 0 | 0 |  |  |  |  |  |
| SCARCE | kNN+scRareRefine | 54 | 0.8603 | 0.8085 | 0.9734 | 0.009768 | 0.009768 | 26 |  |  |  |  |  |
| SCARCE | paired_gain | 54 | 0.1354 |  |  |  |  |  | 27 | 26 | 1 | -0.039 | 0.5229 |

Negative paired cells in scarce region:

| dataset | seed | rts | kNN | kNN+scRareRefine | delta |
| --- | --- | --- | --- | --- | --- |
| pancreas_integrated | 42 | 0.01 | 0.9778 | 0.9388 | -0.039 |

Negative paired cells across all rare_train_size settings:

| dataset | seed | rts | kNN | kNN+scRareRefine | delta |
| --- | --- | --- | --- | --- | --- |
| immune_dc | 44 | all | 0.9431 | 0.9385 | -0.0046 |
| pancreas_integrated | 42 | 0.01 | 0.9778 | 0.9388 | -0.039 |

Interpretation: the rescue mechanism transfers to a weaker predictor in aggregate, but this demo has one negative scarce-region cell and two negative cells overall. It should not be claimed as no-regression evidence.
