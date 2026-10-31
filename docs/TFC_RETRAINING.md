# Neural TF-C retraining on Sensor Logger (26,000 windows)

This replaces the deployed PCA/Logistic Regression baseline with a neural TF-C
pipeline. The HTML, CSS, JavaScript, images, authentication flow and API response
schema are unchanged. `app.py` selects the TF-C input adapter from the model's
`feature_version=tfc6_v1`.

## Authoritative method

Zhang, X., Zhao, Z., Tsiligkaridis, T., and Zitnik, M. (2022).
*Self-Supervised Contrastive Pre-Training For Time Series via Time-Frequency
Consistency*. NeurIPS 2022. https://arxiv.org/abs/2206.08496

- Project: https://zitniklab.hms.harvard.edu/projects/TF-C/
- Official code: https://github.com/mims-harvard/TFC-pretraining
- Reference commit: `5096675826e9ef234a9b01cc63d484c66cb0441bc0`
- Paper PDF consulted: https://zitniklab.hms.harvard.edu/publications/papers/TF-C-self_supervised_contrastive_main.pdf

This is an explicitly documented six-channel implementation of the paper, not a
claim to reproduce the original cross-dataset benchmark numbers or its exact
single-channel executable. It uses neither DINOv3 nor classical feature/PCA models.

## Source selected by the user

The supplied Kaggle dataset API listed latest v7, zero bytes, and downloading it
returned HTTP 404 on 2026-10-31. The user explicitly selected the previously
uploaded **Sensor logger(1).zip** instead of waiting for a repaired Kaggle upload.
Archive SHA-256:
`451c770fedf11b305a8d03ab5cb79140671d23686bcce6242443f0540918e643`.

No raw dataset or account information is committed to the repository. Supply the
archive locally using `--data`. `models/data_audit.json` records the source and
checks. The 26,000 windows contain synthetic/augmented derivatives of phone data;
they are not 26,000 independent phone recordings.

## Full pipeline

1. Validate exactly 20,800 unlabeled and 5,200 labeled windows, 178 x 6 values.
2. Validate the five labels and 1,040 labeled samples per class. Preserve labels;
   do not change or clean noisy labels based on test results. Never use the
   provenance manifest's `true_label` as a training target.
3. Check 24 CSV/NumPy windows against the supplied normalization metadata.
   Normalize each window/channel independently, identically during training and
   serving. Resample only inputs not already 178 long; long live recordings use
   the most energetic 178-step window. All six channels are retained.
4. Resolve known augmentation parents in the manifest. Split labeled groups into
   3,120 training, 1,040 validation, 1,040 held-out test windows with fixed seeds.
   Assert known ancestor groups do not overlap. Reject exact duplicate normalized
   windows or known ancestry between unlabeled pretraining and held-out samples.
5. Pre-train **both encoders and projectors**, from random initialization, using
   only the 20,800 unlabeled windows for 40 complete epochs. No labeled windows
   or target labels participate in this stage.
6. Load the pretrained encoder weights into the downstream model. Add a randomly
   initialized classifier and fine-tune the complete network on training labels
   for 40 epochs using cross-entropy. Select the checkpoint by validation macro F1.
7. Evaluate the held-out test set once after selection. Save predictions, confusion
   matrix, accuracy, macro F1, logs, checkpoint hashes and parameter-change proof.
8. Export a joblib-compatible neural inference adapter and both neural checkpoints.
   The web application executes PyTorch and uses the trained MLP softmax scores.

## Paper equations and upstream differences

- `LT`, `LF`: symmetric cosine NT-Xent at temperature 0.2, within each minibatch.
- Frequency view: magnitude of the full FFT along the temporal axis (178 bins).
- `STF=d(zT,zF)`. Paper Eq. 3:
  `LC=sum(STF - Spair + delta)` for `(zT,zF_aug)`, `(zT_aug,zF)`,
  `(zT_aug,zF_aug)`, with delta=1.
- Paper Eq. 4: `L=.2*(LT+LF)+.8*LC`. No undocumented hinge is added to Eq. 3.
  The consistency expression can be negative; negative total loss alone is not a
  numeric error or a classification accuracy.
- The upstream trainer computes `loss_c` but actually optimizes `l_TF` in its final
  assignment. We follow the stated paper objective, and test this explicitly.
- Upstream Transformers default to sequence-first but receive batch-first inputs.
  We explicitly set `batch_first=True`, preventing information exchange between
  independent samples in a batch. A regression test verifies single/batched
  inference equivalence.
- Two independent two-layer Transformers, each embedding dimension 178, two heads,
  feed-forward 356, dropout .1; the six channel tokens are flattened to 1,068
  features. Projectors: 1068 -> 256 (BatchNorm/ReLU) -> 128. Concatenate zT/zF;
  classifier: 256 -> 64 (sigmoid) -> 5, matching the official classifier structure.
- Temporal bank: jitter SD .2, scaling SD .1 about 1, circular shifts +/-18,
  four-segment permutation. Frequency bank chooses removal or addition with a
  10% bin budget and alpha=.5, as opposed to summing two full spectra upstream.
- Adam lr .0003, betas (.9,.99), batch 128, seed 42; clip gradient norm at 5.
  40 epochs is the upstream SleepEEG config default; these are explicit
  dataset-specific settings, not new benchmark claims. CPU execution is recorded.
- Unlike the upstream example trainer, gradients reset for every batch and test
  scores are not used for selecting an epoch. All pretraining samples are visited
  each epoch, including the final short minibatch.

## Reproduce

```bash
python -m pip install -r requirements.txt
python train_tfc.py --data "/path/Sensor logger(1).zip" --pre-epochs 40 --fine-epochs 40
python tests/test_tfc.py
python -m unittest test_sensor_model.py
python tests/test_application.py
node tests/test_auth_ui.js
```

PyTorch comes from its CPU wheel index. See `training_report.json` for the exact
runtime version and selected checkpoint. Training is a separate offline step;
Render does not train on startup or on every deployment.

## Artifacts and honest limitations

- `models/tfc_pretrained.pt`: neural encoder/projector checkpoint.
- `models/tfc_finetuned.pt`: selected full network checkpoint, with class order.
- `models/gesture_real_v1.joblib`: serving adapter; no Logistic Regression inside.
- `models/training_history.json`: full per-epoch log including loss components.
- `models/training_report.json`: final metrics, changes and artifact hashes.
- `models/evaluation_split.csv`: sample/group split manifest.
- `models/test_predictions.npz`: predictions only (not raw sensor windows).

The source lacks original session/person identifiers and complete ancestry.
Known groups and exact duplicates are controlled, but unknown shared recording
sources cannot be ruled out, including across the original pretrain/labeled
partitions. Do not interpret held-out accuracy as independent-person accuracy.
There is no guarantee of a desired percentage or confidence range.
