# Vision Module — Model Notes (Week 3)

## Model
ResNet18 (ImageNet-pretrained backbone, frozen), fine-tuned final
classification layer for 6 document categories: form, generic_photo,
id_card, invoice, prescription, receipt.

## Training data
- 96 synthetic images/class (programmatically generated, randomized
  layouts) + 5-17 real images/class (sourced from the web), merged
  into one training set. See `data/generate_sample_images.py`,
  `data/split_real_images.py`, `data/merge_real_into_train.py`.
- 24 synthetic images/class held out as a synthetic validation set.
- 3-5 real images/class held out separately, never merged into
  training, as the honest real-world evaluation set.

## Results

**Synthetic validation set:** 100% accuracy, zero confusion across all
classes. This reflects the synthetic generator's clean, distinct
templates rather than real-world document diversity — treated as a
pipeline sanity check, not a production accuracy estimate.

**Real held-out images (21 images, never seen in training):**
71% overall accuracy (macro F1: 0.62). Breakdown:

| Category       | Precision | Recall | Support |
|-----------------|-----------|--------|---------|
| form            | 0.50      | 0.67   | 3       |
| generic_photo   | 1.00      | 1.00   | 4       |
| id_card         | 1.00      | 1.00   | 5       |
| invoice         | 0.67      | 0.67   | 3       |
| prescription    | 0.40      | 0.67   | 3       |
| receipt         | 0.00      | 0.00   | 3       |

## Known limitation
`receipt` failed on all 3 real held-out examples (misclassified as
`form` or `prescription`), driven by it having the fewest real training
examples of any category (5, after train/eval split). This is a data
scarcity issue, not an architecture issue — the fix is collecting more
diverse real receipt images, not changing the model.

## Before-fix baseline (for reference)
Prior to adding any real images, the model — trained purely on
synthetic data — scored 0/4 (0%) on an initial informal test against
real photos, despite 100% synthetic validation accuracy. This confirmed
a sim-to-real generalization gap and motivated collecting real images
for this iteration.

## Next steps (future work, not blocking Week 3 completion)
- Collect 15-20+ additional real `receipt` images specifically.
- Consider unfreezing the last 1-2 ResNet18 blocks (not just the final
  layer) once more real data is available, for deeper fine-tuning.
- Re-evaluate on a larger, more diverse real held-out set as more real
  documents become available through the live product (Week 7+).