# Original LIAR preparation results

Prepared on 2026-09-24 from the [author's original archive](https://sites.cs.ucsb.edu/~william/data/liar_dataset.zip).

Archive SHA-256: `611c1addad919743dde15822b87a60bfb760d8f85597f25289e34621800654c7`.

| Split | Original records | Deduplicated records |
| --- | ---: | ---: |
| Train | 10,269 | 10,243 |
| Validation | 1,284 | 1,284 |
| Test | 1,283 | 1,283 |
| Total | 12,836 | 12,810 |

All original rows have 14 columns, known labels, nonempty IDs and statements,
and globally unique IDs. There are 24 normalized duplicate statement groups,
9 of which cross split boundaries and 6 of which have conflicting labels.
The deterministic text-only policy removes 26 training records. No validation
or test records are removed in this release. No normalized duplicate statements
remain within or across the deduplicated splits.

Conflicting labels are not adjudicated. The retained row supplies its original
label. These cases can depend on missing context; preparation does not establish
that either label is factually correct. Near-duplicates and speaker overlap remain
possible. The raw source and original split variant are preserved unchanged.

| Label ID | Label | Prepared train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| 0 | false | 1,984 | 263 | 250 |
| 1 | half-true | 2,117 | 248 | 267 |
| 2 | mostly-true | 1,964 | 251 | 249 |
| 3 | true | 1,680 | 169 | 211 |
| 4 | barely-true | 1,656 | 237 | 214 |
| 5 | pants-fire | 842 | 116 | 92 |

Training statements have a median of 17 whitespace-separated words, a 95th
percentile of 33, and a maximum of 66. These are not transformer token counts.
The smallest training class is pants-fire; use macro-F1 and per-class metrics.

The machine-readable audit at `data/liar/report.json` lists source/output
checksums and all duplicate memberships and exclusions. Regenerate it using
`python -m training.prepare_dataset`. No model training has occurred.

This dataset supports a six-class claim-text baseline, not the application's
three-way evidence verifier. It is an older political-claim benchmark and is
not representative validation of arbitrary contemporary news. The original
README restricts use to research purposes; dataset files are ignored by Git.
