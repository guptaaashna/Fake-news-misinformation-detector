# DistilBERT tokenization results

Prepared on 2026-09-25 using the saved `distilbert/distilbert-base-uncased`
tokenizer from revision `12040accade4e8a0f71eabdb258fecc2e7e948be`.
Inputs are the deduplicated LIAR splits. Text alone is tokenized; metadata is not
included in model inputs. The tokenizer is pretrained, not fitted on these splits.

| Split | Claims | Median tokens | P95 | P99 | Maximum | Truncated at 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 10,243 | 23 | 42 | 52 | 98 | 0 |
| Validation | 1,284 | 23 | 42 | 52 | 77 | 0 |
| Test | 1,283 | 23 | 41 | 52 | 68 | 0 |

Lengths include the tokenizer's special tokens. Percentiles use nearest rank.
**128 is sufficient for every current claim**; no input tokens are discarded.
The limit was fixed in the existing model setup and confirmed using train lengths.
Held-out length summaries are diagnostic only; no test predictions or performance
metrics are computed. Future longer inputs may be truncated and require a separate
input-length policy when integrating the model into the app.

Tokenized files contain variable-length token IDs and attention masks, with labels
unchanged. Padding is delayed until batch creation: a batch with maximum length 42
uses 42 columns, rather than 128. Padding positions receive attention mask 0.

Default training batch size is 16, shuffled with seed 42; validation/test batch size
is 32 with sequential ordering. Final partial batches are retained. This yields
641 training batches, 41 validation batches, and 41 test batches per pass. Test
loading remains explicit and is reserved for final evaluation. Batch sizes can be
reduced if memory is constrained; optimizer/backward memory has not been benchmarked.

Reproduce with `python -m training.tokenize_dataset` after dataset and model setup.
The generated `data/liar/tokenized/report.json` records exact input/output and
tokenizer checksums plus package versions. Dataset and tokenized files stay out of
Git. No optimizer steps or fine-tuning have been performed.
