# Features (IEEE-CIS)

All features use only information from BEFORE the transaction.

| group | features | idea |
|---|---|---|
| Amount | TransactionAmt (log), amt_decimal | big or "too round" amounts |
| Time | hour_sin, hour_cos, dow | time of day / week (TransactionDT is seconds, not a date) |
| C columns | C1-C14 | Vesta counts (cards / addresses / emails linked together) - behave like graph degrees |
| D columns | D1-D15 | days since earlier events |
| M columns | M1-M9 | match flags (T/F) |
| V columns | ~170 of V1-V339 | Vesta engineered; we drop columns that are >95% correlated with one we keep |
| Categories | card1-6, addr1-2, emails, ProductCD, DeviceType, DeviceInfo, id_12-id_38 | replaced by how often the value appears in train (frequency encoding) |
| Identity numbers | id_01-id_11, has_identity | only ~24% of rows have identity data |
| Missing counts | na_count_D, na_count_V, na_count_id, `<col>_na` | missing itself is a signal |
| Client id | card_key = card1 + addr1, uid = card_key + (day - D1) | uid is a better "same person" id |
| History (per card_key, uid) | count_before, amt_vs_mean, secs_since_last | is this behaviour normal for this client? |
| Rolling counts | uid_count_1h / 1d / 7d, card1_count_1d | bursts of activity |

## Preprocessing
Missing numbers stay NaN (XGBoost uses them; LogReg / MLP get 0 = the mean). Long-tail columns -> signed log. Then standardize (train mean/std) and clip to [-5, 5].
