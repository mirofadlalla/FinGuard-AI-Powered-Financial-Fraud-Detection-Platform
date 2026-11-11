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
| History (per card_key, uid) | amt_vs_mean, secs_since_last | is this behaviour normal for this client? (the all-time `count_before` was removed: it only measures how long the data has been collected, and the ablation showed no gain) |
| Rolling counts | uid_count_1h / 1d / 7d, card1_count_1d | bursts of activity |

## Ideas from the top Kaggle solutions (all past-only here)
| group | features |
|---|---|
| D normalized | Dn_D2, D4, D10, D11, D15 = D - day (almost constant per client) |
| Client aggregates | mean/std over the client's past rows: M9, M5, Dn_D2, Dn_D15, C13, C9, C1, C11, amount (uid); M4, M1, M7, M8 (card+addr) |
| Combo counts | transactions in the last 1d / 7d for card1+addr1+email, card3+addr1+email, card4+addr1+email |
| V blocks | one "is missing" flag per block of V columns; V columns reduced by (same missing count -> correlation > 0.75 -> keep one) |

## Graph for the GNN (`neighbors` in `data/ieee_ready.npz`)
| relation | two transactions are connected when they share |
|---|---|
| uid | the same client id (card + address + start day) |
| card_key | the same card + billing address |
| DeviceInfo | the same device (only devices with <= 1000 transactions) |
| email_addr | the same email domain + address |
Each transaction keeps the last 10 EARLIER neighbours per relation (row numbers, -1 = none).

## Preprocessing
Missing numbers stay NaN (XGBoost uses them; LogReg / MLP get 0 = the mean). Long-tail columns -> signed log. Then standardize (train mean/std) and clip to [-5, 5].
