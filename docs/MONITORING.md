# Monitoring

```
API (/metrics) --scrape--> Prometheus --> alerts.yml (rules)
                               |
                               +--> Grafana dashboard "FinGuard - system & model monitoring"
```
Start everything with `docker compose up --build`, then `python scripts/09_simulate_traffic.py` to see the dashboard move.
Prometheus: http://localhost:9090 (alerts at /alerts) | Grafana: http://localhost:3000 (admin / admin).

## What is measured
| layer | metrics | why |
|---|---|---|
| System | requests per endpoint, 5xx errors, latency histogram, Redis hit ratio | is the service up and fast? |
| Model | decisions per type, fraud score histogram, flagged share (live vs testing) | is the model behaving like in testing? |
| Data drift | PSI of the scores and of every feature (last 2000 scored transactions vs training), top 10 drifted features | did the inputs change? |
| Quality | live precision / recall from `POST /feedback` (real outcomes, ex: chargebacks) vs the test-set baseline | is the model still right? |
| Assistant | answers from LLM vs template | is the LLM available? |

## Alerts (`monitoring/alerts.yml`)
| alert | rule | severity |
|---|---|---|
| ApiDown | no scrape for 1 min | critical |
| HighErrorRate | > 5% of requests are 5xx for 5 min | critical |
| SlowScoring | p95 of `/score` > 2 s for 5 min | warning |
| ScoreDrift | score PSI > 0.25 for 10 min | warning |
| ManyFeaturesDrifting | more than 40 features with PSI > 0.25 | warning |
| TooManyFlagged / TooFewFlagged | flagged share > 2x or < 0.25x of the test baseline | warning |
| LivePrecisionDrop | live precision < 60% of the baseline (with > 100 known outcomes) | critical |

PSI guide: < 0.10 stable, 0.10 - 0.25 small shift, > 0.25 big shift.

## Also available without Prometheus
- `GET /monitoring/summary`, `GET /monitoring/drift`, `GET /monitoring/performance`
- `python scripts/06_drift_report.py` (offline report over 4 time periods)

## Limits (be honest in interviews)
- Metrics live in the memory of one API process: run ONE uvicorn worker, or switch to Prometheus multiprocess mode.
- The thresholds in `alerts.yml` are starting points. Tune them after you see a week of real traffic.
- Real outcomes (chargebacks) arrive late, so the quality gauges lag behind reality.
- Some history-based features show high PSI even on validation because the data starts "empty" (everybody looks new at the start).
