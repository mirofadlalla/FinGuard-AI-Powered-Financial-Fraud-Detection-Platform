# Roadmap
- [x] Real data loader, point-in-time features, time split
- [x] Baselines: LogReg, XGBoost (multi-seed)
- [x] PyTorch residual MLP + weighted BCE / focal / oversampling (multi-seed)
- [x] Feature groups ablation script (`04_ablation.py`)
- [ ] Ablation: residual blocks vs plain MLP
- [ ] Categorical embeddings for the MLP (card1, addr1, emails, DeviceInfo)
- [ ] GNN (GraphSAGE) on the transaction - card / address / email / device graph
- [x] MLflow tracking + registry + promotion gate (`05_train_final.py`)
- [x] SHAP explanations (TreeSHAP) per flagged transaction
- [x] FastAPI + PostgreSQL + Redis + Docker compose
- [x] PSI drift report (`06_drift_report.py`) + `/monitoring/summary`
- [x] Investigation assistant (`/investigate`)
- [ ] Online feature computation (feature store) instead of replay mode
- [ ] Policy RAG for the assistant, shadow deployment, retrain on train+val

## Known limitations
- No past-label features (ex: past fraud rate of a card): real labels arrive late (chargebacks), so they need a delay.
- The loss functions change the probabilities more than the ranking, so PR-AUC barely moves. Judge them at a fixed threshold.
