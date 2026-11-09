# Roadmap
- [x] Real data loader, point-in-time features, time split
- [x] Baselines: LogReg, XGBoost (multi-seed)
- [x] PyTorch residual MLP + weighted BCE / focal / oversampling (multi-seed)
- [x] Feature groups ablation script (`04_ablation.py`)
- [x] Ablation: residual blocks vs plain MLP
- [ ] Categorical embeddings for the MLP (card1, addr1, emails, DeviceInfo)
- [ ] GNN (GraphSAGE) on the transaction - card / address / email / device graph
- [ ] MLflow, SHAP, FastAPI, Docker (+ PostgreSQL, Redis), drift monitoring, GenAI assistant

## Known limitations
- No past-label features (ex: past fraud rate of a card): real labels arrive late (chargebacks), so they need a delay.
- The loss functions change the probabilities more than the ranking, so PR-AUC barely moves. Judge them at a fixed threshold.
