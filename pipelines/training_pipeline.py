"""ZenML Training Pipeline for FinGuard Fraud Detection.

Orchestrates:
1. prepare_data_step: Ingestion, leak-free feature engineering, time-split.
2. train_model_step: XGBoost early-stopping training on validation set.
3. evaluate_step: Out-of-time test evaluation and threshold calibration.
4. register_model_step: MLflow registry, artifact bundling & production promotion.
"""

from zenml import pipeline
from pipelines.steps.prepare_data_step import prepare_data_step
from pipelines.steps.train_model_step import train_model_step
from pipelines.steps.evaluate_step import evaluate_step
from pipelines.steps.register_model_step import register_model_step


@pipeline(name="finguard_training_pipeline")
def finguard_training_pipeline(
    data_dir: str = "data",
    quick: bool = False,
    n_estimators: int = 5000,
    max_depth: int = 9,
    learning_rate: float = 0.05,
):
    """End-to-end FinGuard ML training and validation pipeline."""
    # 1. Prepare data (or verify prepared arrays)
    prep_dir = prepare_data_step(data_dir=data_dir)
    
    # 2. Train model
    clf, metadata = train_model_step(
        data_dir=prep_dir,
        quick=quick,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    
    # 3. Evaluate and calibrate thresholds
    metrics, thresholds = evaluate_step(
        clf=clf,
        metadata=metadata,
        data_dir=prep_dir,
    )
    
    # 4. Register and promote model
    version = register_model_step(
        clf=clf,
        metadata=metadata,
        metrics=metrics,
        thresholds=thresholds,
        data_dir=prep_dir,
    )
    return version
