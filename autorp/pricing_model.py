import mlflow
import pandas as pd

from autorp.features import make_features


class RiskPremiumModel(mlflow.pyfunc.PythonModel):
    """Frequency x severity GLM risk premium, packaged as one MLflow model.

    Dataset-agnostic: the FeatureSpec the GLMs were fitted on is stored with the model, so it can rebuild
    the same features from raw rows of whatever dataset it was trained on.
    Input: one row per policy with the raw columns listed in `feature_spec.features`.
    Output: expected claims per year, expected cost per claim and risk premium per policy-year.
    """

    def __init__(self, freq_glm, sev_glm, large_loss_loading, feature_spec):
        self.freq_glm = freq_glm
        self.sev_glm = sev_glm
        self.large_loss_loading = large_loss_loading   # scales capped severity back up to total (uncapped) cost
        self.feature_spec = feature_spec

    def predict(self, context, model_input, params=None):
        X = make_features(model_input, self.feature_spec)   # glum re-aligns the categories to the levels seen in training
        out = pd.DataFrame(index=model_input.index)
        out["freq"] = self.freq_glm.predict(X)
        out["sev"] = self.sev_glm.predict(X) * self.large_loss_loading
        out["risk_premium"] = out.freq * out.sev
        return out
