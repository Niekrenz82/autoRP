import mlflow
import pandas as pd

from autorp.features import make_features


class RiskPremiumModel(mlflow.pyfunc.PythonModel):
    """Frequency x severity GLM risk premium, packaged as one MLflow model.

    Input: one row per policy with the raw rating factors (DrivAge, BonusMalus, VehAge, VehPower, Area, VehGas).
    Output: expected claims per year, expected cost per claim and risk premium (EUR per policy-year).
    """

    def __init__(self, freq_glm, sev_glm, large_loss_loading):
        self.freq_glm = freq_glm
        self.sev_glm = sev_glm
        self.large_loss_loading = large_loss_loading   # scales capped severity back up to total (uncapped) cost

    def predict(self, context, model_input, params=None):
        X = make_features(model_input)   # glum re-aligns the categories to the levels seen in training
        out = pd.DataFrame(index=model_input.index)
        out["freq"] = self.freq_glm.predict(X)
        out["sev"] = self.sev_glm.predict(X) * self.large_loss_loading
        out["risk_premium"] = out.freq * out.sev
        return out
