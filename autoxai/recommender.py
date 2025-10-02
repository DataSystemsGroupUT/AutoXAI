import numpy as np
from .metrics import (
    fidelity, stability, sparsity, comprehensibility,
    consistency, infidelity, sensitivity, robustness_noise
)
from .explainers import get_explanations

class AutoXAIRecommender:
    """
    AutoXAI: Meta-learner for recommending global explanation techniques
    """
    def __init__(self, methods=None):
        if methods is None:
            self.methods = ["LIME", "Anchor", "RuleFit", "RuleMatrix"]
        else:
            self.methods = methods
        self.recommendations = None

    def fit(self, model, X, y):
        self.dataset_summary = {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "feature_types": [str(X.dtypes[i]) for i in range(X.shape[1])]
        }
        self.scores = {}
        for method in self.methods:
            explanation = get_explanations(model, X, method)
            self.scores[method] = {
                "M1_fidelity": fidelity(model, X, explanation),
                "M2_stability": stability([list(explanation.values())]),
                "M3_sparsity": sparsity(explanation),
                "M4_comprehensibility": comprehensibility(explanation),
                "M5_consistency": consistency({method: list(explanation.values())}),
                "M6_infidelity": infidelity(model, X, explanation),
                "M7_sensitivity": sensitivity(explanation, X),
                "M8_robustness": robustness_noise(explanation, explanation),
            }
        self.recommendations = sorted(self.scores.items(),
                                      key=lambda x: np.mean(list(x[1].values())),
                                      reverse=True)

    def recommend(self, top_k=1):
        if self.recommendations is None:
            raise ValueError("Model not fitted. Call fit(model, X, y) first.")
        return [m for m, _ in self.recommendations[:top_k]]
