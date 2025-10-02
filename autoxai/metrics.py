import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from scipy.spatial.distance import cosine

def fidelity(model, X, explanation, y_true=None):
    y_pred_model = model.predict(X)
    y_pred_expl = explanation_predict(explanation, X)
    return accuracy_score(y_pred_model, y_pred_expl)

def stability(explanations_list):
    sims = []
    for i in range(len(explanations_list)):
        for j in range(i + 1, len(explanations_list)):
            sims.append(1 - cosine(explanations_list[i], explanations_list[j]))
    return np.mean(sims) if sims else 1.0

def sparsity(explanation):
    importances = np.array(list(explanation.values()))
    return np.sum(importances == 0) / len(importances)

def comprehensibility(explanation):
    if isinstance(explanation, list):
        return 1.0 / len(explanation) if len(explanation) > 0 else 0.0
    return 1.0

def consistency(explanations_dict):
    methods = list(explanations_dict.keys())
    sims = []
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            sims.append(1 - cosine(explanations_dict[methods[i]], explanations_dict[methods[j]]))
    return np.mean(sims) if sims else 1.0

def infidelity(model, X, explanation):
    y_pred_model = model.predict(X)
    y_pred_expl = explanation_predict(explanation, X)
    return mean_squared_error(y_pred_model, y_pred_expl)

def sensitivity(explanation, X, perturbation_strength=0.01):
    original = np.array(list(explanation.values()))
    perturbed = []
    for _ in range(5):
        Xp = X.copy()
        noise = np.random.normal(0, perturbation_strength, size=Xp.shape)
        Xp = Xp + noise
        pert_exp = list(explanation.values())  # placeholder
        perturbed.append(np.array(pert_exp))
    diffs = [np.linalg.norm(original - p) for p in perturbed]
    return 1.0 / (1.0 + np.mean(diffs))

def robustness_noise(explanation_before, explanation_after):
    return 1 - cosine(list(explanation_before.values()), list(explanation_after.values()))

def explanation_predict(explanation, X):
    return np.random.randint(0, 2, size=len(X))
