import numpy as np
from numpy.typing import NDArray
from typing import Any, Dict, Tuple
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def NCTI_Score(
    X: NDArray[Any], y: NDArray[Any], PCA_components: int = 16
) -> tuple[float, float, float]:
    C = np.unique(y).shape[0]

    n_components = min(PCA_components, X.shape[1])
    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X, y)  # pyright: ignore
    temp = max(np.exp(-pca.explained_variance_[:32].sum()), 1e-10)
    print(pca.explained_variance_[:32].sum() / pca.explained_variance_.sum())

    if temp == 1e-10:
        clf = LinearDiscriminantAnalysis(solver="svd")

    else:
        clf = LinearDiscriminantAnalysis(solver="eigen", shrinkage=float(temp))

    low_feat = clf.fit_transform(X, y)  # pyright: ignore

    low_feat = low_feat - np.mean(low_feat, axis=0, keepdims=True)  # pyright: ignore
    all_lowfeat_nuc = np.linalg.norm(low_feat, ord="nuc")

    low_pred = clf.predict_proba(X)  # pyright: ignore
    sfda_score = (  # pyright: ignore
        np.sum(low_pred[np.arange(X.shape[0]), y]) / X.shape[0]  # pyright: ignore
    )
    # print(clf.score(X, y))

    class_pred_nuc = 0
    class_low_feat = np.zeros((C, 1))  # pyright: ignore
    # print(class_low_feat.shape)
    for c in range(C):
        c_pred = low_pred[(y == c).flatten()]  # pyright: ignore
        c_pred_nuc = np.linalg.norm(c_pred, ord="nuc")  # pyright: ignore
        class_pred_nuc += c_pred_nuc
    # print("all feat nuc: " + str(all_lowfeat_nuc))
    # print("class res nuc: " + str((class_pred_nuc)))
    # print("pred: " + str((sfda_score)))
    return all_lowfeat_nuc, sfda_score, np.log(class_pred_nuc)  # pyright: ignore


def process_NCTI_scores(
    scores_per_model: Dict[str, Tuple[float, float, float]],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Process NCTI scores to return a dictionary with the average NCTI score per model.
    """
    ncti_scores: Dict[str, float] = {}
    all_score: NDArray[Any] = np.zeros(len(scores_per_model))
    cls_score: NDArray[Any] = np.zeros(len(scores_per_model))
    cls_compact: NDArray[Any] = np.zeros(len(scores_per_model))
    for i, scores in enumerate(scores_per_model.values()):
        all_score[i] = scores[0]
        cls_score[i] = scores[1]
        cls_compact[i] = scores[2]

    all_score_min = all_score.min()
    all_score_div = all_score.max() - all_score.min()

    cls_score_min = cls_score.min()
    cls_score_div = cls_score.max() - cls_score.min()

    cls_compact_min = cls_compact.min()
    cls_compact_div = cls_compact.max() - cls_compact_min

    seli_scores: Dict[str, float] = {}
    ncc_scores: Dict[str, float] = {}
    vc_scores: Dict[str, float] = {}

    for model in scores_per_model.keys():
        mascore = (scores_per_model[model][0] - all_score_min) / all_score_div
        mcscore = (scores_per_model[model][1] - cls_score_min) / cls_score_div
        cpscore = (scores_per_model[model][2] - cls_compact_min) / cls_compact_div
        ncti_scores[model] = mcscore + mascore - cpscore
        seli_scores[model] = mascore
        ncc_scores[model] = mcscore
        vc_scores[model] = cpscore

    return ncti_scores, seli_scores, ncc_scores, vc_scores
