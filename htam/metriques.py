# -*- coding: utf-8 -*-
"""Métriques de performance d'un modèle de prédiction du risque.

Toutes prennent en entrée `y` (0/1) et `p` (probabilité prédite), et toutes sont calculées
sur des prédictions HORS ÉCHANTILLON — jamais sur les probabilités que le modèle attribue
aux patients qui ont servi à l'ajuster.

Deux principes tenus partout :
  · un générateur aléatoire par calcul, jamais un générateur partagé. Deux calculs qui se
    passent le même générateur ne sont plus indépendants ni reproductibles séparément ;
  · les comparaisons de deux modèles sont APPARIÉES : les mêmes patients sont rééchantillonnés
    pour les deux modèles comparés, sinon l'intervalle de confiance est trop large.
"""
import numpy as np
from scipy.stats import beta, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

from . import config


# ── Discrimination ────────────────────────────────────────────────────────────
def auc(y, p):
    return float(roc_auc_score(y, p))


def ic_auc(y, p, n_boot=None, graine=None):
    """Intervalle de confiance à 95 % de l'AUC, par percentiles bootstrap."""
    n_boot = n_boot or config.N_BOOTSTRAP
    rng = np.random.RandomState(config.GRAINE if graine is None else graine)
    y, p, n = np.asarray(y), np.asarray(p), len(y)
    valeurs = []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if len(set(y[i])) < 2:
            continue
        valeurs.append(roc_auc_score(y[i], p[i]))
    return tuple(np.percentile(valeurs, [2.5, 97.5]))


def delta_auc(y, p_reference, p_alternatif, n_boot=None, graine=None):
    """Comparaison APPARIÉE de deux AUC : écart observé, IC 95 %, p bilatéral.

    Le p est celui du bootstrap : deux fois la plus petite des deux proportions de
    rééchantillons où l'écart change de signe. Il ne suppose ni normalité ni variance connue.
    """
    n_boot = n_boot or config.N_BOOTSTRAP
    rng = np.random.RandomState(config.GRAINE if graine is None else graine)
    y = np.asarray(y)
    p_reference, p_alternatif = np.asarray(p_reference), np.asarray(p_alternatif)
    n, ecarts = len(y), []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if len(set(y[i])) < 2:
            continue
        ecarts.append(roc_auc_score(y[i], p_alternatif[i]) - roc_auc_score(y[i], p_reference[i]))
    ecarts = np.array(ecarts)
    bas, haut = np.percentile(ecarts, [2.5, 97.5])
    p_valeur = min(2 * min((ecarts <= 0).mean(), (ecarts >= 0).mean()), 1.0)
    return dict(delta=float(roc_auc_score(y, p_alternatif) - roc_auc_score(y, p_reference)),
                bas=float(bas), haut=float(haut), p=float(p_valeur))


# ── DeLong (variance analytique de la différence de deux AUC appariées) ───────
def _rangs_moyens(x):
    ordre = np.argsort(x)
    trie = x[ordre]
    n = len(x)
    rangs = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and trie[j] == trie[i]:
            j += 1
        rangs[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    sortie = np.empty(n)
    sortie[ordre] = rangs
    return sortie


def delong(y, predictions):
    """AUC et matrice de covariance de plusieurs modèles évalués sur les mêmes patients.

    `predictions` : tableau (nombre de modèles × nombre de patients).
    """
    y = np.asarray(y)
    predictions = np.atleast_2d(predictions)
    positifs, negatifs = predictions[:, y == 1], predictions[:, y == 0]
    m, n, k = positifs.shape[1], negatifs.shape[1], predictions.shape[0]
    tx = np.array([_rangs_moyens(positifs[r]) for r in range(k)])
    ty = np.array([_rangs_moyens(negatifs[r]) for r in range(k)])
    tz = np.array([_rangs_moyens(np.concatenate([positifs[r], negatifs[r]])) for r in range(k)])
    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(covariance)


def test_delong(y, p_reference, p_alternatif):
    """Test de DeLong pour deux AUC appariées : ΔAUC, z et p bilatéral."""
    aucs, S = delong(y, np.vstack([p_alternatif, p_reference]))
    contraste = np.array([1, -1])
    variance = float(contraste @ S @ contraste)
    z = (aucs[0] - aucs[1]) / np.sqrt(variance) if variance > 0 else 0.0
    return dict(delta=float(aucs[0] - aucs[1]), z=float(z),
                p=float(2 * (1 - norm.cdf(abs(z)))))


# ── Calibration ───────────────────────────────────────────────────────────────
def pente_calibration(y, p):
    """Régression logistique de y sur le logit des probabilités prédites.

    1 = calibration idéale ; en deçà, le modèle est trop confiant (prédictions trop étalées).
    """
    e = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    logit = np.log(e / (1 - e)).reshape(-1, 1)
    return float(LogisticRegression(C=1e9, max_iter=10000).fit(logit, y).coef_[0][0])


def calibration_dans_le_grand(y, p):
    """Ordonnée à l'origine de calibration (« calibration in the large »).

    Définition : ordonnée d'une régression logistique du devenir sur le logit des probabilités
    prédites, la pente étant FIXÉE à 1 (le logit entre en décalage). Elle vaut 0 quand le risque
    moyen prédit égale le risque moyen observé ; négative, le modèle surestime le risque.

    Ne pas la confondre avec la différence « logit de la prévalence − moyenne des logits »,
    qui n'est pas la même quantité : la moyenne des logits n'est pas le logit de la moyenne.
    Sur ce modèle, l'une vaut 0,00 et l'autre −0,16 pour un rapport observé/attendu de 1,00.

    Rend (ordonnée, erreur-type). Newton-Raphson sur un seul paramètre.
    """
    e = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    logit = np.log(e / (1 - e))
    y = np.asarray(y, dtype=float)
    a = 0.0
    for _ in range(100):
        q = 1.0 / (1.0 + np.exp(-(logit + a)))
        pas = float((y - q).sum()) / float((q * (1 - q)).sum())
        a += pas
        if abs(pas) < 1e-12:
            break
    q = 1.0 / (1.0 + np.exp(-(logit + a)))
    return float(a), float(1.0 / np.sqrt((q * (1 - q)).sum()))


def observe_sur_attendu(y, p):
    """Rapport du nombre d'événements observés au nombre attendu (somme des probabilités)."""
    return float(np.asarray(y, dtype=float).sum() / np.asarray(p, dtype=float).sum())


def brier(y, p):
    """Score de Brier : erreur quadratique moyenne de la probabilité prédite."""
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


# ── Seuils, tables de contingence ─────────────────────────────────────────────
def clopper_pearson(k, n):
    """Intervalle binomial exact — la règle pour une proportion, pas l'approximation normale."""
    if n == 0:
        return (float("nan"), float("nan"))
    bas = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    haut = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return bas, haut


def contingence(y, p, seuil):
    """Table 2 × 2 au seuil demandé et tout ce qui s'en déduit."""
    y, p = np.asarray(y), np.asarray(p)
    predit = p >= seuil
    vp = int((predit & (y == 1)).sum()); fp = int((predit & (y == 0)).sum())
    fn = int((~predit & (y == 1)).sum()); vn = int((~predit & (y == 0)).sum())
    n = vp + fp + fn + vn
    se = vp / (vp + fn) if vp + fn else float("nan")
    sp = vn / (vn + fp) if vn + fp else float("nan")
    return dict(
        seuil=float(seuil), vp=vp, fp=fp, fn=fn, vn=vn, n=n,
        sensibilite=se, ic_sensibilite=clopper_pearson(vp, vp + fn),
        specificite=sp, ic_specificite=clopper_pearson(vn, vn + fp),
        vpp=vp / (vp + fp) if vp + fp else float("nan"),
        ic_vpp=clopper_pearson(vp, vp + fp),
        vpn=vn / (vn + fn) if vn + fn else float("nan"),
        ic_vpn=clopper_pearson(vn, vn + fn),
        exactitude=(vp + vn) / n, ic_exactitude=clopper_pearson(vp + vn, n),
        # Les rapports de vraisemblance ne dépendent PAS de la prévalence : eux se
        # transportent d'une population à l'autre, contrairement aux valeurs prédictives.
        rv_positif=se / (1 - sp) if sp < 1 else float("inf"),
        rv_negatif=(1 - se) / sp if sp > 0 else float("inf"),
    )


def au_seuil(y, p, seuil=0.5):
    """Sensibilité, spécificité et exactitude à un seuil fixe."""
    t = contingence(y, p, seuil)
    return t["sensibilite"], t["specificite"], t["exactitude"]


def seuil_youden(y, p):
    """Seuil qui maximise sensibilité + spécificité − 1."""
    fpr, tpr, seuils = roc_curve(y, p)
    return float(seuils[int(np.argmax(tpr - fpr))])


def benefice_net(y, p, seuil):
    """Bénéfice net d'une stratégie « traiter si p ≥ seuil » (courbe de décision, Vickers).

    Le seuil est aussi le taux d'échange déclaré entre un faux positif et un vrai positif :
    intervenir à partir de 0,60 revient à dire qu'on accepte au plus 1,5 faux positif pour
    un vrai positif évité.
    """
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    predit = p >= seuil
    vp = int((predit & (y == 1)).sum()); fp = int((predit & (y == 0)).sum())
    return vp / n - fp / n * (seuil / (1 - seuil))


def benefice_net_tous(y, seuil):
    """Bénéfice net de la stratégie « traiter tout le monde », terme de comparaison obligé."""
    prevalence = float(np.mean(y))
    return prevalence - (1 - prevalence) * (seuil / (1 - seuil))


# ── Reclassification ──────────────────────────────────────────────────────────
def nri_continu(y, p1, p2):
    """NRI continu : proportion nette de patients dont le risque bouge dans le bon sens."""
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    ev, ne = y == 1, y == 0
    evenements = np.mean(p2[ev] > p1[ev]) - np.mean(p2[ev] < p1[ev])
    non_evenements = np.mean(p2[ne] < p1[ne]) - np.mean(p2[ne] > p1[ne])
    return evenements + non_evenements, evenements, non_evenements


def nri_categoriel(y, p1, p2, seuils=None):
    """NRI catégoriel : mêmes mouvements, mais comptés entre catégories de risque."""
    seuils = seuils or config.SEUILS_NRI
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    c1, c2 = np.digitize(p1, seuils), np.digitize(p2, seuils)
    ev, ne = y == 1, y == 0
    evenements = np.mean(c2[ev] > c1[ev]) - np.mean(c2[ev] < c1[ev])
    non_evenements = np.mean(c2[ne] < c1[ne]) - np.mean(c2[ne] > c1[ne])
    net_ev = int(np.sum(c2[ev] > c1[ev]) - np.sum(c2[ev] < c1[ev]))
    net_ne = int(np.sum(c2[ne] < c1[ne]) - np.sum(c2[ne] > c1[ne]))
    return evenements + non_evenements, evenements, non_evenements, net_ev, net_ne


def idi(y, p1, p2):
    """Amélioration de la discrimination intégrée : écart des risques moyens prédits."""
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    ev, ne = y == 1, y == 0
    return float((p2[ev].mean() - p1[ev].mean()) - (p2[ne].mean() - p1[ne].mean()))


def ic_bootstrap(y, p1, p2, fonction, n_boot=None, graine=None):
    """IC 95 % et p bilatéral d'un indice de reclassification, par bootstrap."""
    n_boot = n_boot or config.N_BOOTSTRAP
    rng = np.random.RandomState(config.GRAINE if graine is None else graine)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    n, valeurs = len(y), []
    for _ in range(n_boot):
        i = rng.randint(0, n, n)
        if len(set(y[i])) < 2:
            continue
        v = fonction(y[i], p1[i], p2[i])
        valeurs.append(v[0] if isinstance(v, tuple) else v)
    valeurs = np.array(valeurs)
    part_negative = float((valeurs <= 0).mean())
    return (tuple(np.percentile(valeurs, [2.5, 97.5])),
            float(2 * min(part_negative, 1 - part_negative)))


# ── Mise en forme française ───────────────────────────────────────────────────
def fr(x, decimales=3):
    """0.847 → « 0,847 ». Le manuscrit est en français, ses chiffres aussi."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = round(float(x), decimales) + 0.0   # efface le zéro négatif : −0,001 s'écrit « 0,00 »
    return f"{x:.{decimales}f}".replace(".", ",").replace("-", "−")


def fr_signe(x, decimales=3):
    """+0,028 / −0,037 — le signe explicite, pour tout ce qui est un ÉCART."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    x = round(float(x), decimales) + 0.0
    return ("+" if x >= 0 else "−") + f"{abs(x):.{decimales}f}".replace(".", ",")


def fr_ic(bornes, decimales=2, signe=False):
    mise_en_forme = fr_signe if signe else fr
    return f"[{mise_en_forme(bornes[0], decimales)} ; {mise_en_forme(bornes[1], decimales)}]"
