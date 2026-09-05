# -*- coding: utf-8 -*-
"""Ajustement des modèles et estimation honnête de leur performance.

LE PLAN DE VALIDATION, VALABLE POUR TOUS LES MODÈLES DE CE TRAVAIL
    validation croisée stratifiée à cinq plis, répétée vingt fois, graine 42.
    Chaque patient reçoit ainsi vingt prédictions faites par des modèles qui ne l'avaient pas
    vu ; on en prend la moyenne. Répéter réduit la part de hasard du découpage, qui est
    considérable à quatre-vingt-six patients.

CE QUI SE PASSE À L'INTÉRIEUR D'UN PLI, ET POURQUOI
    Tout ce qui APPREND quoi que ce soit des données est ajusté sur le seul pli
    d'apprentissage : le centrage-réduction, le filtre de variance, et surtout la SÉLECTION
    des paramètres radiomiques. Sélectionner avant la validation croisée — même « juste »
    pour choisir cinq paramètres — revient à laisser le modèle voir le devenir des patients
    qui serviront à l'évaluer. C'est la fuite de sélection ; elle gonfle l'AUC d'une manière
    qui ne se retrouve jamais chez un nouveau patient. La fonction `oof_radiomique` refait
    donc la sélection dans chaque pli, et `oof_radiomique_avec_fuite` reproduit la faute pour
    en montrer l'effet chiffré.
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from . import config


def _regression():
    """Régression logistique non pénalisée en pratique (C = 1 sur variables réduites)."""
    return LogisticRegression(C=1.0, max_iter=8000, random_state=config.GRAINE)


def plan_validation():
    return RepeatedStratifiedKFold(n_splits=config.N_PLIS, n_repeats=config.N_REPETITIONS,
                                   random_state=config.GRAINE)


def oof(X, y):
    """Probabilités hors échantillon, moyennées sur les vingt répétitions."""
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    somme, compte = np.zeros(len(y)), np.zeros(len(y))
    for app, val in plan_validation().split(X, y):
        echelle = StandardScaler().fit(X[app])
        modele = _regression().fit(echelle.transform(X[app]), y[app])
        somme[val] += modele.predict_proba(echelle.transform(X[val]))[:, 1]
        compte[val] += 1
    return somme / compte


def ajuster(X, y):
    """Ajuste sur toute la cohorte. Sert au modèle publié et au calcul de l'optimisme."""
    X = np.asarray(X, dtype=float)
    echelle = StandardScaler().fit(X)
    return echelle, _regression().fit(echelle.transform(X), y)


def predire(echelle, modele, X):
    return modele.predict_proba(echelle.transform(np.asarray(X, dtype=float)))[:, 1]


def coefficients(X, y, noms):
    """Coefficients du modèle ajusté sur toute la cohorte, par écart-type de la variable."""
    echelle, modele = ajuster(X, y)
    return dict(zip(noms, modele.coef_[0])), float(modele.intercept_[0])


# ── Sélection des paramètres radiomiques, à l'intérieur d'un pli ──────────────
def selection(strategie, R_app, y_app, k=None):
    """Indices retenus, calculés sur le SEUL pli d'apprentissage (déjà réduit et filtré).

    · lasso / elasticnet : pénalisation choisie par validation croisée interne à cinq plis,
      nombre de paramètres NON fixé à l'avance — c'est la méthode déclarée pour M3 ;
    · univarie / mutuelle / foret : filtres à nombre fixé (k paramètres), pour la table de
      robustesse qui montre que la conclusion ne dépend pas du sélecteur employé.
    """
    k = k or config.K_FILTRE
    if R_app.shape[1] == 0:
        return []
    if strategie == "lasso":
        m = LogisticRegressionCV(Cs=20, cv=5, penalty="l1", solver="saga", scoring="roc_auc",
                                 max_iter=8000, random_state=config.GRAINE).fit(R_app, y_app)
        idx = list(np.where(np.abs(m.coef_.ravel()) > 1e-6)[0])
    elif strategie == "elasticnet":
        m = LogisticRegressionCV(Cs=10, cv=5, penalty="elasticnet", solver="saga",
                                 l1_ratios=[0.5], scoring="roc_auc", max_iter=8000,
                                 random_state=config.GRAINE).fit(R_app, y_app)
        idx = list(np.where(np.abs(m.coef_.ravel()) > 1e-6)[0])
    elif strategie == "univarie":
        idx = list(np.argsort(np.nan_to_num(f_classif(R_app, y_app)[0]))[::-1][:k])
    elif strategie == "mutuelle":
        idx = list(np.argsort(mutual_info_classif(R_app, y_app,
                                                  random_state=config.GRAINE))[::-1][:k])
    elif strategie == "foret":
        foret = RandomForestClassifier(n_estimators=300, random_state=config.GRAINE,
                                       n_jobs=-1).fit(R_app, y_app)
        idx = list(np.argsort(foret.feature_importances_)[::-1][:k])
    else:
        raise ValueError(f"stratégie de sélection inconnue : {strategie}")
    # Si la pénalisation annule tous les coefficients, on retient les k premiers paramètres
    # plutôt que de rendre un modèle vide — le pli doit produire une prédiction.
    return idx if len(idx) else list(range(min(k, R_app.shape[1])))


def oof_radiomique(Xc, Xr, y, strategie):
    """Prédictions hors échantillon d'un modèle clinique + radiomique, sélection INTRA-PLI.

    Rend (probabilités, nombre moyen de paramètres retenus par pli). Le bloc clinique et le
    bloc radiomique sont réduits séparément, chacun sur le pli d'apprentissage.
    """
    Xc, y = np.asarray(Xc, dtype=float), np.asarray(y)
    tailles = []
    somme, compte = np.zeros(len(y)), np.zeros(len(y))
    for app, val in plan_validation().split(Xc, y):
        ech_c = StandardScaler().fit(Xc[app])
        C_app, C_val = ech_c.transform(Xc[app]), ech_c.transform(Xc[val])
        if Xr is None:
            X_app, X_val = C_app, C_val
        else:
            ech_r = StandardScaler().fit(Xr[app])
            R_app, R_val = ech_r.transform(Xr[app]), ech_r.transform(Xr[val])
            filtre = VarianceThreshold(1e-8).fit(R_app)
            R_app, R_val = filtre.transform(R_app), filtre.transform(R_val)
            retenus = selection(strategie, R_app, y[app])
            tailles.append(len(retenus))
            X_app = np.hstack([C_app, R_app[:, retenus]])
            X_val = np.hstack([C_val, R_val[:, retenus]])
        modele = _regression().fit(X_app, y[app])
        somme[val] += modele.predict_proba(X_val)[:, 1]
        compte[val] += 1
    return somme / compte, (float(np.mean(tailles)) if tailles else 0.0)


def oof_radiomique_avec_fuite(Xc, Xr, y, k=5):
    """⚠️ MÉTHODE FAUTIVE, reproduite volontairement — ne pas s'en servir pour conclure.

    Les k paramètres les mieux classés par test F sont choisis sur TOUTE la cohorte, critère
    de jugement compris, AVANT la validation croisée. Le modèle est ensuite validé de façon
    irréprochable — mais le mal est fait : le jeu de paramètres a déjà vu le devenir des
    patients qui servent à l'évaluer. Ce script existe pour chiffrer ce que cette fuite fait
    gagner à l'AUC, et pour que le lecteur puisse le vérifier lui-même.
    """
    Xc, Xr, y = np.asarray(Xc, float), np.asarray(Xr, float), np.asarray(y)
    reduit = StandardScaler().fit_transform(Xr)
    choisis = np.argsort(-np.nan_to_num(f_classif(reduit, y)[0]))[:k]
    X = np.hstack([Xc, reduit[:, choisis]])
    somme, compte = np.zeros(len(y)), np.zeros(len(y))
    for app, val in plan_validation().split(X, y):
        echelle = StandardScaler().fit(X[app])
        modele = _regression().fit(echelle.transform(X[app]), y[app])
        somme[val] += modele.predict_proba(echelle.transform(X[val]))[:, 1]
        compte[val] += 1
    return somme / compte, [int(i) for i in choisis]


# ── Optimisme et transportabilité ────────────────────────────────────────────
def optimisme_bootstrap(X, y, n_boot=None, graine=None):
    """Optimisme de Harrell : de combien la performance apparente flatte le modèle.

    À chaque rééchantillon avec remise, le modèle est RÉAJUSTÉ, puis on retranche à sa
    performance sur son propre rééchantillon celle qu'il obtient sur la cohorte d'origine.
    L'estimateur est bruité : à cinq cents tirages il varie du simple au double selon la
    graine, d'où les deux mille tirages et l'erreur de Monte-Carlo rapportée avec lui.
    """
    n_boot = n_boot or config.N_BOOTSTRAP_OPTIMISME
    rng = np.random.default_rng((config.GRAINE + 1) if graine is None else graine)
    X, y = np.asarray(X, dtype=float), np.asarray(y)
    n, ecarts = len(y), []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(set(y[i])) < 2:
            continue
        echelle, modele = ajuster(X[i], y[i])
        ecarts.append(roc_auc_score(y[i], predire(echelle, modele, X[i]))
                      - roc_auc_score(y, predire(echelle, modele, X)))
    ecarts = np.array(ecarts)
    return dict(optimisme=float(ecarts.mean()),
                erreur_monte_carlo=float(ecarts.std(ddof=1) / np.sqrt(len(ecarts))),
                n=len(ecarts))


def validation_croisee_temporelle(X, y, annees, annee_bascule=2023):
    """Validation croisée externe interne (IECV) dans le temps.

    La cohorte est coupée en deux périodes ; le modèle ajusté sur l'une est évalué sur
    l'autre, et réciproquement. C'est le seul contrôle de transportabilité possible sans
    cohorte extérieure : il répond à « ce modèle tiendrait-il sur des patients recrutés
    autrement ? », au moins dans le temps.

    ⚠️ La coupure est celle de l'ANNÉE déclarée, et non la médiane des patients : les deux ne
    donnent pas les mêmes effectifs ni la même AUC.
    """
    X, y, annees = np.asarray(X, float), np.asarray(y), np.asarray(annees, dtype=float)
    valides = ~np.isnan(annees)
    anciens = np.where(valides & (annees < annee_bascule))[0]
    recents = np.where(valides & (annees >= annee_bascule))[0]
    resultats = []
    for nom, app, val in (("période ancienne", recents, anciens),
                          ("période récente", anciens, recents)):
        if len(set(y[val])) < 2 or len(set(y[app])) < 2:
            continue
        echelle, modele = ajuster(X[app], y[app])
        resultats.append(dict(periode=nom, n=len(val),
                              auc=float(roc_auc_score(y[val], predire(echelle, modele, X[val])))))
    moyenne = float(np.mean([r["auc"] for r in resultats])) if resultats else float("nan")
    return resultats, moyenne


# ── Raccourci partagé par les scripts 03, 04, 05 et 08 ───────────────────────
FICHIER_PREDICTIONS = "03_predictions_hors_echantillon.npy"


def predictions_parcimonieux(df, cache=True):
    """Prédictions hors échantillon du modèle parcimonieux et de la créatininémie seule.

    Rend (y, p_creatininemie, p_modele, df_retenu). Le script 03 dépose ces prédictions dans
    `resultats/` ; les scripts suivants les relisent plutôt que de les recalculer, pour que
    tous les tableaux d'un même rapport portent EXACTEMENT sur les mêmes probabilités. Si le
    fichier est absent ou ne correspond plus à la cohorte chargée, elles sont recalculées.
    """
    from . import donnees  # import tardif : évite un cycle à l'import du paquet

    X, y, retenus = donnees.matrice(df, config.VARIABLES_PARCIMONIEUX)
    X_creat, _, _ = donnees.matrice(retenus, ["creat_admi"])

    chemin = config.DOSSIER_RESULTATS / FICHIER_PREDICTIONS
    if cache and chemin.exists():
        table = np.load(chemin)
        if table.shape == (len(y), 3) and np.array_equal(table[:, 0].astype(int), y):
            return y, table[:, 1], table[:, 2], retenus
    return y, oof(X_creat, y), oof(X, y), retenus
