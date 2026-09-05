# -*- coding: utf-8 -*-
"""Valeurs publiées, et vérification automatique qu'on les retrouve.

POURQUOI CE FICHIER. Un dépôt de code ne prouve rien tant qu'on ne peut pas confronter ce
qu'il produit à ce qui a été écrit. Chaque script compare donc ses résultats aux valeurs
ci-dessous et signale tout écart, au lieu de laisser au lecteur le soin de vérifier à la main.

Les écarts de dernière décimale sont attendus d'une version de bibliothèque à l'autre : la
tolérance par défaut est de 0,002 sur une AUC. Un écart plus grand est un vrai désaccord, et
doit être traité comme tel.

Sur données synthétiques, la vérification est SAUTÉE — comparer des chiffres simulés à des
chiffres cliniques n'aurait aucun sens.
"""

# Cohorte principale (modèle parcimonieux) et sous-cohorte radiomique.
EFFECTIFS = {
    "n_principal": 86, "evenements_principal": 44,
    "n_radiomique": 79, "evenements_radiomique": 41,
}

MODELE_PARCIMONIEUX = {
    "auc_creat_volume": 0.864,
    "ic_auc_creat_volume": (0.779, 0.935),
    "auc_creat_seule": 0.876,
    "auc_apparente": 0.876,          # 0,8755 — à ne pas confondre avec l'AUC de la créat seule
    "optimisme": 0.009,
    "auc_corrigee": 0.867,
    "pente_calibration": 1.08,
    "citl": -0.16,
    "brier": 0.146,
    "delong_delta": -0.011,
    "delong_p": 0.681,
    "iecv_ancienne": 0.904, "iecv_ancienne_n": 47,
    "iecv_recente": 0.858, "iecv_recente_n": 39,
    "iecv_moyenne": 0.881,
}

RECLASSIFICATION = {
    "nri_continu": 0.602, "nri_continu_ic": (0.186, 0.998), "nri_continu_p": 0.005,
    "nri_categoriel": 0.396, "nri_categoriel_ic": (0.147, 0.635), "nri_categoriel_p": 0.001,
    "nri_categoriel_net": 17,
    "idi": 0.062, "idi_ic": (0.002, 0.119), "idi_p": 0.044,
}

REGLE_TROIS_ZONES = {
    "seuil_bas": 0.31, "seuil_haut": 0.66,
    "bas_n": 25, "bas_risque": 0.16, "bas_sensibilite": 0.91, "bas_specificite": 0.50,
    "bas_vpn": 0.84, "bas_rv_negatif": 0.18,
    "grise_n": 33, "grise_risque": 0.42,
    "haut_n": 28, "haut_risque": 0.93, "haut_sensibilite": 0.59, "haut_specificite": 0.95,
    "haut_vpp": 0.93, "haut_rv_positif": 12.41,
}

# ⚠️ Le M3 de la version imprimée de la Table 2 vaut 0,847 : il provient d'une sélection de
# cinq paramètres par test F calculée sur toute la cohorte AVANT la validation croisée. Le M3
# CONFORME à la méthode déclarée (LASSO refait dans chaque pli, nombre non fixé) vaut 0,810.
# C'est cette valeur qu'il faut citer ; 0,847 n'est reproduit que pour établir la provenance.
MODELES_RADIOMIQUES = {
    "M1": 0.818, "M2": 0.847, "M3_conforme": 0.810, "M3_avec_fuite": 0.847,
    "M3_parametres_par_pli": 12.6,
    "delta_M2_M1": 0.028, "delta_M2_M1_p": 0.43,
    "delta_M3_M2": -0.037, "delta_M3_M2_ic": (-0.084, 0.006), "delta_M3_M2_p": 0.09,
    "delta_M3_M1": -0.008,
}

ROBUSTESSE_SELECTION = {
    "M2 + LASSO (L1)": (0.810, -0.037, 12.6),
    "M2 + elastic net": (0.800, -0.046, 25.3),
    "M2 + sélection univariée (test F)": (0.842, -0.004, 5.0),
    "M2 + information mutuelle": (0.836, -0.011, 5.0),
    "M2 + forêts aléatoires": (0.836, -0.011, 5.0),
    "M2 + paramètres de forme": (0.831, -0.015, 4.8),
    "M2 + paramètres de premier ordre": (0.819, -0.028, 5.3),
    "M2 + paramètres de texture": (0.823, -0.024, 5.0),
}


class Verificateur:
    """Compare les valeurs recalculées aux valeurs publiées et tient le compte des écarts."""

    def __init__(self, synthetique=False, tolerance=0.002):
        self.synthetique = synthetique
        self.tolerance = tolerance
        self.ecarts = []
        self.controles = 0

    def verifier(self, libelle, obtenu, attendu, tolerance=None):
        if self.synthetique or attendu is None or obtenu is None:
            return
        tol = self.tolerance if tolerance is None else tolerance
        self.controles += 1
        if abs(float(obtenu) - float(attendu)) > tol:
            self.ecarts.append((libelle, float(obtenu), float(attendu), tol))

    def bilan(self):
        if self.synthetique:
            print("\nContrôle des valeurs publiées : sauté (données synthétiques).")
            return True
        if not self.controles:
            return True
        if not self.ecarts:
            print(f"\n✓ Contrôle : les {self.controles} valeurs recalculées reproduisent "
                  f"celles du manuscrit.")
            return True
        print(f"\n⚠️  Contrôle : {len(self.ecarts)} écart(s) sur {self.controles} valeurs.")
        for libelle, obtenu, attendu, tol in self.ecarts:
            print(f"    {libelle:<44s} recalculé {obtenu:.3f}   publié {attendu:.3f}   "
                  f"(tolérance {tol})")
        print("    Un écart au-delà de la tolérance n'est pas un arrondi : il demande examen.")
        return False
