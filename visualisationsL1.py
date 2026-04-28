import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

df = pd.read_csv("matches_model_L1.csv")
df_model = pd.read_csv("matches_model_L1.csv")

variables = ["covid", "home_form", "away_form", "ranking_diff"]
X = sm.add_constant(df_model[variables])
y = df_model["home_win"]
modele = sm.Logit(y, X).fit(disp=0)

VERT  = "#1D9E75"   # public présent
ROUGE = "#D85A30"   # huis clos Covid

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Analyse de l'avantage à domicile en Ligue 1 (2017–2022)",
             fontsize=15, fontweight="bold", y=1.01)

# GRAPHIQUE 1 : Taux de victoire domicile par saison
ax1 = axes[0, 0]

taux_par_saison = df_model.groupby("Season")["home_win"].mean().reindex(["2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022"])
etiquettes = ["2017-2018", "2018-2019", "2019-2020\n(saison arrêtée)", "2020-2021\n(huis clos complet)", "2021-2022\n(retour public)"]
couleurs_barres = [VERT, VERT, VERT, ROUGE, VERT]

barres = ax1.bar(etiquettes, taux_par_saison.values * 100, color=couleurs_barres, width=0.5)

# Afficher le pourcentage au dessus de chaque barre
for barre, val in zip(barres, taux_par_saison.values):
    ax1.text(barre.get_x() + barre.get_width() / 2, barre.get_height() + 0.5,
             f"{val*100:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

# Ligne de moyenne
ax1.axhline(taux_par_saison.mean() * 100, color="gray", linestyle="--", linewidth=1.2)
ax1.set_ylim(0, 60)
ax1.set_ylabel("Taux de victoire à domicile (%)")
ax1.set_title("1. Taux de victoire domicile par saison", fontweight="bold")
ax1.legend(handles=[mpatches.Patch(color=VERT, label="Public présent"),
                    mpatches.Patch(color=ROUGE, label="Huis clos")], fontsize=9)

# GRAPHIQUE 2 : Effets marginaux
ax2 = axes[0, 1]

margins     = modele.get_margeff()
noms_vars   = ["Covid\n(huis clos)", "Forme\ndomicile", "Forme\nextérieur", "Écart\nclassement"]
effets      = margins.margeff
ic          = margins.conf_int()
erreur_bas  = effets - ic[:, 0]
erreur_haut = ic[:, 1] - effets
couleurs_effets = [ROUGE if e < 0 else VERT for e in effets]

barres2 = ax2.barh(noms_vars, effets * 100,
                   xerr=[erreur_bas * 100, erreur_haut * 100],
                   color=couleurs_effets, height=0.5,
                   error_kw={"elinewidth": 1.5, "capsize": 5, "ecolor": "gray"})

ax2.axvline(0, color="black", linewidth=1)
ax2.set_xlabel("Variation de la probabilité de victoire (points de %)")
ax2.set_title("2. Effets marginaux des variables", fontweight="bold")

# Afficher la valeur à côté de chaque barre
for val, barre in zip(effets, barres2):
    decalage = 0.3 if val >= 0 else -0.3
    ax2.text(val * 100 + decalage, barre.get_y() + barre.get_height() / 2,
             f"{val*100:+.2f} pp", va="center", fontsize=9, fontweight="bold")
  
# GRAPHIQUE 3 : Courbe logistique (probabilité prédite)

ax3 = axes[1, 0]

ecart_range  = np.linspace(df_model["ranking_diff"].min(), df_model["ranking_diff"].max(), 200)
moy_hf = df_model["home_form"].mean()
moy_af = df_model["away_form"].mean()
coefs  = modele.params

def proba_victoire(ecart, covid_val):
    z = (coefs["const"]
         + coefs["covid"] * covid_val
         + coefs["home_form"] * moy_hf
         + coefs["away_form"] * moy_af
         + coefs["ranking_diff"] * ecart)
    return 1 / (1 + np.exp(-z))

prob_public = [proba_victoire(e, 0) * 100 for e in ecart_range]
prob_covid  = [proba_victoire(e, 1) * 100 for e in ecart_range]

ax3.plot(ecart_range, prob_public, color=VERT,  linewidth=2.5, label="Public présent")
ax3.plot(ecart_range, prob_covid,  color=ROUGE, linewidth=2.5, linestyle="--", label="Huis clos")
ax3.fill_between(ecart_range, prob_public, prob_covid, alpha=0.12, color=ROUGE)
ax3.axhline(50, color="gray", linestyle=":", linewidth=1)
ax3.set_xlabel("Écart de points au classement (domicile − extérieur)")
ax3.set_ylabel("Probabilité de victoire domicile (%)")
ax3.set_title("3. Probabilité prédite selon le niveau des équipes", fontweight="bold")
ax3.legend()
ax3.set_ylim(0, 100)

# GRAPHIQUE 4 : Distribution de la forme domicile
ax4 = axes[1, 1]

forme_victoire = df_model[df_model["home_win"] == 1]["home_form"]
forme_defaite  = df_model[df_model["home_win"] == 0]["home_form"]

ax4.hist(forme_victoire, bins=15, alpha=0.65, color=VERT,  label="Victoire domicile",    density=True)
ax4.hist(forme_defaite,  bins=15, alpha=0.65, color=ROUGE, label="Défaite / nul",         density=True)
ax4.axvline(forme_victoire.mean(), color=VERT,  linestyle="--", linewidth=2,
            label=f"Moy. victoires : {forme_victoire.mean():.2f}")
ax4.axvline(forme_defaite.mean(),  color=ROUGE, linestyle="--", linewidth=2,
            label=f"Moy. défaites : {forme_defaite.mean():.2f}")
ax4.set_xlabel("Forme récente de l'équipe domicile (moy. pts sur 5 matchs)")
ax4.set_ylabel("Densité")
ax4.set_title("4. Forme domicile selon le résultat", fontweight="bold")
ax4.legend(fontsize=9)

plt.tight_layout()
plt.savefig("visualisations_L1.png", dpi=150, bbox_inches="tight")
plt.show()
print("Graphiques sauvegardés : visualisations.png")
