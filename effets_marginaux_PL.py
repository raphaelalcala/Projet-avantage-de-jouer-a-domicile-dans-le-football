import pandas as pd
import statsmodels.api as sm

# Recharger les données et réestimer le modèle
df_model = pd.read_csv("matches_model_PL.csv")

variables = ["covid", "home_form", "away_form", "ranking_diff"]
X = sm.add_constant(df_model[variables])
y = df_model["home_win"]

modele = sm.Logit(y, X).fit(disp=0)

print(modele.summary())
print("\n--- EFFETS MARGINAUX ---")
print(modele.get_margeff().summary())