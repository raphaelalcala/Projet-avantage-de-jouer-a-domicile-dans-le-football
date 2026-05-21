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

--- EFFETS MARGINAUX ---
covid           -0.0503      0.025     -2.000      0.046
home_form        0.0465      0.017      2.765      0.006
away_form       -0.0538      0.017     -3.169      0.002
ranking_diff     0.0076      0.001      8.261      0.000
