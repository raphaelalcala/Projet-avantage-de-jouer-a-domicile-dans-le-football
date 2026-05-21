import pandas as pd
import statsmodels.api as sm

df_model = pd.read_csv("matches_model_L1.csv")

variables = ["covid", "home_form", "away_form", "ranking_diff"]
X = sm.add_constant(df_model[variables])
y = df_model["home_win"]

modele = sm.Logit(y, X).fit(disp=0)

print(modele.summary())
print("\n--- EFFETS MARGINAUX ---")
print(modele.get_margeff().summary())

with open("resultats_L1.txt", "w") as f:
    f.write(modele.summary().as_text())
    f.write("\n\n--- EFFETS MARGINAUX ---\n")
    f.write(modele.get_margeff().summary().as_text())

print("\nRésultats sauvegardés : resultats_L1.txt")

--- EFFETS MARGINAUX ---
covid           -0.0852      0.028     -3.022      0.003
home_form        0.0287      0.018      1.565      0.117
away_form       -0.0158      0.018     -0.883      0.377
ranking_diff     0.0098      0.001      8.648      0.000
