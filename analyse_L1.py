import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# Chaque saison est dans un fichier séparé
fichiers = {
    "2017-2018": "F1 (2).csv",
    "2018-2019": "F1 (3).csv",
    "2019-2020": "F1 (1).csv",  # saison arrêtée en mars (Covid)
    "2020-2021": "F1.csv",       # huis clos complet
    "2021-2022": "F1 (4).csv",
}

dfs = []
for saison, fichier in fichiers.items():
    temp = pd.read_csv(fichier)
    temp["Season"] = saison
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

df = df[["Season", "Date", "HomeTeam", "AwayTeam",
         "FTHG", "FTAG", "FTR"]].copy()

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.sort_values(["Season", "Date"]).reset_index(drop=True)

print(f"Nombre de matchs : {len(df)}")
print(df["Season"].value_counts().sort_index())

# Variable cible : 1 si victoire domicile, 0 sinon
df["home_win"] = (df["FTR"] == "H").astype(int)

# Variable Covid
# Note : en Ligue 1 la saison 2019-2020 a été arrêtée en mars
# sans reprendre → pas de huis clos, saison annulée
# Seule la 2020-2021 est entièrement à huis clos
df["covid"] = 0
df.loc[df["Season"] == "2020-2021", "covid"] = 1

home = df[["Season", "Date", "HomeTeam", "FTHG", "FTAG", "FTR"]].copy()
home["pts"] = home["FTR"].map({"H": 3, "D": 1, "A": 0})
home = home.rename(columns={"HomeTeam": "team"})

away = df[["Season", "Date", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
away["pts"] = away["FTR"].map({"H": 0, "D": 1, "A": 3})
away = away.rename(columns={"AwayTeam": "team"})

all_pts = pd.concat([home[["Season", "Date", "team", "pts"]],
                     away[["Season", "Date", "team", "pts"]]]).sort_values(["Season", "Date"])

all_pts["cumpts"] = all_pts.groupby(["Season", "team"])["pts"].cumsum()
all_pts["cumpts_avant"] = all_pts.groupby(["Season", "team"])["cumpts"].shift(1).fillna(0)
all_pts = all_pts.drop_duplicates(["Date", "team"])

df = df.merge(
    all_pts[["Date", "team", "cumpts_avant"]].rename(
        columns={"team": "HomeTeam", "cumpts_avant": "pts_domicile"}),
    on=["Date", "HomeTeam"], how="left"
)
df = df.merge(
    all_pts[["Date", "team", "cumpts_avant"]].rename(
        columns={"team": "AwayTeam", "cumpts_avant": "pts_exterieur"}),
    on=["Date", "AwayTeam"], how="left"
)
df["ranking_diff"] = df["pts_domicile"] - df["pts_exterieur"]

# Forme récente : moyenne de points sur les 5 derniers matchs
home_res = df[["Date", "Season", "HomeTeam", "FTR"]].rename(columns={"HomeTeam": "team"})
home_res["pts"] = home_res["FTR"].map({"H": 3, "D": 1, "A": 0})

away_res = df[["Date", "Season", "AwayTeam", "FTR"]].rename(columns={"AwayTeam": "team"})
away_res["pts"] = away_res["FTR"].map({"H": 0, "D": 1, "A": 3})

tous_matchs = pd.concat([home_res, away_res]).sort_values("Date")

tous_matchs["forme"] = tous_matchs.groupby(["Season", "team"])["pts"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
tous_matchs = tous_matchs.drop_duplicates(["Date", "team"])

df = df.merge(
    tous_matchs[["Date", "team", "forme"]].rename(
        columns={"team": "HomeTeam", "forme": "home_form"}),
    on=["Date", "HomeTeam"], how="left"
)
df = df.merge(
    tous_matchs[["Date", "team", "forme"]].rename(
        columns={"team": "AwayTeam", "forme": "away_form"}),
    on=["Date", "AwayTeam"], how="left"
)

variables = ["covid", "home_form", "away_form", "ranking_diff"]
df_model = df[variables + ["home_win", "Season"]].dropna().reset_index(drop=True)

print(f"\nMatchs dans le modèle : {len(df_model)}")
print(f"Taux victoire domicile : {df_model['home_win'].mean():.1%}")
print(f"Taux victoire domicile sans Covid : {df_model[df_model['covid']==0]['home_win'].mean():.1%}")
print(f"Taux victoire domicile avec Covid : {df_model[df_model['covid']==1]['home_win'].mean():.1%}")

df_model.to_csv("matches_model_L1.csv", index=False)
print("\nFichier sauvegardé : matches_model_L1.csv")


X = sm.add_constant(df_model[variables])
y = df_model["home_win"]

modele = sm.Logit(y, X).fit()
print(modele.summary())

print("\n--- EFFETS MARGINAUX ---")
print(modele.get_margeff().summary())


Nombre de matchs : 1799
2017-2018    380
2018-2019    380
2019-2020    279
2020-2021    380
2021-2022    380

Matchs dans le modèle : 1746
Taux victoire domicile : 43.1%
Taux victoire domicile sans Covid : 44.7%
Taux victoire domicile avec Covid : 36.8%

Fichier sauvegardé : matches_model_L1.csv

                           Logit Regression Results
==============================================================================
Dep. Variable:               home_win   No. Observations:                 1746
Pseudo R-squ.:                 0.06391   Log-Likelihood:                -1117.1
LLR p-value:                 5.795e-32
================================================================================
                   coef    std err          z      P>|z|
--------------------------------------------------------------------------------
const           -0.2880      0.147     -1.965      0.049
covid           -0.3791      0.127     -2.994      0.003
home_form        0.1277      0.082      1.561      0.119
away_form       -0.0701      0.079     -0.883      0.378
ranking_diff     0.0436      0.005      8.057      0.000
