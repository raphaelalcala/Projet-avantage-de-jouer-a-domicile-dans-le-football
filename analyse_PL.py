import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# 1. CHARGEMENT DES DONNÉES

df = pd.read_csv("PremierLeague.csv")

df = df[df["Season"].isin(["2017-2018", "2018-2019", "2019-2020", "2020-2021", "2021-2022"])].copy()
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["Season", "MatchWeek", "Date"]).reset_index(drop=True)

print(f"Nombre de matchs : {len(df)}")

# 2. CRÉATION DES VARIABLES

# Variable cible : 1 si l'équipe à domicile gagne, 0 sinon
df["home_win"] = (df["FullTimeResult"] == "H").astype(int)

# Variable Covid : 1 si le match se joue à huis clos
df["covid"] = 0
df.loc[(df["Season"] == "2019-2020") & (df["Date"] >= "2020-03-01"), "covid"] = 1
df.loc[df["Season"] == "2020-2021", "covid"] = 1

# Écart au classement : points cumulés domicile - points cumulés extérieur
# On crée d'abord un tableau avec les points de chaque équipe match par match
home = df[["Season", "Date", "HomeTeam", "HomeTeamPoints"]].rename(
    columns={"HomeTeam": "team", "HomeTeamPoints": "pts"})
away = df[["Season", "Date", "AwayTeam", "AwayTeamPoints"]].rename(
    columns={"AwayTeam": "team", "AwayTeamPoints": "pts"})

all_pts = pd.concat([home, away]).sort_values(["Season", "Date"])

# On cumule les points au fil de la saison, AVANT chaque match (shift)
all_pts["cumpts"] = all_pts.groupby(["Season", "team"])["pts"].cumsum()
all_pts["cumpts_avant"] = all_pts.groupby(["Season", "team"])["cumpts"].shift(1).fillna(0)
all_pts = all_pts.drop_duplicates(["Date", "team"])

# On recolle les points cumulés à notre tableau principal
df = df.merge(
    all_pts[["Date", "team", "cumpts_avant"]].rename(columns={"team": "HomeTeam", "cumpts_avant": "pts_domicile"}),
    on=["Date", "HomeTeam"], how="left"
)
df = df.merge(
    all_pts[["Date", "team", "cumpts_avant"]].rename(columns={"team": "AwayTeam", "cumpts_avant": "pts_exterieur"}),
    on=["Date", "AwayTeam"], how="left"
)
df["ranking_diff"] = df["pts_domicile"] - df["pts_exterieur"]

# Forme récente : moyenne de points sur les 5 derniers matchs
# On crée un tableau avec le résultat de chaque équipe à chaque match
home_res = df[["Date", "Season", "HomeTeam", "FullTimeResult"]].rename(columns={"HomeTeam": "team"})
home_res["pts"] = home_res["FullTimeResult"].map({"H": 3, "D": 1, "A": 0})

away_res = df[["Date", "Season", "AwayTeam", "FullTimeResult"]].rename(columns={"AwayTeam": "team"})
away_res["pts"] = away_res["FullTimeResult"].map({"H": 0, "D": 1, "A": 3})

tous_matchs = pd.concat([home_res, away_res]).sort_values("Date")

# Moyenne glissante sur 5 matchs, décalée d'un match (on exclut le match courant)
tous_matchs["forme"] = tous_matchs.groupby(["Season", "team"])["pts"].transform(
    lambda x: x.shift(1).rolling(5, min_periods=1).mean()
)
tous_matchs = tous_matchs.drop_duplicates(["Date", "team"])

# On recolle la forme à notre tableau principal
df = df.merge(
    tous_matchs[["Date", "team", "forme"]].rename(columns={"team": "HomeTeam", "forme": "home_form"}),
    on=["Date", "HomeTeam"], how="left"
)
df = df.merge(
    tous_matchs[["Date", "team", "forme"]].rename(columns={"team": "AwayTeam", "forme": "away_form"}),
    on=["Date", "AwayTeam"], how="left"
)

# 3. DATASET FINAL

variables = ["covid", "home_form", "away_form", "ranking_diff"]
df_model = df[variables + ["home_win", "Season"]].dropna().reset_index(drop=True)

print(f"Matchs dans le modèle : {len(df_model)}")
print(f"Taux victoire domicile : {df_model['home_win'].mean():.1%}")
print(f"Taux victoire domicile sans Covid : {df_model[df_model['covid']==0]['home_win'].mean():.1%}")
print(f"Taux victoire domicile avec Covid : {df_model[df_model['covid']==1]['home_win'].mean():.1%}")

# Sauvegarde (utilisé ensuite sous R)
df_model.to_csv("matches_model_PL.csv", index=False)
print("\nFichier sauvegardé : matches_model_PL.csv")

# 4. MODÈLE LOGIT

X = sm.add_constant(df_model[variables])
y = df_model["home_win"]

modele = sm.Logit(y, X).fit()
print(modele.summary())
