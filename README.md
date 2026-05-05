Analyse de l'avantage à domicile dans le football professionnel:

-Objectif:
Estimer un modèle logit pour expliquer la probabilité de victoire à domicile en Premier League et en Ligue 1, avec un focus sur l'impact du huis clos pendant la période Covid.

-Données:
Premier League 2017/2018 → 2021/2022 1 848 matchs
Ligue 1 2017/2018 → 2021/2022 1 799 matchs 

-Source des données:
Premier League : dataset Kaggle (1993-2024)
Ligue 1 : football-data.co.uk (5 fichiers CSV fusionnés)

-Variables:
-home_win: Variable cible — 1 si victoire domicile, 0 sinon
-covid1: si le match se joue à huis clos
-home_form: Moyenne de points sur les 5 derniers matchs (domicile)
-away_form: Moyenne de points sur les 5 derniers matchs (extérieur)
-ranking_diff: Écart de points cumulés au classement (domicile − extérieur)

-Résultats
-Premier League
Taux victoire domicile global: 43.8%
Taux avec public: 45.1%
Taux huis clos (Covid): 40.3% 
Effet Covid: −5.0 pp*

-Ligue 1
Taux victoire domicile global: 43.1%
Taux avec public: 44.7%
Taux huis clos (Covid): 36.8%
Effet Covid: −8.5 pp **

Principal déterminant: Écart au classement***

seuil 5% · ** seuil 1% · *** seuil 0,1% · pp = points de pourcentage

-Structure du projet:
├── data/
│   ├── PremierLeague.csv
│   └── Ligue1.csv
│
├── python/
│   ├── analyse_PL.py
│   ├── analyse_L1.py
│   ├── effets_marginaux_PL.py
│   ├── effets_marginaux_L1.py
│   └── visualisations.py
│
├── r/
│   ├── visualisations_R_PL.R
│   └── visualisations_L1.R
│
├── outputs/
│   ├── python/
│   └── r/
│
└── README.md
Lancer le projet

-Outils:
Python 3.12 — pandas, numpy, statsmodels, matplotlib
R 4.5 — ggplot2, dplyr
