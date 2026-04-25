library(ggplot2)
library(dplyr)

library(ggplot2)
library(dplyr)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================

df <- read.csv("matches_model_PL.csv")

# Couleurs
VERT  <- "#1D9E75"
ROUGE <- "#D85A30"

# ============================================================
# GRAPHIQUE 1 : Taux de victoire domicile par saison
# ============================================================

taux_saison <- df %>%
  group_by(Season) %>%
  summarise(taux = mean(home_win) * 100)

taux_saison$couleur <- ifelse(taux_saison$Season == "2020-2021", ROUGE, VERT)

g1 <- ggplot(taux_saison, aes(x = Season, y = taux, fill = couleur)) +
  geom_bar(stat = "identity", width = 0.5) +
  geom_text(aes(label = paste0(round(taux, 1), "%")),
            vjust = -0.5, fontface = "bold", size = 4) +
  geom_hline(yintercept = mean(taux_saison$taux),
             linetype = "dashed", color = "gray50") +
  scale_fill_identity() +
  scale_x_discrete(labels = c("2017-2018", "2018-2019",
                              "2019-2020\n(public → huis clos)",
                              "2020-2021\n(huis clos complet)",
                              "2021-2022\n(retour public)")) +
  ylim(0, 60) +
  labs(title = "1. Taux de victoire domicile par saison",
       x = "", y = "Taux de victoire (%)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# ============================================================
# GRAPHIQUE 2 : Courbe logistique (probabilité prédite)
# ============================================================

# Réestimer le modèle logit
modele <- glm(home_win ~ covid + home_form + away_form + ranking_diff,
              data = df, family = binomial)

# Créer une grille de valeurs pour ranking_diff
grille <- data.frame(
  ranking_diff = seq(min(df$ranking_diff), max(df$ranking_diff), length.out = 200),
  home_form    = mean(df$home_form),
  away_form    = mean(df$away_form)
)

# Probabilités prédites avec et sans Covid
grille_public <- grille %>% mutate(covid = 0)
grille_covid  <- grille %>% mutate(covid = 1)

grille_public$prob <- predict(modele, grille_public, type = "response") * 100
grille_covid$prob  <- predict(modele, grille_covid,  type = "response") * 100

grille_public$condition <- "Public présent"
grille_covid$condition  <- "Huis clos (Covid)"

courbes <- rbind(grille_public, grille_covid)

g2 <- ggplot(courbes, aes(x = ranking_diff, y = prob,
                          color = condition, linetype = condition)) +
  geom_line(size = 1.2) +
  geom_hline(yintercept = 50, linetype = "dotted", color = "gray50") +
  scale_color_manual(values = c("Public présent" = VERT,
                                "Huis clos (Covid)" = ROUGE)) +
  scale_linetype_manual(values = c("Public présent" = "solid",
                                   "Huis clos (Covid)" = "dashed")) +
  ylim(0, 100) +
  labs(title = "2. Probabilité prédite selon l'écart au classement",
       x = "Écart de points au classement (domicile − extérieur)",
       y = "Probabilité de victoire domicile (%)",
       color = "", linetype = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

# ============================================================
# GRAPHIQUE 3 : Distribution de la forme domicile
# ============================================================

df$resultat <- ifelse(df$home_win == 1, "Victoire domicile", "Défaite / nul")

g3 <- ggplot(df, aes(x = home_form, fill = resultat)) +
  geom_histogram(aes(y = ..density..), bins = 15,
                 alpha = 0.65, position = "identity") +
  scale_fill_manual(values = c("Victoire domicile" = VERT,
                               "Défaite / nul"     = ROUGE)) +
  labs(title = "3. Forme domicile selon le résultat",
       x = "Forme récente (moy. pts sur 5 matchs)",
       y = "Densité", fill = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

# ============================================================
# GRAPHIQUE 4 : Boxplot forme par résultat et Covid
# ============================================================

df$covid_label <- ifelse(df$covid == 1, "Huis clos", "Public présent")

g4 <- ggplot(df, aes(x = resultat, y = home_form, fill = covid_label)) +
  geom_boxplot(alpha = 0.7) +
  scale_fill_manual(values = c("Public présent" = VERT,
                               "Huis clos"      = ROUGE)) +
  labs(title = "4. Forme domicile : Covid vs public",
       x = "", y = "Forme récente domicile", fill = "") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"),
        legend.position = "bottom")

# ============================================================
# SAUVEGARDE
# ============================================================

# Sauvegarder chaque graphique
ggsave("outputs/g1_taux_saison.png",      g1, width = 8, height = 5, dpi = 150)
ggsave("outputs/g2_courbe_logistique.png", g2, width = 8, height = 5, dpi = 150)
ggsave("outputs/g3_distribution_forme.png", g3, width = 8, height = 5, dpi = 150)
ggsave("outputs/g4_boxplot_covid.png",     g4, width = 8, height = 5, dpi = 150)

print("Graphiques sauvegardés dans outputs/")