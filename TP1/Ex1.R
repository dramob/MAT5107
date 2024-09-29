# Chargement de la bibliothèque nécessaire
library(MASS)

# Chargement du jeu de données 'crabs'
data(crabs)

# Attribution du jeu de données à 'crabs_data' (optionnel)
crabs_data <- crabs

# Vérification du chargement des données
head(crabs_data)

# Sélection des variables quantitatives (FL, RW, CL, CW, BD)
crabsquant <- crabs_data[, 4:8]

# Question 1 : ACP sans traitement préalable
# Réalisation de l'ACP sans standardisation
acp1 <- prcomp(crabsquant, scale. = FALSE)

# Résumé des résultats
summary(acp1)

# Biplot de l'ACP
# Enregistrement du graphique
png("biplot_acp1.png")
biplot(acp1, main = "Biplot ACP sans standardisation")
dev.off()

# Question 2 : Amélioration de la représentation
# Réalisation de l'ACP avec standardisation
acp2 <- prcomp(crabsquant, scale. = TRUE)

# Résumé des résultats
summary(acp2)

# Biplot amélioré
# Enregistrement du graphique
png("biplot_acp2.png")
biplot(acp2, main = "Biplot ACP avec standardisation")
dev.off()

# Question 3 : Qualité de la nouvelle ACP et nombre d'axes à retenir
# Affichage du pourcentage de variance expliquée
summary(acp2)

# Scree plot
# Enregistrement du graphique
png("scree_plot.png")
plot(acp2, type = "l", main = "Scree Plot")
dev.off()

# Question 4 : Interprétation des axes à partir du cercle des corrélations
# Calcul des coordonnées des variables pour le cercle des corrélations
correlations_circle <- t(t(acp2$rotation) * acp2$sdev)

# Tracé du cercle des corrélations
# Enregistrement du graphique
png("cercle_correlations.png")
plot(c(-1, 1), c(-1, 1), type = "n", xlab = "CP1", ylab = "CP2",
     main = "Cercle des corrélations")
symbols(0, 0, circles = 1, inches = FALSE, add = TRUE)
arrows(0, 0, correlations_circle[,1], correlations_circle[,2],
       length = 0.1, angle = 15, code = 2)
text(correlations_circle[,1], correlations_circle[,2],
     labels = rownames(correlations_circle), pos = 4)
dev.off()

# Question 5 : Caractérisation des mâles/femelles et des crabes oranges/bleus
# Création d'une variable groupement combinant espèce et sexe
group <- interaction(crabs_data$sex, crabs_data$sp)

# Récupération des scores des individus
scores <- acp2$x

# Projection des individus avec coloration par groupe
# Enregistrement du graphique
png("projection_individus.png")
plot(scores[,1], scores[,2], col = as.numeric(group), pch = 19,
     xlab = "CP1", ylab = "CP2", main = "Projection des individus")
legend("topright", legend = levels(group), col = 1:length(levels(group)), pch = 19)
dev.off()