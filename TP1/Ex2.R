# Chargement des données
load("TP1/DonneesSNPnormalisees.RData")

# Question 1 : Observation du jeu de données
# Affichage de la structure de l'objet data
str(data)

# Question 2 : Réalisation de l'ACP sur la matrice des génotypes
# Utilisation de prcomp car plus de variables que d'observations
pca_result <- prcomp(data$scaled.Geno)

# Question 3 : Observation du nombre de composantes principales
# Affichage du résumé de l'ACP
summary(pca_result)

# Question 4 : Détermination du nombre d'axes à conserver
# Scree plot pour visualiser les valeurs propres
# Enregistrement du graphique
png("scree_plot_snp.png")
plot(pca_result$sdev^2, type = "l", main = "Scree Plot", xlab = "Composante principale", ylab = "Variance expliquée")
dev.off()

# Question 5 : Représentation des individus en distinguant les populations
# Création d'un vecteur de couleurs pour les populations
population_colors <- as.factor(data$origin)
col_vector <- as.numeric(population_colors)

# Projection des individus sur les deux premiers axes
# Enregistrement du graphique
png("individus_snp.png")
plot(pca_result$x[,1], pca_result$x[,2], col = col_vector, pch = 19,
     xlab = "CP1", ylab = "CP2", main = "Projection des individus sur les axes principaux")
legend("topright", legend = levels(population_colors), col = 1:length(levels(population_colors)), pch = 19)
dev.off()