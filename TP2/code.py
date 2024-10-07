# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

# Chemin vers le fichier de données
file_path = './Metasample.txt'

# 1. Chargement des données
# -------------------------

# Chargement du fichier texte en spécifiant le séparateur de tabulation
data = pd.read_csv(file_path, sep='\t', index_col=0)

# Affichage des premières lignes pour vérifier le chargement
print("Aperçu des données originales :")
print(data.head())

# Suppression de la première ligne (lectures non affectées)
data = data.drop(data.index[0])

# Vérification après suppression
print("\nDonnées après suppression de la première ligne :")
print(data.head())

# 2. Préparation des données pour l'analyse
# -----------------------------------------

# Transposition du tableau pour avoir les échantillons en lignes et les genres en colonnes
data_T = data.transpose()

# Vérification de la forme des données transposées
print(f"\nDimensions des données transposées : {data_T.shape}")

# Remplacement des valeurs manquantes éventuelles par 0
data_T.fillna(0, inplace=True)

# Normalisation des données pour que chaque variable ait une moyenne de 0 et un écart-type de 1
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_T)

# 3. Détermination du nombre optimal de clusters avec le coefficient de silhouette
# --------------------------------------------------------------------------------

# Liste pour stocker les coefficients de silhouette pour chaque valeur de k
silhouette_scores = []

# Tester k de 2 à 10
K = range(2, 11)
for k in K:
    # Application du k-means avec k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(data_scaled)
    # Calcul du coefficient de silhouette moyen
    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)
    print(f"Pour k = {k}, le coefficient de silhouette moyen est de {silhouette_avg:.4f}")

# Tracé du graphe du coefficient de silhouette en fonction de k
plt.figure(figsize=(8, 6))
plt.plot(K, silhouette_scores, 'bx-')
plt.xlabel('Nombre de clusters k')
plt.ylabel('Coefficient de silhouette moyen')
plt.title('Coefficient de silhouette pour déterminer le k optimal')
plt.xticks(K)
plt.show()

# Détermination du k optimal (k qui maximise le coefficient de silhouette)
optimal_k = K[silhouette_scores.index(max(silhouette_scores))]
print(f"\nLe nombre optimal de clusters est k = {optimal_k}")

# 4. Application du k-means avec le k optimal
# -------------------------------------------

# Application du k-means avec le nombre optimal de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(data_scaled)

# Ajout des labels de clusters aux données originales
data_T['Cluster'] = kmeans.labels_

# 5. Identification de la variable la plus discriminante par ANOVA
# ---------------------------------------------------------------

# Dictionnaire pour stocker les p-valeurs pour chaque variable
p_values = {}
variables = data_T.columns[:-1]  # Exclure la colonne 'Cluster'

# 5. Identification de la variable la plus discriminante par ANOVA
# ---------------------------------------------------------------

# Liste des variables à tester (exclure celles avec variance nulle)
zero_variance_vars = []

for var in variables:
    variances = [data_T[data_T['Cluster'] == i][var].var() for i in range(optimal_k)]
    if any(v == 0 for v in variances):
        zero_variance_vars.append(var)

# Exclusion des variables avec variance nulle
variables_to_test = [var for var in variables if var not in zero_variance_vars]

print(f"Nombre de variables avec variance non nulle : {len(variables_to_test)}")

# Dictionnaire pour stocker les p-valeurs
p_values = {}

for var in variables_to_test:
    groups = [data_T[data_T['Cluster'] == i][var] for i in range(optimal_k)]
    try:
        f_stat, p_val = f_oneway(*groups)
        if not np.isnan(p_val):
            p_values[var] = p_val
    except Exception as e:
        print(f"Erreur lors du test ANOVA pour la variable '{var}': {e}")

if len(p_values) == 0:
    print("Aucune variable n'a pu être testée avec l'ANOVA.")
else:
    # Tri des p-valeurs par ordre croissant
    sorted_p_values = sorted(p_values.items(), key=lambda item: item[1])

    # Identification de la variable avec la plus petite p-valeur
    most_discriminative_var = sorted_p_values[0][0]
    print(f"La variable la plus discriminante est : {most_discriminative_var}")
    print(f"P-valeur associée : {sorted_p_values[0][1]:.4e}")
# 6. Représentation des clusters par un boxplot sur la variable la plus discriminante
# -----------------------------------------------------------------------------------

plt.figure(figsize=(8, 6))
sns.boxplot(x='Cluster', y=most_discriminative_var, data=data_T)
plt.title(f'Boxplot de {most_discriminative_var} par cluster')
plt.xlabel('Cluster')
plt.ylabel(f'Abondance de {most_discriminative_var}')
plt.show()

# 7. Analyse en composantes principales (ACP) et visualisation
# ------------------------------------------------------------

# Application de l'ACP pour réduire la dimensionnalité à 2 composantes principales
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled)

# Création d'un DataFrame pour les composantes principales
principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
principalDf['Cluster'] = data_T['Cluster'].values

# Tracé du scatter plot des deux premières composantes principales
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=principalDf, palette='Set1', s=100)
plt.title('Projection des données sur les deux premières composantes principales')
plt.xlabel('Composante principale 1')
plt.ylabel('Composante principale 2')
plt.legend(title='Cluster')
plt.show()

# 8. Observations
# ---------------

print("\nObservations :")
print("En visualisant les données projetées sur les deux premières composantes principales,")
print("nous pouvons observer comment les échantillons se répartissent et si les clusters")
print("sont bien séparés. Une bonne séparation entre les clusters indique que le nombre de")
print("clusters choisi est approprié et que les clusters identifiés par le k-means sont cohérents.")