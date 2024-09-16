import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.multivariate.manova import MANOVA

# Crear DataFrame con todos los datos
data_dict = {
    'Accession': ['Los naranjos', 'Los naranjos', 'Los naranjos', 'Los naranjos', 'Queixpan Ameca', 'Queixpan Ameca', 
                  'Queixpan Ameca', 'Queixpan Ameca', 'Las raíces', 'Las raíces', 'Las raíces', 'Las raíces', 'Talpitita', 
                  'Talpitita', 'Talpitita', 'Talpitita', 'Ejutla entronque', 'Ejutla entronque', 'Ejutla entronque', 
                  'Ejutla entronque', 'Ejutla', 'Ejutla', 'Ejutla', 'Ejutla', 'Los cimientos', 'Los cimientos', 'Los cimientos', 
                  'Los cimientos', 'Amacautitlanejo', 'Amacautitlanejo', 'Amacautitlanejo', 'Amacautitlanejo', 'La paz de Milpillas', 
                  'La paz de Milpillas', 'La paz de Milpillas', 'La paz de Milpillas', 'El cuyotomate', 'El cuyotomate', 
                  'El cuyotomate', 'El cuyotomate', 'Guachinango', 'Guachinango', 'Guachinango', 'Guachinango', 
                  'La cienega (El fresno)', 'La cienega (El fresno)', 'La cienega (El fresno)', 'La cienega (El fresno)', 
                  'El Tablillo', 'El Tablillo', 'El Tablillo', 'El Tablillo', 'San Lorenzo', 'San Lorenzo', 'San Lorenzo', 
                  'San Lorenzo', 'El rodeo', 'El rodeo', 'El rodeo', 'El rodeo', 'B73', 'B73', 'B73', 'B73'],
    'Fresh_stem_weight_Infected': [0.89, 0.88, 0.77, 0.98, 1.26, 1.39, 1.27, 1.16, 1.08, 1.12, 1.11, 1.03, 0.52, 0.66, 0.59, 0.62, 0.76, 0.74, 0.78, 0.98, 0.3, 0.89, 1.02, 1.01, 0.78, 1.28, 0.94, 1.13, 0.32, 0.74, 1.12, 0.54, 0.6, 1.84, 0.94, 0.81, 0.75, 0.8, 0.93, 0.95, 1.13, 0.75, 1.13, 1.28, 1.02, 1.14, 0.35, 0.74, 1.28, 1.19, 2.35, 1.62, 0.72, 0.56, 0.83, 0.56, 1, 0.72, 0.93, 0.4, 1.17, 1.24, 1.41, 1.42],
    'Fresh_root_weight_Infected': [5.43, 4.94, 2.9, 2.94, 1.87, 2.44, 2.2, 1.29, 3.32, 2.55, 3.78, 1.73, 1.98, 1.26, 1.44, 1.52, 2.74, 0.91, 2.28, 2.85, 1.24, 2.48, 3.27, 2.52, 1.94, 2.53, 2.95, 1.99, 1.28, 2.44, 3.4, 2.02, 1.38, 2.45, 1.88, 2.16, 3.24, 2.68, 3.71, 3.34, 1.65, 1.79, 2.51, 3.48, 2.55, 3.18, 0.43, 2.17, 2.12, 4.36, 3.5, 1.53, 2.9, 2.54, 1.87, 1.35, 3.74, 3.33, 3.37, 2.2, 0.44, 2.22, 1.55, 0.92],
    'Fresh_stem_weight_Control': [0.92, 0.58, 0.95, 0.61, 1.59, 1.24, 1.34, 1.37, 0.92, 0.91, 0.80, 0.94, 0.63, 0.51, 0.48, 0.56, 0.92, 0.74, 0.94, 1.12, 0.63, 0.86, 0.92, 1.23, 1.18, 1.01, 0.90, 1.49, 1.19, 1.32, 1.04, 1.17, 0.99, 0.82, 0.52, 0.65, 1.38, 1.98, 1.15, 1.38, 1.01, 0.53, 0.83, 1.06, 1.25, 1.39, 2.03, 2.55, 1.4, 1.87, 2.13, 1.58, 0.78, 0.87, 1.22, 0.62, 0.81, 0.91, 0.77, 0.84, 2.38, 2.75, 2.75, 3.05],
    'Fresh_root_weight_Control': [4.19, 3.35, 2.59, 2.63, 5.49, 2.73, 5.08, 2.74, 3.76, 2.54, 2.78, 3.51, 4.00, 0.38, 3.89, 3.82, 1.89, 2.59, 5.36, 4.37, 1.57, 3.22, 2.76, 0.29, 4.07, 2.90, 3.33, 3.97, 3.95, 3.89, 3.71, 2.37, 3.85, 1.82, 4.15, 3.28, 3.98, 3.80, 2.95, 3.22, 2.41, 0.79, 4.50, 4.08, 4.24, 2.05, 2.61, 2.15, 3.47, 4.24, 6.12, 2.52, 1.2, 2.65, 3.48, 3.15, 2.69, 1.65, 2.56, 2.63, 3.99, 3.91, 4.04, 3.12],
    'Dry_Stem_weight_Infected': [0.42, 0.42, 0.37, 0.47, 0.7, 0.77, 0.71, 0.64, 0.49, 0.51, 0.5, 0.47, 0.31, 0.39, 0.35, 0.36, 0.33, 0.32, 0.34, 0.43, 0.16, 0.47, 0.54, 0.53, 0.46, 0.75, 0.55, 0.66, 0.19, 0.44, 0.66, 0.32, 0.24, 0.74, 0.38, 0.32, 0.39, 0.42, 0.49, 0.5, 0.54, 0.36, 0.54, 0.61, 0.41, 0.46, 0.14, 0.3, 0.58, 0.54, 1.07, 0.74, 0.28, 0.22, 0.32, 0.22, 0.59, 0.42, 0.55, 0.24, 0.53, 0.56, 0.64, 0.65],
    'Dry_root_weight_Infected': [0.75, 0.69, 0.4, 0.41, 0.41, 0.53, 0.48, 0.28, 0.77, 0.59, 0.88, 0.4, 0.35, 0.22, 0.26, 0.27, 0.55, 0.18, 0.45, 0.57, 0.42, 0.85, 1.12, 0.86, 0.29, 0.38, 0.44, 0.3, 0.24, 0.45, 0.63, 0.38, 0.25, 0.45, 0.35, 0.4, 0.52, 0.43, 0.6, 0.54, 0.24, 0.27, 0.37, 0.52, 0.37, 0.46, 0.06, 0.31, 0.34, 0.7, 0.56, 0.25, 0.39, 0.34, 0.25, 0.18, 0.52, 0.46, 0.47, 0.3, 0.1, 0.5, 0.35, 0.21],
    'Dry_Stem_weight_Control': [0.71, 0.45, 0.73, 0.47, 1.06, 0.83, 0.89, 0.91, 0.71, 0.7, 0.62, 0.72, 0.39, 0.32, 0.3, 0.35, 0.66, 0.53, 0.67, 0.8, 0.39, 0.54, 0.58, 0.77, 0.91, 0.78, 0.69, 1.15, 0.99, 1.1, 0.87, 0.98, 0.58, 0.48, 0.31, 0.38, 0.99, 1.41, 0.82, 0.99, 0.63, 0.33, 0.52, 0.66, 0.89, 0.99, 1.45, 1.82, 0.82, 1.1, 1.25, 0.93, 0.52, 0.58, 0.81, 0.41, 0.54, 0.61, 0.51, 0.56, 1.49, 1.72, 1.72, 1.91],
    'Dry_root_weight_Control': [2.46, 1.97, 1.52, 1.54, 2.89, 1.44, 2.68, 1.44, 2.35, 1.58, 2.19, 2.19, 2.22, 0.21, 2.16, 2.12, 0.99, 1.36, 2.82, 2.3, 1.05, 2.15, 1.84, 0.19, 2.14, 1.53, 1.75, 2.09, 2.82, 2.78, 2.65, 1.69, 2.14, 1.01, 2.3, 1.82, 2.84, 2.71, 2.11, 2.3, 1.72, 0.57, 3.21, 2.92, 2.02, 0.98, 1.24, 1.02, 2.04, 2.49, 3.6, 1.48, 0.66, 1.47, 1.94, 1.75, 1.42, 0.87, 1.35, 1.38, 1.81, 1.78, 1.84, 1.42],
    'Average_weight_mg': [3.479, 4.422, 2.082, 3.292, 0.0, 1.738, 2.995, 1.581, 3.318, 0.0, 2.825, 3.034, 0.0, 0.657, 7.198, 6.067, 1.51, 2.576, 4.198, 0.0, 0.0, 3.999, 3.195, 0.248, 2.359, 1.964, 2.53, 1.821, 0.0, 2.524, 3.054, 1.733, 0.0, 2.096, 0.0, 0.0, 2.883, 1.917, 2.566, 2.336, 2.721, 1.71, 6.193, 4.401, 2.26, 0.985, 0.0, 0.561, 2.479, 2.266, 2.875, 1.596, 1.278, 2.54, 0.0, 4.228, 2.623, 1.432, 2.626, 2.468, 1.22, 1.035, 1.069, 0.744],
    'Survival': [20, 10, 60, 50, 0, 90, 70, 40, 60, 0, 70, 50, 0, 20, 40, 30, 20, 50, 20, 0, 0, 10, 50, 50, 20, 70, 60, 90, 0, 30, 20, 20, 0, 90, 0, 0, 50, 40, 50, 10, 60, 80, 40, 40, 50, 40, 0, 40, 50, 10, 60, 30, 20, 20, 0, 90, 80, 70, 70, 20, 80, 80, 60, 40]
}

# Crear DataFrame
data = pd.DataFrame(data_dict)

# Calcular índice de reducción de peso
data['stem_weight_reduction'] = (data['Fresh_stem_weight_Control'] - data['Fresh_stem_weight_Infected']) / data['Fresh_stem_weight_Control']
data['root_weight_reduction'] = (data['Fresh_root_weight_Control'] - data['Fresh_root_weight_Infected']) / data['Fresh_root_weight_Control']

# Calcular medias y desviaciones estándar por accesión
means = data.groupby('Accession').mean().reset_index()
stds = data.groupby('Accession').std().reset_index()

# Estandarizar datos
scaler = StandardScaler()
variables = ['stem_weight_reduction', 'root_weight_reduction', 'Average_weight_mg', 'Survival',
             'Dry_Stem_weight_Infected', 'Dry_root_weight_Infected', 'Dry_Stem_weight_Control', 'Dry_root_weight_Control']
means[variables] = scaler.fit_transform(means[variables])
data[variables] = scaler.transform(data[variables])

# Realizar MANOVA
manova = MANOVA.from_formula('stem_weight_reduction + root_weight_reduction + Average_weight_mg + Survival + Dry_Stem_weight_Infected + Dry_root_weight_Infected + Dry_Stem_weight_Control + Dry_root_weight_Control ~ Accession', data=means)
fit = manova.mv_test()
print(fit)

# Obtener centroids y graficar Canonical Centroid Plot
pca = PCA(n_components=2)
pca_result_means = pca.fit_transform(means[variables])
pca_result_data = pca.transform(data[variables])

means['pca_one'] = pca_result_means[:, 0]
means['pca_two'] = pca_result_means[:, 1]
data['pca_one'] = pca_result_data[:, 0]
data['pca_two'] = pca_result_data[:, 1]

# Realizar clustering K-means
kmeans = KMeans(n_clusters=3, random_state=42)  # Número de grupos (ajusta según sea necesario)
means['cluster'] = kmeans.fit_predict(pca_result_means)

# Obtener porcentaje de variación explicada
explained_variance = pca.explained_variance_ratio_ * 100

# Diccionario de variables y colores
var_dict = {1: 'stem_weight_reduction', 2: 'root_weight_reduction', 3: 'Average_weight_mg', 4: 'Survival',
            5: 'Dry_Stem_weight_Infected', 6: 'Dry_root_weight_Infected', 7: 'Dry_Stem_weight_Control', 8: 'Dry_root_weight_Control'}
colors = plt.cm.tab10(np.linspace(0, 1, len(var_dict)))

#Colores para los clusters
cluster_colors = ['red', 'blue', 'green']

# Graficar
plt.figure(figsize=(14, 10))
for cluster in range(3):
    subset = means[means['cluster'] == cluster]
    plt.scatter(subset['pca_one'], subset['pca_two'], label=f'Cluster {cluster + 1}', color=cluster_colors[cluster], edgecolor='k', s=100)
    for _, row in subset.iterrows():
        plt.text(row['pca_one'], row['pca_two'] - 0.1, row['Accession'], fontsize=10, fontname='Arial', weight='bold', color='black', ha='center', va='center')

# Añadir flechas para variables dependientes con números y flechas punteadas
for i, (num, var) in enumerate(var_dict.items()):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i], color=colors[i], alpha=0.7, linestyle=':', linewidth=2)
    plt.text(pca.components_[0, i]*1.15, pca.components_[1, i]*1.15, str(num), color=colors[i], fontsize=12, fontname='Arial', ha='center', va='center')

# Cuadro de acotaciones
legend_labels = [f"{num}: {var}" for num, var in var_dict.items()]
plt.legend(legend_labels, title="Variables", loc='upper right', prop={'family': 'Arial', 'size': 12})

plt.title('Canonical Centroid Plot with Clustering', fontsize=16, fontname='Arial')
plt.xlabel(f'PCA Component 1 ({explained_variance[0]:.2f}%)', fontsize=14, fontname='Arial')
plt.ylabel(f'PCA Component 2 ({explained_variance[1]:.2f}%)', fontsize=14, fontname='Arial')
plt.grid(True)
plt.show()