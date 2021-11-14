#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from R_square_clustering import r_square
from sklearn.cluster import KMeans

# lecture du fichier csv dans une dataframe
data = pd.read_csv("data/Data_World_Development_Indicators2.csv")
categorical_attrs = [data.columns[i] for i in range(2)]
numerical_attrs = [data.columns[i] for i in range(2, len(data.columns))]

# # le nombre d'attributs/colonnes
# len(data.columns)

# # les premières valeurs et le type de chaque colonne
# for c in data.columns:
#     data.head()[c]
#     print()

# data.isnull().sum().sum()

# # Permet de créer un boxplot par attribut pour voir la distribution des valeurs
# for col in numerical_attrs:
#     df = data[[col]]
#     fig = plt.figure()
#     df.boxplot(col)
#     plt.savefig("fig/boxplot" + col + ".png")
#     plt.close(fig)

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
transformed = imputer.fit_transform(data[numerical_attrs])
filled_data = pd.DataFrame(data=transformed, index=data[numerical_attrs].index, columns=data[numerical_attrs].columns)
filled_data = data[categorical_attrs].join(filled_data)

# print(filled_data)
# print(filled_data.isnull().sum().sum()) # 0 missing value

transformed = StandardScaler().fit_transform(filled_data[numerical_attrs])
standardized_data = pd.DataFrame(data=transformed, index=data[numerical_attrs].index, columns=data[numerical_attrs].columns)
standardized_data = filled_data[categorical_attrs].join(standardized_data)





#############PCA





pca = PCA(svd_solver='full')
coord = pca.fit_transform(standardized_data[numerical_attrs])

n = np.size(standardized_data[numerical_attrs], 0)
p = np.size(standardized_data[numerical_attrs], 1)
# print(n, p)

# # nb of computed components
# print(pca.n_components_) 

# # explained variance scores
# print(pca.explained_variance_ratio_)

# plot instances on the first plan (first 2 factors)
fig, axes = plt.subplots(figsize=(6,6))
axes.set_xlim(-5,10)
axes.set_ylim(-5,8)
plt.title('ACP premier plan')
plt.xlabel('Composante 1')
plt.ylabel('Composante 2')
for i in range(n):
    plt.annotate(data['Country Code'].values[i],(coord[i,0],coord[i,1]))
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
plt.savefig('fig/acp_factor_1_2.png')
plt.close(fig)

# plot instances on factors 3 and 4
fig, axes = plt.subplots(figsize=(6,6))
axes.set_xlim(-5,8)
axes.set_ylim(-5,8)
plt.title('ACP second plan')
plt.xlabel('Composante 3')
plt.ylabel('Composante 4')
for i in range(n):
    plt.annotate(data['Country Code'].values[i],(coord[i,2],coord[i,3]))
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
plt.savefig('fig/acp_factor_3_4.png')
plt.close(fig)

# plot eigen values
eigval = float(n-1)/n*pca.explained_variance_
fig = plt.figure()
plt.plot(eigval)
plt.title('Scree plot')
plt.ylabel('Eigen values')
plt.xlabel('Factor number')
plt.savefig('fig/acp_eigen_values')
plt.close(fig)

# # print eigen vectors
# print(pca.components_)
# # lines: factors
# # columns: variables

# print correlations between factors and original variables
sqrt_eigval = np.sqrt(eigval)
corvar = np.zeros((p,p))
for k in range(p):
    corvar[:,k] = pca.components_[k,:] * sqrt_eigval[k]
# print(corvar)
# lines: variables
# columns: factors

fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)
# label with variable names
for j in range(p):
    # ignore two first columns of df: Nom and Code^Z
    plt.annotate(data.columns[j+2],(corvar[j,0],corvar[j,3]))
# axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
# add a circle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
plt.savefig('fig/acp_correlation_circle_axes_'+str(0)+'_'+str(3))
plt.close(fig)




#############KMEANS





# Plot elbow graphs for KMeans using R square and purity scores
lst_k=range(2,21)
lst_rsq = []
for k in lst_k:
    est=KMeans(n_clusters=k)
    est.fit(standardized_data[numerical_attrs])
    lst_rsq.append(r_square(standardized_data[numerical_attrs].to_numpy(), est.cluster_centers_,est.labels_,k))
fig = plt.figure()
plt.plot(lst_k, lst_rsq, 'bx-')
plt.xlabel('k')
plt.ylabel('RSQ score')
plt.title('The Elbow Method showing the optimal k')
plt.savefig('fig/k-means_elbow_method')
plt.close()

kmeans=KMeans() #default n_clusters=8
clusters = kmeans.fit_predict(standardized_data[numerical_attrs])
fig = plt.figure(figsize=(8, 8))
plt.scatter(standardized_data[numerical_attrs].to_numpy()[:,0], standardized_data[numerical_attrs].to_numpy()[:,1], c=clusters)
plt.savefig('fig/k-means_k8')
plt.close()
