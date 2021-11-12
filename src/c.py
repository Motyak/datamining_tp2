import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

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
#     df.boxplot(col)
#     plt.savefig("boxplot" + col + ".png")

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
transformed = imputer.fit_transform(data[numerical_attrs])
filled_data = pd.DataFrame(data=transformed, index=data[numerical_attrs].index, columns=data[numerical_attrs].columns)
filled_data = data[categorical_attrs].join(filled_data)

print(filled_data)
# print(filled_data.isnull().sum().sum()) # 0 missing value
