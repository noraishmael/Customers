## import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

##load the dataset
df = pd.read_csv("C:/Users/nora_/Downloads/customers (1).csv")
df.head()

#check the type of values
df.info()

#check if there are values missing
df.isnull().sum()

# fill missing values in Ever_Married, Graduated, and Profession with Mode (i.e. most common answer)
df['Ever_Married'] = df['Ever_Married'].fillna(df['Ever_Married'].mode()[0])
df['Graduated'] = df['Graduated'].fillna(df['Graduated'].mode()[0])
df['Profession'] = df['Profession'].fillna(df['Profession'].mode()[0])


# fill missing values in Work_Experience and Family_Size with median
df['Work_Experience'] = df['Work_Experience'].fillna(df['Work_Experience'].median())
df['Family_Size'] = df['Family_Size'].fillna(df['Family_Size'].median())

## change object data for Gender, Ever_Married, Graduated, Profession, Spending Score, Var_1 to numerical value

df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Ever_Married'] = df['Ever_Married'].map({'No': 0, 'Yes': 1})
df['Graduated'] = df['Graduated'].map({'No': 0, 'Yes': 1})
df['Profession'] = df['Profession'].astype('category').cat.codes 
df['Spending_Score'] = df['Spending_Score'].map({'Low': 0, 'Average': 1, 'High': 2})
print(df.head())

print(df.columns)

# drop ID and Var_1

df.drop(columns=['ID'], inplace=True)
df.drop(columns=['Var_1'], inplace=True)

# compute the correlation matrix to demonstrate how strong the numerical variables are related to each other
correlation_matrix = df.corr()

# generate a correlation heatmap 
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# generate histograms for numerical features
df.hist(figsize=(10, 6), bins=20, edgecolor="black")
plt.suptitle("Feature Distributions (Histograms)", fontsize=14)
plt.show()

# generate boxplots for numerical features
plt.figure(figsize=(12, 6))
df.boxplot(rot=45)  # Rotate labels for better readability
plt.title("Feature Outliers (Boxplots)")
plt.show()

# scale data as some numerical values too far apart
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

## finding the k value
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df)
    sse.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('WCSS')
plt.plot(k_rng, sse)
plt.title('WCSS against K value')
plt.tight_layout()

## use the cluster to be 3
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df)
y_predicted

## put the cluster into the dataframe
df['Cluster'] = y_predicted

df

## centroid
km.cluster_centers_

df1 = df[df.Cluster==0]
df2 = df[df.Cluster==1]
df3 = df[df.Cluster==2]

plt.scatter(df1['Age'], df1['Spending_Score'], color = 'green')
plt.scatter(df2['Age'], df2['Spending_Score'], color = 'red')
plt.scatter(df3['Age'], df3['Spending_Score'], color = 'black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color = 'purple', marker='*', label = 'centroid')
plt.xlabel('Age')
plt.ylabel('Spending_Score')
plt.legend()
plt.title('Customer Clusters')
plt.tight_layout()

df1 = df[df.Cluster==0]
df2 = df[df.Cluster==1]
df3 = df[df.Cluster==2]

plt.scatter(df1['Spending_Score'], df1['Work_Experience'], color = 'green')
plt.scatter(df2['Spending_Score'], df2['Work_Experience'], color = 'red')
plt.scatter(df3['Spending_Score'], df3['Work_Experience'], color = 'black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color = 'purple', marker='*', label = 'centroid')
plt.xlabel('Spending_Score')
plt.ylabel('Work_Experience')
plt.legend()
plt.title('Customer Clusters')
plt.tight_layout()

df1 = df[df.Cluster==0]
df2 = df[df.Cluster==1]
df3 = df[df.Cluster==2]

plt.scatter(df1['Family_Size'], df1['Age'], color = 'green')
plt.scatter(df2['Family_Size'], df2['Age'], color = 'red')
plt.scatter(df3['Family_Size'], df3['Age'], color = 'black')
plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], color = 'purple', marker='*', label = 'centroid')
plt.xlabel('Family_Size')
plt.ylabel('Age')
plt.legend()
plt.title('Customer Clusters')
plt.tight_layout()
