import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
df=pd.read_csv('/HCV-Egy-Data.csv')
df.head()
df.info()
df.describe()
print(df.isna().sum(),df.isnull().sum())
from sklearn.preprocessing import StandardScaler
st=StandardScaler()
norm_data=st.fit_transform(df)
norm_data
k=3
k_means=KMeans(n_clusters=k)
k_means.fit(norm_data)
print(k_means.labels_)
df['cluster']=k_means.labels_
df.head()