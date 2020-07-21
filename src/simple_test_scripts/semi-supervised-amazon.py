import json
import numpy as np
import pandas as pd
# from pandarallel import pandarallel

from os.path import join
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

data_path = r"..\input\data"
Data_folder = "itemmaster"
# Data_folder = "amazon"

fname = "ukecomm_data.csv"
# fname = "trn.raw.json"
# pandarallel.initialize()
trn_path = join(data_path,Data_folder,fname)
# raw_trn_path = join(data_path,"trn.raw.json")

# trn_df = pd.read_json(trn_path,lines=True)
# raw_trn_df = pd.read_json(raw_trn_path,lines=True)
# df = trn_df
# print(trn_df.head())
# print(raw_trn_df.head())

model = SentenceTransformer('bert-base-nli-mean-tokens')

# loading data
if Data_folder == "amazon":
    trn_df = pd.read_json(trn_path,lines=True)
    df = trn_df
    df.rename(columns={"title":"Description"},inplace =True)
else:
    df = pd.read_csv(trn_path,encoding="latin-1")

def driver(df):
    

    df = df.astype(str)[:1000]

    # drop unwanted columns
    df = df[['Description']]

    # vectorizing input
    df['embed'] = model.encode(df['Description'])
    df['words'] = df['Description'].apply(lambda x : np.array(x.lower().split()))

    #Kmeans
    # for k in range(2,int(np.sqrt(len(df)))):
    for k in range(70,100):
        cluster = KMeans(n_clusters=k, random_state=0).fit(df['embed'].to_list())
        df['cluster_id'] = cluster.labels_
        filtered_df = df
        n_clusters = len(filtered_df['cluster_id'].unique())
        # n_noise = len(df[df['cluster_id']==-1])
        cluster_centers = cluster.cluster_centers_
        ss = silhouette_score(filtered_df['embed'].to_list(), filtered_df['cluster_id'].to_list())

        print( "\tClusters: ", k, "\tsilhouette_score: " ,ss)

    '''
    # DBSCAN
    for eps in np.arange(0.1, 1, 0.1):
        db = DBSCAN(eps=eps, min_samples=5,metric="cosine").fit(df['embed'].to_list())
        df['cluster_id'] = db.labels_
        filtered_df = df
        # filtered_df = df[df['cluster_id']!=-1]
        n_clusters = len(filtered_df['cluster_id'].unique())
        n_noise = len(df[df['cluster_id']==-1])
        if len(filtered_df)>0 and 2<=n_clusters<len(df)-1:
            ss = silhouette_score(filtered_df['embed'].to_list(), filtered_df['cluster_id'].to_list(), metric='cosine')
        else:
            ss = "NA"
        print("EPS: ", round(eps,3), "\tClusters: ", n_clusters, "\tNoise: ", n_noise, "\tsilhouette_score: " ,ss)

        # if n_clusters ==1:
        #     break
    print(df)
    '''
driver(df)

