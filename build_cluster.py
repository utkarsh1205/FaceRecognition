"""
Builds clusters from face embedding data and stores it in
cluster_tree.csv. Output dictionary stored in output_dict.json
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import time
import json

#array of face embeddings storred in face_embeddings.npy
face_embeddings = np.load("face_embeddings.npy")
#shape of original array is of (num_samples,1,512)
face_embeddings = np.reshape(face_embeddings,(face_embeddings.shape[0],512))


#converting to DataFrame
df = pd.DataFrame(face_embeddings)

output_dict = {}
num_layers=4
num_samples=df.shape[0]

for i in range(num_layers):
    print("****Layer{}***".format(i))
    start = time.time()
    #Converting to numpy array for faster traversing
    df_arr=np.asarray(df)
    if i==0:
        pos=[0,num_samples]
    else:
        pos=[0]
        curr=0
        for index in range(num_samples):
            if df_arr[index,-1]!=curr:
                pos.append(index)
                curr=df_arr[index,-1]
        pos.append(num_samples)
        
    temp_y = []
    temp_y = np.asarray(temp_y)
    for j in range(len(pos)-1):
        temp_df=df_arr[pos[j]:pos[j+1],:]
        
        if(i==0):
            X = temp_df
        else:
            X = temp_df[:,:-i]
        kmeans = KMeans(n_clusters=9,random_state=0, max_iter=1000)
        temp_y = np.append(temp_y,kmeans.fit_predict(X))
        centroids = (kmeans.cluster_centers_).astype(float)
        key=0
        index=j 
        for div in [9**exp for exp in range(i-1,-1,-1)]:
            key = key*10 + (index//div)%9 + 1
        
        for clust in range(9):
            output_dict[int(key*10+clust+1)] = list(centroids[clust])
            

    df["layer{}".format(i)]=temp_y    
    list_layers=["layer{}".format(k) for k in range(i+1)]
    df = df.sort_values(by=list_layers)
    end = time.time()
    print("[INFO] Clustering took {:.5} seconds".format(end - start))


df.to_csv("cluster_tree.csv")
    
with open("output_dict.json", 'w') as fp:
    json.dump(output_dict, fp)
