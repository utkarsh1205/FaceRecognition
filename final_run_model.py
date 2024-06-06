"""
For performing predictions
"""
import numpy as np
import pandas as pd
import cv2
import time
from numpy.linalg import norm
import json
from detect_face import gen_embedding, detect_embed
from os.path import join

def get_divs(df_arr,layer):
    """ 
    Dataset is sorted according to their cluster numbers.
    This is a uility function to get indexed partitions of data
    according to clusters.

    Parameters
    ----------
    df : numpy array
        Numpy array oject of shape(num_samples,512+num_layers).
    layer : int
        0 indexed layer number.

    Returns
    -------
    pos : list
        list of indices where division of cluster
        corresponding to layer occurs.
        Also includes 0 and last index

    """
    pos=[0]
    curr=0
    num_samples=df_arr.shape[0]
    for index in range(num_samples):
        if df_arr[index,-(4-layer)]!=curr:
            pos.append(index)
            curr=df_arr[index,-(4-layer)]
    pos.append(num_samples)
    return pos

def score(test_emb,df_emb):
    """
    Calculates face recognition score

    Parameters
    ----------
    test_emb : np array of shape(512,)
        Face embedding of test image.
    df_emb : np array of shape(512,)
        Face embedding of sample to compare with.

    Returns
    -------
    score : float64
        face recognition score.

    """
    dot_prod = np.dot(test_emb,df_emb)
    test_l2 = norm(test_emb)
    df_l2 = norm(df_emb)
    score = dot_prod / (test_l2 * df_l2)
    return score
        

def predict_cluster(test_embedding,output_dict):
    """
    Predicts the cluster image belongs to

    Parameters
    ----------
    test_embedding : numpy array, shape(1,512)
        face embedding data of test image.
    output_dict : python dictionary
        Output dictionary-
        key -> cluster number.
        value -> centroid of cluster, shape(512)

    Returns
    -------
    im_preds : list, shape(4,)
        list of cluster predictions.

    """ 
    start=time.time()
    im_preds=[]
    key=0
    for i in range(4):
        min_dist=-1
        key=0
        prediction = 0
        if len(im_preds)==0:
            key=0
        else:
            for pred in im_preds:
                key = key*10 + pred + 1
                
        for cluster in range(1,10):
            a = np.asarray(output_dict[str(key*10 + cluster)])
            b = np.asarray(test_embedding[0])
            dist = norm(a-b)
            if min_dist==-1:
                min_dist=dist
                prediction=cluster-1
            elif dist<min_dist:
                min_dist = dist
                prediction = cluster-1
        im_preds.append(prediction)
    end = time.time()
    print("Prediction took {:.5} seconds".format(end-start))
    return im_preds

                
def get_scores(im_preds,df,test_embedding):
    """
    

    Parameters
    ----------
    im_preds : list, shape(4,)
        cluster predictions of test embedding. Output of predict_cluster().
    df : pandas.Dataframe object, shape(num_samples, 516)
        Contains entire dataset.
    test_embedding : numpy array, shape(1,512)
        face embedding data of test image.

    Returns
    -------
    scores : list, shape(number of dataPoints in final cluster)
        List for scores corresponding to each node in dataset
        in last predicted cluster.

    """
    
    start = time.time()
    df_arr = np.asarray(df)
    pos = get_divs(df_arr,3)
    cluster_num=0
    for pred in im_preds:
        cluster_num *=9
        cluster_num+=pred
            
    pred_cluster = df_arr[pos[cluster_num]:pos[cluster_num+1],:-4]
    scores = [] 
    for sample in range(pred_cluster.shape[0]):
        temp_score = score(test_embedding[0],pred_cluster[sample])
        scores.append(temp_score)                

    end = time.time()
    print("Generating scores took {:.5} seconds".format(end-start))
    return scores

if __name__ == '__main__':
#Example code to implement above function

    df = pd.read_csv("cluster_tree.csv")
    #df is of shape(num_samples,517)
    #Unnamed: 0 is an extra column having values of original indexes corresponding to original image numbers before dataframe got sorted.
    df = df.drop(columns='Unnamed: 0')

    with open("output_dict.json",'r') as fp:
        output_dictionary = json.load(fp)
    
    model_path = 'mxnet_exported_res34.onnx'
    model = cv2.dnn.readNetFromONNX(model_path)
    img_path = join("Aligned_Test_Images","0_0.jpg")
    
    #embedding=detect_embed(img_path,model) #Use this if need to perform detection and alignment before generating embedding. Comment out below two lines if using this.    
    img_data = cv2.imread(img_path)
    embedding = gen_embedding(img_data,model)
    
    cluster_preds= predict_cluster(embedding,output_dictionary)
    all_scores = get_scores(cluster_preds,df,embedding)

    import matplotlib.pyplot as plt
    plt.grid('True')
    plt.plot(all_scores)
