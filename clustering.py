# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 22:54:14 2018
@author: murali.sai
"""
import os, cv2, numpy as np, pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import random
random.seed(10)

import params
from data_m3 import get_bw_mask
all_train_folders = os.listdir(params.train_folder_org)

#==============================================================================
# YCR_CB Replaced with HSV!!!
#==============================================================================

def dominant_clusters(image, n_clusters=1):
    img = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    clt = KMeans(n_clusters = n_clusters)
    clt.fit(img)
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    return clt.cluster_centers_.tolist(), hist.tolist(), clt
def get_all_dominant_clusters(dominant_centers = []):
    for ind, item in enumerate(all_train_folders):
        print(ind);
        # Get Paths
        image_path = os.path.join(params.train_folder_org,item+'/images');
        # Load image and masks
        image_file = os.listdir(image_path)[0];
        bgr_image = cv2.imread(os.path.join(image_path,image_file));
        HSV_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        clusters, _ , _ = dominant_clusters(HSV_image, n_clusters=1);
        dominant_centers.append(clusters[0]);
    return dominant_centers

if __name__=="__main__":

    # Get representative cluster of each image
    dominant_centers = get_all_dominant_clusters()
    inertias = [];
    all_clusters_centers = np.stack(dominant_centers);
    all_clusters_centers = np.expand_dims(all_clusters_centers, axis=0)
    for n_clusters in range(1,10):
        _, _, clt = dominant_clusters(all_clusters_centers, n_clusters=n_clusters);
        inertias.append(clt.inertia_);
    plt.plot(inertias[1:])
    del inertias, n_clusters, clt;

    # Real Clustering
    n_clusters = 4
    cluster_centers_, cluster_percent_, clt = dominant_clusters(all_clusters_centers, n_clusters=n_clusters);
    del all_clusters_centers;

    # Make Folders/ Initialization
    images_in_each_cluster = {}
    for i in range(n_clusters):
        images_in_each_cluster[i]=[];
        if not os.path.exists(os.path.join(params.clusters_folder_m3,str(i))):
            os.makedirs(os.path.join(params.clusters_folder_m3,str(i)))

    # Get final lists
    image_labels = clt.labels_;
    for i in range(n_clusters):
        for ind in np.where(image_labels==i)[0]:
            item = all_train_folders[ind];
            images_in_each_cluster[i].append(item)
            # Get Paths
            image_path = os.path.join(params.train_folder_org,item+'/images');
            mask_path = os.path.join(params.train_folder_org,item+'/masks');
            # Load image and masks
            image_file = os.listdir(image_path)[0];
            bgr_image = cv2.imread(os.path.join(image_path,image_file));
            bw_mask = get_bw_mask(mask_path);
            cv2.imwrite(os.path.join(params.clusters_folder_m3,str(i)+'/'+item+'.png'), bgr_image)
            cv2.imwrite(os.path.join(params.clusters_folder_m3,str(i)+'/'+item+'_mask.png'), bw_mask)
        with open(os.path.join(params.clusters_folder_m3,str(i)+'/'+'org_names.pickle'),'wb') as writeFile:
            pickle.dump(images_in_each_cluster[i], writeFile); writeFile.close();
    del i, ind, item, image_path, mask_path, image_file, bgr_image, bw_mask, writeFile

    # Save cluster details
    clustering_data = {'dominant_centers':dominant_centers,
                       'n_clusters':n_clusters,
                       'cluster_centers_':cluster_centers_,
                       'cluster_percent_':cluster_percent_,
                       'image_labels':image_labels,
                       'images_in_each_cluster':images_in_each_cluster,
                       'all_train_folders':all_train_folders,
                       'clt':clt
                       }
    with open(os.path.join(params.clusters_folder_m3,'clustering_data.pickle'), 'wb') as opfile:
        pickle.dump(clustering_data, opfile); opfile.close();


################################ BACKGROUNDS ##############################################
#for i in range(n_clusters):
#    HSV = np.array(256*[256*[cluster_centers_[i]]])
#    HSV = HSV.astype(np.uint8)
#    bgr = cv2.cvtColor(HSV, cv2.COLOR_HSV2BGR);
#    cv2.imshow('{0}'.format(i), bgr)