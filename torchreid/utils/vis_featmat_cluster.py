import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import torch
# from PIL import Image
import os
import errno
from matplotlib import pyplot as plt

def mkdir_if_missing(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def euclidean_distance(inputs):
    n = inputs.size(0)
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, inputs, inputs.t())
    dist = dist.clamp(min=0).sqrt()
    return dist

def vis_featmat_DBSCAN(feat_map, top_per=0.05, save_path='/home/hh/tmp/featmap_DBSCAN'):
    H,W = feat_map.size(2), feat_map.size(3)
    mkdir_if_missing(save_path)
    for i in range(feat_map.size(0)):
        feat = feat_map[i].view(feat_map.size(1), -1).transpose(0,1)
        dist_mat = euclidean_distance(feat).data.cpu().numpy()
        tri_mat = np.triu(dist_mat, 1)  # tri_mat.dim=2
        tri_mat = tri_mat[np.nonzero(tri_mat)]  # tri_mat.dim=1
        tri_mat = np.sort(tri_mat, axis=None)
        top_num = np.round(top_per*tri_mat.size).astype(int)
        eps = tri_mat[:top_num].mean()
        cluster = DBSCAN(eps=eps, min_samples=4, metric='precomputed', n_jobs=8)
        labels = cluster.fit_predict(dist_mat)
        labels = labels.reshape(H,W)
        # labels = labels.reshape(H,W).astype(float)
        # labels = (labels/labels.max())*255
        # im = Image.fromarray(labels).convert('L')
        # im.save(os.path.join(save_path,'Dbscan_{}.jpg'.format(i)))
        plt.imshow(labels, cmap=plt.cm.hot_r)
        fig = plt.gcf()
        fig.savefig(os.path.join(save_path,'Dbscan_{}.jpg'.format(i)))

def vis_featmat_Kmeans(feat_map, num_cluster=4, save_path='/home/hh/tmp/featmap_kmeans'):
    H,W = feat_map.size(2), feat_map.size(3)
    mkdir_if_missing(save_path)
    for i in range(feat_map.size(0)):
        feat = feat_map[i].view(feat_map.size(1), -1).transpose(0,1)
        dist_mat = euclidean_distance(feat).data.cpu().numpy()
        cluster = KMeans(n_clusters=num_cluster)
        labels = cluster.fit_predict(dist_mat)
        labels = labels.reshape(H,W)
        # labels = labels.reshape(H,W).astype(float)
        # labels = (labels/labels.max())*255
        # im = Image.fromarray(labels).convert('L')
        # im.save(os.path.join(save_path,'kmeans_{}.jpg'.format(i)))
        plt.imshow(labels, cmap=plt.cm.hot_r)
        fig = plt.gcf()
        fig.savefig(os.path.join(save_path,'kmeans_{}_{}.jpg'.format(i,num_cluster)))