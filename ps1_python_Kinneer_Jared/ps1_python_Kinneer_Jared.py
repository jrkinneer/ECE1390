import numpy as np
import cv2
import random
import scipy

def kmeans_single(X, K, iters):
    m, n = X.shape
    # x-> [m,n]
        #m is number of pixels
        #n is the dimension of the pixel value
            #if color n = 3 (RGB), if BW n = 1
    
    # K number of clusters
    # iters, number of times calculating C_ik and cluster groups
    
    #ids -> [m, 1] 
        #m is number of pixels
        #1 is the cluster id of said pixel
    ids = np.empty([m,1])
    
    #means -> [K, n]
        #K cluster rows
        #n is the dimensional vector of the averages of the pixels in the cluster
    means = np.empty([K, n])
    
    #ssd -> scalar value
        #sum of squared distances between all points and their assigned means
    ssd = 0
    
    #find the range of values in m
    #min_m is of size n, holds mins of all measurements of n_i
    min_m = np.min(X, axis=0)
    max_m = np.max(X, axis=0)
    
    #randomly generate [K, n] values in range of min-max of m for means
    for y in range(K):
        for z in range(n):
            means[y][z] = random.randint(int(min_m[z]), int(max_m[z]))
    
    #loop over in range iters
    for y in range(iters):
        
        #finds distances between all m points and current K means
        distances = scipy.spatial.distance.cdist(X, means)
        for z in range(m):
            #find minimum distance of each point and attach it to the ids list
            closest_cluster_idx = np.argmin(distances[z])
            ids[z][0] = closest_cluster_idx
          
        #calculate new average  
        sum = np.zeros([K,n])
        points_in_cluster = [0] * K
        for v in range(m):
            current_cluster = int(ids[v][0])
            points_in_cluster[current_cluster] += 1
            for a in range(n):
                sum[current_cluster][a] += X[v][a] 
            
        for i in range(K):
            for j in range(n):
                means[i][j] = sum[i][j]/points_in_cluster[i]
        
    #calculate ssd
    for i in range(m):
        cluster = int(ids[v][0])
        
        distance = X[i] - means[cluster]
        ssd += distance**2
    return (ids, means, ssd)


def kmeans_multiple(X, K, iters, R):
    m, n = X.shape
    
    ids = np.empty([m,1])
    
    means = np.empty([K, n])
    
    ssd = 0
    
    #find the range of values in m
    #min_m is of size n, holds mins of all measurements of n_i
    min_m = np.min(X, axis=0)
    max_m = np.max(X, axis=0)
    
    #find best random means to start
    best_mean = np.empty([K, n])
    current_mean = np.empty([K, n])
    
    start_ssd = 0
    current_ssd = 0
    for y in range(K):
        for z in range(n):
            current_mean[y][z] = random.randint(int(min_m[z]), int(max_m[z]))
    best_mean = current_mean
    
    #finds distances between all m points and current K means
    distances = scipy.spatial.distance.cdist(X, current_mean)
    for z in range(m):
        #find minimum distance of each point and attach it to the ids list
        closest_cluster_idx = np.argmin(distances[z])
        ids[z][0] = closest_cluster_idx
    #calculate ssd
    for i in range(m):
        cluster = int(ids[i][0])
        distance = X[i] - current_mean[cluster]
        start_ssd += distance**2
    
    #do the above R times and figure out lowest ssd and use that for iters loops
    for r in range(R):
        current_mean = np.empty([K, n])
        current_ssd = 0
        for y in range(K):
            for z in range(n):
                current_mean[y][z] = random.randint(int(min_m[z]), int(max_m[z]))
        
        #finds distances between all m points and current K means
        distances = scipy.spatial.distance.cdist(X, current_mean)
        for z in range(m):
            #find minimum distance of each point and attach it to the ids list
            closest_cluster_idx = np.argmin(distances[z])
            ids[z][0] = closest_cluster_idx
        #calculate ssd
        for i in range(m):
            cluster = int(ids[i][0])
            distance = X[i] - current_mean[cluster]
            current_ssd += distance**2
            
        
        if (np.sum(current_ssd) < np.sum(start_ssd)):
            best_mean = current_mean
            start_ssd = current_ssd
        
    means = best_mean

    #loop over in range iters
    for y in range(iters):
        
        #finds distances between all m points and current K means
        distances = scipy.spatial.distance.cdist(X, means)
        for z in range(m):
            #find minimum distance of each point and attach it to the ids list
            closest_cluster_idx = np.argmin(distances[z])
            ids[z][0] = closest_cluster_idx
          
        #calculate new average  
        sum = np.zeros([K,n])
        points_in_cluster = [0] * K
        for v in range(m):
            current_cluster = int(ids[v][0])
            points_in_cluster[current_cluster] += 1
            for a in range(n):
                sum[current_cluster][a] += X[v][a] 
            
        for i in range(K):
            for j in range(n):
                means[i][j] = sum[i][j]/points_in_cluster[i]
        
    #calculate ssd
    for i in range(m):
        cluster = int(ids[i][0])
        
        distance = X[i] - means[cluster]
        ssd += distance**2
        
    return (ids, means, ssd)


test_X = np.array([[7,8,9],[1,2,3], [4,4,4], [5,5,5], [3,3,3], [5,6,7], [3,3,3], [5,6,7]])
K = 2
iters = 10
ids, means, ssd = kmeans_multiple(test_X, K, iters, 5)
