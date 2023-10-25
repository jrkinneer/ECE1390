import numpy as np
import random
import math

#1a
# norm_2d = np.loadtxt("./input/pts2d-norm-pic_a.txt", dtype="float32")
# norm_3d = np.loadtxt("./input/pts3d-norm.txt", dtype='float32')

# norm_2d = np.hstack((norm_2d, np.ones((norm_2d.shape[0], 1))))
# norm_3d = np.hstack((norm_3d, np.ones((norm_3d.shape[0], 1))))

# diag = np.zeros((2 * len(norm_3d), 12))

# for i in range(norm_3d.shape[0]):
#     diag[2 * i, :3] = norm_3d[i, :3]
#     diag[2 * i, 3] = 1
#     diag[2 * i + 1, 4:7] = norm_3d[i, :3]
#     diag[2 * i + 1, 7] = 1
#     diag[2 * i, 8:11] = -norm_3d[i, :3] * norm_2d[i][0]
#     diag[2 * i + 1, 8:11] = -norm_3d[i, :3] * norm_2d[i][1]
#     diag[2 * i, 11] = -norm_2d[i][0]
#     diag[2 * i + 1, 11] = -norm_2d[i][1]

# _, _, V = np.linalg.svd(diag)
# m = V[-1, :].reshape(3,4)
# m = m*-1

# last_3d = np.array([
#     [1.2323],
#     [1.4421],
#     [.4506],
#     [1]
# ])
# first_3d = np.array([
#     [1.5706],
#     [-.1490],
#     [.2598],
#     [1]
# ])


# last_2d = np.array([
#     [.1406],
#     [-.04527],
#     [1]
# ])
# first_2d = np.array([
#     [1.0486],
#     [-.3645],
#     [1]
# ])
# homogenous_1 = np.dot(m, last_3d)
# homogenous_2 = np.dot(m, first_3d)
# inhomo_1 = np.array([
#     homogenous_1[0]/homogenous_1[2],
#     homogenous_1[1]/homogenous_1[2]
# ])
# inhomo_2 = np.array([
#     homogenous_2[0]/homogenous_2[2],
#     homogenous_2[1]/homogenous_2[2]
# ])
# residual_1 = np.sqrt((inhomo_1[0]-last_2d[0])**2 + (inhomo_1[1]-last_2d[1])**2)
# residual_2 = np.sqrt((inhomo_2[0] - first_2d[0])**2 + (inhomo_2[1] - first_2d[1])**2)
# print("transform\n",m)
# print("<u, v> first\n", inhomo_2)
# print("<u, v> last\n", inhomo_1)
# print("residual first point\n", residual_2)
# print("residual last point\n", residual_1)

#1b
norm_2d = np.loadtxt("./input/pts2d-norm-pic_a.txt", dtype="float32")
norm_3d = np.loadtxt("./input/pts3d-norm.txt", dtype='float32')

norm_2d = np.hstack((norm_2d, np.ones((norm_2d.shape[0], 1))))
norm_3d = np.hstack((norm_3d, np.ones((norm_3d.shape[0], 1))))
k_list = [8,12,16]

m_saved = np.array([])
avg_residual = math.inf
for k in k_list:
    for i in range(10):
        #all possible indices
        indices_array = np.linspace(0, norm_2d.shape[0] - 1, norm_2d.shape[0], dtype='uint8')
        
        #k randomly chosen points
        chosen_ind = random.sample(indices_array.tolist(), k)
        
        #all indices not chosen
        not_chosen = [j for j in indices_array if j not in chosen_ind]
        
        #4 random points for testing 
        test_ind = random.sample(not_chosen, 4)
        
        
        #get matrix m
        diag = np.zeros((2 * len(chosen_ind), 12))

        for z in range(len(chosen_ind)):
            ind = chosen_ind[z]
            diag[2 * z, :3] = norm_3d[ind, :3]
            diag[2 * z, 3] = 1
            diag[2 * z + 1, 4:7] = norm_3d[ind, :3]
            diag[2 * z + 1, 7] = 1
            diag[2 * z, 8:11] = -norm_3d[ind, :3] * norm_2d[ind][0]
            diag[2 * z + 1, 8:11] = -norm_3d[ind, :3] * norm_2d[ind][1]
            diag[2 * z, 11] = -norm_2d[ind][0]
            diag[2 * z + 1, 11] = -norm_2d[ind][1]
            
        _, _, V = np.linalg.svd(diag)
        m = V[-1, :].reshape(3,4)
        m = m*-1
        
        #calculate residual
        sum_residual = 0
        for a in range(4):
            ind = test_ind[a]
            homogenous = np.dot(m, norm_3d[ind].reshape(4,1))
            inhomo = np.array([
                homogenous[0]/homogenous[2],
                homogenous[1]/homogenous[2]
            ])
            sum_residual += np.sqrt((inhomo[0] - norm_2d[ind][0])**2 + (inhomo[1] - norm_2d[ind][1])**2)
            
        curr_avg_residual = sum_residual/4
        
        #update residual and m
        if (curr_avg_residual < avg_residual):
            avg_residual = curr_avg_residual
            m_saved = m
        
        print("k = ", k, "trial = ", i, " average residual = ", curr_avg_residual)
        
    
print("lowest residual = ", avg_residual)
print("best m\n", m_saved)
        