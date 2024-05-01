import cv2
import numpy as np
import os, cv2, torch
import matplotlib.pyplot as plt
from math import pi
import os


# Performing transformations on keypoints
def warpPerspectivePoints(src_points, H):
    # normalize H
    H /= H[2][2]

    ones = np.ones((src_points.shape[0], 1))
    points = np.append(src_points, ones, axis = 1)

    warpPoints = np.dot(H, points.T)
    warpPoints = warpPoints.T / warpPoints.T[:, 2][:,None]

    return warpPoints[:,0:2]


# Calculate repeatability
def compute_repeatability_fast(k1, k2, H, pixel_threshold=3):

    warp_points = warpPerspectivePoints(k1, H) ## warp kpts1 to image2 (using GT)
    k1_to_k2_pts = warp_points

    repeatable_matrix = []
    k1_to_k2_pts = torch.tensor(k1_to_k2_pts).cuda()
    k2 = torch.tensor(k2).cuda()
    for kpt in k1_to_k2_pts:
        repeatable_matrix.append(torch.sqrt(torch.pow(kpt - k2, 2).sum(1)))
    repeatable_matrix = torch.stack(repeatable_matrix)
    is_repeatable = torch.relu(-repeatable_matrix + pixel_threshold).sum(1).bool()  ## check if any number in pixel threshold for each point
    assert len(is_repeatable) == len(k1)

    return is_repeatable.cpu().numpy().tolist()


# Obtain feature points of public areas, 
# and do not consider coordinates outside the area after transformation
def get_keypoints_within(kp_src,h,roi):
    kp_src_trans = warpPerspectivePoints(kp_src,h)
    res = [(loc[0]>roi[2] and loc[0]<roi[3] and loc[1]>roi[0] and loc[1]<roi[1]) for loc in kp_src_trans]
    return res


# Calculate R_avg
def get_repeatability(model,img_path_src,img_path_dst,h_path,offset_path,kp_src,kp_dst,output_dir):
    if(os.path.exists(output_dir+'res.png')):
        print('pass')
        return
    print(output_dir)

    # Reading keypoints
    if(model=='superpoint'):
        kp1 = np.loadtxt(kp_src,dtype=np.float32, delimiter=' ')
        kp2 = np.loadtxt(kp_dst,dtype=np.float32, delimiter=' ')
        kp1 = kp1.T
        kp2 = kp2.T
        kp1 = kp1[:,:2]
        kp2 = kp2[:,:2]
    elif(model=='keynet' or model == 'rekd'):
        kp1 = np.load(kp_src)
        kp2 = np.load(kp_dst)
        kp1 = kp1[:,:2]
        kp2 = kp2[:,:2]
    else:
        kp1 = np.loadtxt(kp_src,dtype=np.float32, delimiter=' ')
        kp2 = np.loadtxt(kp_dst,dtype=np.float32, delimiter=' ')


    # If no keypoints are extracted or there is only one
    if(kp1.size==0 or kp2.size==0):
        print('empty')
        return
    if(kp1.ndim==1):
        print('only one')
        kp1=np.array([kp1])
    if(kp2.ndim==1):
        print('only one')
        kp2=np.array([kp2])


    p_src=kp1[:,0:2]
    p_dst=kp2[:,0:2]
    # Read the homography matrix h to calculate the repeatability
    h1_2 = np.loadtxt(h_path, dtype=np.float32, delimiter=' ')
    # Add an offset and convert it to a larger image coordinate, 
    # as the transformation matrix is obtained from a larger image
    arr_offset = np.loadtxt(offset_path,dtype=np.float32, delimiter=' ')
    # Scale the coordinates of 512x512 to 620x620
    # p_src= p_src*1.2109375
    # p_dst= p_dst*1.2109375
    # Convert small image coordinates to large image coordinates
    p_src_add = p_src+[arr_offset[2],arr_offset[0]]
    p_dst_add = p_dst+[arr_offset[2],arr_offset[0]]
    # Finding duplicate points in the forward direction (whether there are matching points in the transformation map after the feature points of the original image are transformed by the transformation matrix)
    is_repeatable = compute_repeatability_fast(p_src_add, p_dst_add, h1_2) 
    is_within = get_keypoints_within(p_src_add,h1_2,arr_offset)

    # Reverse finding duplicate points
    is_repeatable2 = compute_repeatability_fast(p_dst_add, p_src_add, np.linalg.inv(h1_2)) 
    is_within2 = get_keypoints_within(p_dst_add,np.linalg.inv(h1_2),arr_offset)

    # Calculate R_avg
    points_right = kp1[is_repeatable,:]     #原图中重复的特征点
    points_right2 = kp2[is_repeatable2,:]   #变换图中重复的特征点
    np.savetxt(output_dir+'kp_repeat_src.txt',points_right)
    np.savetxt(output_dir+'kp_repeat_dst.txt',points_right2)
    rep_final = (len(points_right)/sum(is_within)+len(points_right2)/sum(is_within2))/2*100
    # The number of detected points (including those in non-public areas)
    points_count = (len(p_src_add)+len(p_dst_add))/2
    # save
    np.savetxt(output_dir+'rep.txt',[rep_final])
    np.savetxt(output_dir+'kp_num.txt',[points_count])
    # R_avg
    print(rep_final)

    # visualization
    plt.ioff()
    img1 = cv2.imread(img_path_src)
    img2 = cv2.imread(img_path_dst)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(15,8))
    # Detected points (Figure 1)
    plt.subplot(2,2,1)
    plt.imshow(img1)
    plt.scatter(x= kp1[:, 0], y=kp1[:, 1])
    # Detected points (Figure 2)
    plt.subplot(2,2,2)
    plt.imshow(img2)
    plt.scatter(x= kp2[:, 0], y=kp2[:, 1])
    # Repeated points (Figure 1)
    plt.subplot(2,2,3)
    plt.imshow(img1)
    plt.scatter(x= points_right[:, 0], y=points_right[:, 1])
    # Repeated points (Figure 2)
    plt.subplot(2,2,4)
    plt.imshow(img2)
    plt.scatter(x= points_right2[:, 0], y=points_right2[:, 1])

    plt.savefig(output_dir+'res.png')
    plt.ion()
    # plt.show()

# Start
# Read all data in the folder
path_dataset = '../dataset/'
# models = ['rekd','superpoint','keynet','sift','orb']
models = ['rekd']
# flag_find_start=False
for model in models:
    for filename in os.listdir(path_dataset):
        temp = path_dataset+filename+'/'
        for typee in os.listdir(temp):
            dir_temp = temp+typee+'/'
            # print(dir_temp)
            if(typee=='src'):
                continue

            
            img_src_path = temp+'src/'+'im_src_clip.png'
            img_dst_path = dir_temp+'im_dst_clip.png'
            h_path = dir_temp+'h.txt'
            offset_path = dir_temp+'roi.txt'
            output_dir = dir_temp+model+'_result/'
            kp_src =  temp+'src/'+model+'_result/'+'kp_src.txt'
            kp_dst = output_dir+'kp_dst.txt'

            get_repeatability(model,img_src_path,img_dst_path,h_path,offset_path,kp_src,kp_dst,output_dir)