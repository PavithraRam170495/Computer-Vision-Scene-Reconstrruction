#import packages
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import glob

seed = np.random.seed(10)

#task 1
def camera_caliberation():

    #subtask A
    checkerboard = (7,5)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    obj_3Dpoints = []
    img_2Dpoints = []

    objp = np.zeros((1, checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    prev_img_shape = None

    images = glob.glob('Assignment_MV_02_calibration/*.png')
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard )
        if ret == True:
            obj_3Dpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_2Dpoints.append(corners2)
            img = cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    #subtask B
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_3Dpoints, img_2Dpoints, gray.shape[::-1], None, None)
    return mtx



def get_tracks(filename):

    #subtask C
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    camera = cv2.VideoCapture(filename)
    # initialise features to track
    while camera.isOpened():
        ret, img = camera.read()
        if ret:
            new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            p0 = cv2.goodFeaturesToTrack(new_img, 200, 0.3, 7)
            p0 = cv2.cornerSubPix(new_img, p0, (11, 11), (-1, -1), criteria)
            break

    #subtask D
    # initialise tracks
    index = np.arange(len(p0))
    tracks = {}
    for i in range(len(p0)):
        tracks[index[i]] = {0: p0[i]}

    frame = 0
    while camera.isOpened():
        ret, img = camera.read()
        if not ret:
            break

        frame += 1

        old_img = new_img
        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # calculate optical flow
        if len(p0) > 0:
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_img, new_img, p0, None)

            # visualise points
            for i in range(len(st)):
                if st[i]:
                    cv2.circle(img, (p1[i, 0, 0], p1[i, 0, 1]), 2, (0, 0, 255), 2)
                    cv2.line(img, (p0[i, 0, 0], p0[i, 0, 1]), (
                    int(p0[i, 0, 0] + (p1[i][0, 0] - p0[i, 0, 0]) * 5), int(p0[i, 0, 1] + (p1[i][0, 1] - p0[i, 0, 1]) * 5)),
                             (0, 0, 255), 2)

            p0 = p1[st == 1].reshape(-1, 1, 2)
            index = index[st.flatten() == 1]

        # refresh features, if too many lost
        if len(p0) < 100:
            new_p0 = cv2.goodFeaturesToTrack(new_img, 200 - len(p0), 0.3, 7)
            new_p0 = cv2.cornerSubPix(new_img, new_p0, (11, 11), (-1, -1), criteria)
            for i in range(len(new_p0)):
                if np.min(np.linalg.norm((p0 - new_p0[i]).reshape(len(p0), 2), axis=1)) > 10:
                    p0 = np.append(p0, new_p0[i].reshape(-1, 1, 2), axis=0)
                    index = np.append(index, np.max(index) + 1)

        # update tracks
        for i in range(len(p0)):
            if index[i] in tracks:
                tracks[index[i]][frame] = p0[i]
            else:
                tracks[index[i]] = {frame: p0[i]}

        # visualise last frames of active tracks
        for i in range(len(index)):
            for f in range(frame - 20, frame):
                if (f in tracks[index[i]]) and (f + 1 in tracks[index[i]]):
                    cv2.line(img,
                             (tracks[index[i]][f][0, 0], tracks[index[i]][f][0, 1]),
                             (tracks[index[i]][f + 1][0, 0], tracks[index[i]][f + 1][0, 1]),
                             (0, 255, 0), 1)
        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break

        cv2.imshow("camera", img)

    camera.release()
    cv2.destroyWindow("camera")
    return tracks, frame



#task 2
def calculate_homography(tracks, frame1, frame2):

    #subtask A
    correspondences = []
    x1_x_coordinates = []
    x1_y_coordinates = []
    x2_x_coordinates = []
    x2_y_coordinates = []
    for track in tracks:
        if (frame1 in tracks[track]) and (frame2 in tracks[track]):
            x1 = [tracks[track][frame1][0, 1], tracks[track][frame1][0, 0], 1]
            x2 = [tracks[track][frame2][0, 1], tracks[track][frame2][0, 0], 1]
            correspondences.append((np.array(x1), np.array(x2)))
            x1_x_coordinates.append(x1[0])
            x1_y_coordinates.append(x1[1])
            x2_x_coordinates.append(x2[0])
            x2_y_coordinates.append(x2[1])

    #subtask B
    x1_x_mean = np.mean(x1_x_coordinates)
    x2_x_mean = np.mean(x2_x_coordinates)
    x1_x_std = np.std(x1_x_coordinates)
    x2_x_std = np.std(x2_x_coordinates)
    x1_y_mean = np.mean(x1_y_coordinates)
    x2_y_mean = np.mean(x2_y_coordinates)
    x1_y_std = np.std(x1_y_coordinates)
    x2_y_std = np.std(x2_y_coordinates)

    T1 = np.array([[1 / x1_x_std, 0, -x1_x_mean / x1_x_std],
                   [0, 1 / x1_y_std, -x1_y_mean / x1_y_std],
                   [0, 0, 1]])
    T2 = np.array([[1 / x2_x_std, 0, -x2_x_mean / x2_x_std],
                   [0, 1 / x2_y_std, -x2_y_mean / x2_y_std],
                   [0, 0, 1]])

    y1 = []
    y2 = []
    for x1,x2 in correspondences :
        y1.append(np.matmul(T1, x1))
        y2.append(np.matmul(T2, x2))

    best_sum_inliers = 1e100
    best_count_outliers = len(correspondences)+1
    best_F = np.eye(3)

    for i in range(10000):

        #subtask C
        eight_correspondences = np.random.random_integers(0,len(correspondences)-1,8)
        A = np.zeros((0, 9))
        for i in range(len(correspondences)):
            if i in eight_correspondences:
                ai = np.kron(y1[i].T, y2[i].T)
                A = np.append(A, [ai], axis=0)

        #subtask D
        U, S, V = np.linalg.svd(A)
        F = V[8, :].reshape(3, 3).T

        U, S, V = np.linalg.svd(F)
        F = np.matmul(U, np.matmul(np.diag([S[0], S[1], 0]), V))

        F = np.matmul(T2.T, np.matmul(F, T1))

        #subtask E
        cxx = np.array([[1,0,0],
                        [0,1,0],
                        [0,0,0]])
        count_outliers = 0
        sum_inliers = 0
        inliers = []
        for i in range(len(correspondences)):
            if i not in eight_correspondences:
                x1 = np.array(correspondences[i][0])
                x2 = np.array(correspondences[i][1])
                model_equation = np.matmul(x2.T, np.matmul(x1, F))
                variance = np.matmul(x2.T,np.matmul(F,np.matmul(cxx,np.matmul(F.T,x2))))
                variance += np.matmul(x1.T, np.matmul(F.T, np.matmul(cxx, np.matmul(F,x1))))
                test_statistic = np.square(model_equation)/variance

                #subtask F
                if test_statistic >6.635:
                    count_outliers += 1
                else:
                    sum_inliers += test_statistic
                    inliers.append((x1,x2))

        #subtask G
        if count_outliers < best_count_outliers:
            best_count_outliers = count_outliers
            best_sum_inliers = sum_inliers
            best_F = F
            best_inliers = inliers.copy()
        elif count_outliers == best_count_outliers:
            if best_sum_inliers < sum_inliers:
                best_count_outliers = count_outliers
                best_sum_inliers = sum_inliers
                best_F = F
                best_inliers = inliers.copy()

    print("Best count outliers : ",best_count_outliers)
    print("Best Count inliers : ",len(best_inliers))

    #subtask H
    frame = extract_frames("Assignment_MV_02_video.mp4", (0,frame1))
    for x1, x2 in correspondences:
        flag = False
        for x, y in best_inliers:
            if (x1==x).all() and (x2==y).all():
                flag =True
                break
        #outliers : green
        if not flag:
            cv2.circle(frame[frame1], (int(x1[1]),int(x1[0])), 1, (0, 255, 0), 2)
            cv2.circle(frame[frame1], (int(x2[1]), int(x2[0])), 1, (0, 255, 0), 2)
        #inliers :red
        else:
            cv2.circle(frame[frame1], (int(x1[1]), int(x1[0])), 1, (0, 0, 255), 2)
            cv2.circle(frame[frame1], (int(x2[1]), int(x2[0])), 1, (0, 0, 255), 2)
    cv2.imshow("image",frame[30])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return best_F, best_inliers


def calculate_epipoles(F):
    U,S,V = np.linalg.svd(F)
    e1 = V[2,:]

    U,S,V = np.linalg.svd(F.T)
    e2 = V[2,:]

    return e1,e2



def extract_frames(filename, frames):
    result = {}
    camera = cv2.VideoCapture(filename)
    last_frame = max(frames)
    frame=0
    while camera.isOpened():
        ret,img= camera.read()
        if not ret:
            break
        if frame in frames:
            result[frame] = img
        frame += 1
        if frame>last_frame:
            break

    return result

#task 3
def essential_matrix(F,K,frames,best_inliers):

    #subtask A
    E = np.matmul(K.T, np.matmul(F, K))
    U, S, V = np.linalg.svd(E)
    S = (S[0]+S[1])/2
    E = np.matmul(U, np.matmul(np.diag([S,S, 0]), V.T))
    print("Essential matrix : ",E)
    if np.linalg.det(U) <0:
        U[:,2] *= -1
    if np.linalg.det(V) <0:
        V[2,:] *= -1
    print("Rotation matrix determinant : ", np.linalg.det(np.matmul(V,U.T)))

    #subtask B
    W = np.array([[0,-1,0],
                 [1,0,0],
                 [0,0,1]])
    Z = np.array([[0,1,0],

                  [-1,0,0],
                 [0,0,0]])
    R1 = np.matmul(U,np.matmul(W,V.T))
    R2 = np.matmul(U,np.matmul(W.T, V.T))
    beta = (50 * 1000 / (60 * 60)) * (frames / 30)
    skew1 = beta*(U@(Z@U.T))
    skew2 = -(beta*np.matmul(U,np.matmul(Z,U.T)))
    R_t1 = np.array([skew1[2][1],skew1[0][2],skew1[1][0]])
    t1 = np.matmul(np.linalg.inv(R1),R_t1)
    R_t2 = np.array([skew2[2][1], skew2[0][2], skew2[1][0]])
    t2 = np.matmul(np.linalg.inv(R2), R_t2)

    print("Rotation Matrix 1 :\n",R1)
    print("Translation Vector: ",t1)
    print("Rotation Matrix 1 :\n", R1)
    print("Translation Vector: ",t2)
    print("Rotation Matrix 2 :\n", R2)
    print("Translation Vector: ", t1)
    print("Rotation Matrix 2 :\n", R2)
    print("Translation Vector: ", t2)

    #subtask C
    threed_points_R1_t1 = []
    threed_points_R1_t2 = []
    threed_points_R2_t1 = []
    threed_points_R2_t2 = []
    original_inliers_R1_t1 = []
    original_inliers_R1_t2 = []
    original_inliers_R2_t1 = []
    original_inliers_R2_t2 = []
    for x1,x2 in best_inliers:
        m1 = np.matmul(np.linalg.inv(K),x1)
        m2 = np.matmul(np.linalg.inv(K),x2)

        #R1- t1
        lhs = np.array([[np.matmul(m1.T,m1), -np.matmul(m1.T,np.matmul(R1, m2))],
                       [np.matmul(m1.T, np.matmul(R1, m2)), -np.matmul(m2.T, m2)]])
        inverse_lhs = np.linalg.inv(lhs)
        lambda_mu = np.array([[np.matmul(t1.T,m1)],
                             [np.matmul(t1.T, np.matmul(R1, m2))]])
        R1_t1 = np.matmul(inverse_lhs,lambda_mu)


        R1_t1_x_lambda = R1_t1[0][0] * m1
        R1_t1_x_mu = t1 + R1_t1[1][0] * np.matmul(R1, m2)

        # R2- t1
        lhs = np.array([[np.matmul(m1.T, m1), -np.matmul(m1.T, np.matmul(R2, m2))],
                        [np.matmul(m1.T, np.matmul(R2, m2)), -np.matmul(m2.T, m2)]])
        inverse_lhs = np.linalg.inv(lhs)
        lambda_mu = np.array([[np.matmul(t1.T, m1)],
                              [np.matmul(t1.T, np.matmul(R2, m2))]])
        R2_t1 = np.matmul(inverse_lhs, lambda_mu)

        R2_t1_x_lambda = R2_t1[0][0] * m1
        R2_t1_x_mu = t1 + R2_t1[1][0] * np.matmul(R2, m2)

        # R1- t2
        lhs = np.array([[np.matmul(m1.T, m1), -np.matmul(m1.T, np.matmul(R1, m2))],
                        [np.matmul(m1.T, np.matmul(R1, m2)), -np.matmul(m2.T, m2)]])
        inverse_lhs = np.linalg.inv(lhs)
        lambda_mu = np.array([[np.matmul(t2.T, m1)],
                              [np.matmul(t2.T, np.matmul(R1, m2))]])
        R1_t2 = np.matmul(inverse_lhs, lambda_mu)

        R1_t2_x_lambda = R1_t2[0][0] * m1
        R1_t2_x_mu = t2 + R1_t2[1][0] * np.matmul(R1, m2)

        # R2- t2
        lhs = np.array([[np.matmul(m1.T, m1), -np.matmul(m1.T, np.matmul(R2, m2))],
                        [np.matmul(m1.T, np.matmul(R2, m2)), -np.matmul(m2.T, m2)]])
        inverse_lhs = np.linalg.inv(lhs)
        lambda_mu = np.array([[np.matmul(t2.T, m1)],
                              [np.matmul(t2.T, np.matmul(R2, m2))]])
        R2_t2 = np.matmul(inverse_lhs, lambda_mu)

        R2_t2_x_lambda = R2_t2[0][0] * m1
        R2_t2_x_mu = t2 + R2_t2[1][0] * np.matmul(R2, m2)

        if R1_t1[0][0] > 0 and R1_t1[1][0] >0:
            threed_points_R1_t1.append((R1_t1_x_lambda, R1_t1_x_mu))
            original_inliers_R1_t1.append((x1,x2))
        if R1_t2[0][0] > 0 and R1_t2[1][0] >0:
            threed_points_R1_t2.append((R1_t2_x_lambda, R1_t2_x_mu))
            original_inliers_R1_t2.append((x1, x2))
        if R2_t1[0][0] > 0 and R2_t1[1][0] >0:
            threed_points_R2_t1.append((R2_t1_x_lambda, R2_t1_x_mu))
            original_inliers_R2_t1.append((x1, x2))
        if R2_t2[0][0] > 0 and R2_t2[1][0] >0:
            threed_points_R2_t2.append((R2_t2_x_lambda, R2_t2_x_mu))
            original_inliers_R2_t2.append((x1, x2))

    true_inliers = []
    original_inliers = []
    if len(threed_points_R1_t1) > len(true_inliers):
        true_inliers = threed_points_R1_t1.copy()
        original_inliers = original_inliers_R1_t1.copy()
    if len(threed_points_R1_t2) > len(true_inliers):
        true_inliers = threed_points_R1_t2.copy()
        original_inliers = original_inliers_R1_t2.copy()
    if len(threed_points_R2_t1) > len(true_inliers):
        true_inliers = threed_points_R2_t1.copy()
        original_inliers = original_inliers_R2_t1.copy()
    if len(threed_points_R2_t2) > len(true_inliers):
        true_inliers = threed_points_R2_t2.copy()
        original_inliers = original_inliers_R2_t2.copy()

    #subtask D
    x = []
    y = []
    z = []
    for i,j in true_inliers:
        x.append((i[0]+j[0])/2)
        y.append((i[1]+j[1])/2)
        z.append((i[2]+j[2])/2)



    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(x, y, z, c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()

    #subtask E
    frames_video = extract_frames("Assignment_MV_02_video.mp4", (0,frames))
    first_frame = frames_video[frames]

    for i in range(len(true_inliers)):
        m1 = true_inliers[i][0]
        m2 = true_inliers[i][1]
        m1_ = original_inliers[i][0]
        m2_ = original_inliers[i][1]


        res = np.matmul(K,m1)/m1[2]
        x,y =  int(res[0]),int(res[1])
        cv2.circle(first_frame,(y,x),1, (0, 0, 255), 2)
        res = np.matmul(K, m2) / m2[2]
        x1,y1 = int(res[0]),int(res[1])
        cv2.circle(first_frame,(int(y1),int(x1)) , 1, (0, 255, 0), 2)

        x_,y_ =  int(m1_[0]),int(m1_[1])
        cv2.circle(first_frame,(y_,x_),1, (0, 0, 255), 2)
        x1_,y1_ =  int(m2_[0]),int(m2_[1])
        cv2.circle(first_frame,(y1_,x1_) , 1, (0, 255, 0), 2)

        cv2.line(first_frame, (y,x), (y_,x_), (255,0,0),1)
        cv2.line(first_frame, (y1, x1), (y1_, x1_), (255, 0, 0), 1)


    cv2.imshow("Resulting Image", first_frame)




    cv2.waitKey(0)
    cv2.destroyAllWindows()






def main():
    K = camera_caliberation()
    print("K \n:",K)
    tracks, frames = get_tracks("Assignment_MV_02_video.mp4")
    F , best_inliers= calculate_homography(tracks, int(frames), 0)
    print("F :", F)
    print("Funadamental matrix determinant :",np.linalg.det(F))
    e1, e2 = calculate_epipoles(F)

    print("e1 : " ,e1,"\ne2 : " ,e2)
    essential_matrix(F,K,int(frames), best_inliers)


main()
