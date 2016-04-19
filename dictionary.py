import numpy as np
import pickle
import os
import cv2
from sklearn.cluster import KMeans
from sklearn import svm
from subprocess import call
import random
import time
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def cal_sift(in_img):
    img = cv2.imread(in_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)

    # result = img
    # cv2.drawKeypoints(gray, kp, result, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("result", result)
    # k = cv2.waitKey()

    return kp, des


# def sift_match_test():
#     img1, kp1, des1 = cal_sift("/mnt/scratch/tongjiang/_car_frames/02200923_0014-3507.png")
#     img2, kp2, des2 = cal_sift("/mnt/scratch/tongjiang/_car_frames/02200923_0014-3615.png")
#
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#
#     bf = cv2.BFMatcher()
#     matches = bf.knnMatch(des1,des2, k=2)
#
#     good = []
#     for m,n in matches:
#         if m.distance < 0.75*n.distance:
#             good.append([m])
#     print good[0]
#
#     img3 = img1
#     img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,img3,flags=2)
#
#     cv2.imshow("window", img3)
#     k = cv2.waitKey()


def pickle_key_points(key_points):
    key_points_array = []
    for point in key_points:
        temp = (point.pt, point.size, point.angle, point.response, point.octave,point.class_id)
        key_points_array.append(temp)
    return key_points_array


def unpickle_keypoints(key_points_array):
    key_points = []
    for point in key_points_array:
        temp_feature = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3],
                                    _octave=point[4], _class_id=point[5])
        key_points.append(temp_feature)
    return key_points


def sift_sample(frames_dir):
    key_points = np.array([])
    descriptors = np.array([])
    index = []
    i = 0
    for root, dirs, files in os.walk(frames_dir):
        for file_path in files:
            frame_file = os.path.join(root, file_path)
            kp, des = cal_sift(frame_file)
            if len(key_points):
                key_points = np.append(key_points, kp)
                descriptors = np.append(descriptors, des, axis=0)
                index = index + [frame_file] * len(kp)
            else:
                key_points = kp
                descriptors = des
                index = [frame_file] * len(kp)
            i += 1
            if i % 99 == 0:
                print i+1, "sampled"
    return pickle_key_points(key_points), descriptors, index


def descriptor_cluster(descriptors, cluster_num, iter_num, result_dir):
    cluster = KMeans(n_clusters=cluster_num, max_iter=iter_num).fit(descriptors)
    print cluster.inertia_
    pickle.dump(cluster, open(os.path.join(result_dir, str(cluster_num)) + '.p', 'wb'))


def show_descriptor_cluster(cluster, cluster_num, key_points, descriptors, index, visual_dir):
    result_dir = os.path.join(visual_dir, str(cluster_num))
    if os.path.isdir(result_dir):
        call(['rm', '-r', result_dir])
        call(['mkdir', result_dir])
    else:
        call(['mkdir', result_dir])
    for i in range(cluster_num):
        cluster_dir = os.path.join(result_dir, str(i))
        call(['mkdir', cluster_dir])
    result = cluster.predict(descriptors)
    '''
    for i in range(cluster_num):
        print i, ":", list(result).count(i)
    print
    '''
    i = 0
    for i in range(50000):
        cur = random.randint(0, len(result) - 1)
        img = cv2.imread(index[cur])
        height, width, channels = img.shape
        kp = [key_points[cur]]
        point = key_points[cur]
        drawn = img
        cv2.drawKeypoints(img, kp, drawn, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        # cv2.drawKeypoints(img, kp, drawn, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG)
        size = 20
        left = (int(point.pt[0]) - size if (int(point.pt[0]) - size >= 0) else 0)
        right = (int(point.pt[0]) + size if (int(point.pt[0]) + size <  width) else width - 1)
        up = (int(point.pt[1]) - size if (int(point.pt[1]) - size >= 0) else 0)
        down = (int(point.pt[1]) + size if (int(point.pt[1]) + size < height) else height - 1)
        drawn = drawn[up:down, left:right, :]
        cv2.imwrite(os.path.join(visual_dir, str(cluster_num), str(result[cur]), str(cur) + ".png"), drawn)
        i += 1
        if i % 100 == 0:
            print i, "sampled"


def clip_to_feature(in_clip, cluster, cluster_num):
    sift = cv2.xfeatures2d.SIFT_create()
    feature = [0] * cluster_num
    for cur_frame in in_clip:
        kp, des = sift.detectAndCompute(cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY), None)
        cluster_result = cluster.predict(des)
        for c in cluster_result:
            feature[c] += 1
    return np.array(feature)


def video_to_clip(video, max_img_size=None, save=False, clips_dir=''):

    def shrink_img(img_mat, max_size):
        img_h, img_w = img_mat.shape[:2]
        ratio = min(float(max_size) / max(img_h, img_w), 1)
        new_h = int(img_h * ratio)
        new_w = int(img_w * ratio)
        return cv2.resize(img_mat, (new_w, new_h))


    try:
        cap = cv2.VideoCapture(video)
    except:
        print 'Cannot read file'
        exit()
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = []
    for i in range(video_len):
        _, cur_frame = cap.read()
        if max_img_size:
            cur_frame = shrink_img(cur_frame, max_img_size)
        result.append(cur_frame)

    result = np.array(result)
    filename = video.split('/')[-1].split('.')[0]
    if save and not (os.path.isfile(os.path.join(clips_dir, filename) + '.npz')):
        np.savez(os.path.join(clips_dir, filename), clip=np.array(result))
        filename = os.path.join(clips_dir, filename) + '.npz'
    else:
        filename = None
    return result , filename


def generate_clips(video_dir, clips_dir):
    for root, dirs, files in os.walk(video_dir):
        for file_path in files:
            clip_file = os.path.join(root, file_path)
            if clip_file.split('.')[-1] != 'mp4':
                continue
            video_to_clip(clip_file, None, True, clips_dir)
            print clip_file


def generate_features(clips_dir, features_dir, cluster, cluster_num):
    for root, dirs, files in os.walk(clips_dir):
        for file_path in files:
            clip_file = os.path.join(root, file_path)
            if clip_file.split('.')[-1] != 'npz':
                continue
            clip = np.load(clip_file)['clip']
            feature = clip_to_feature(clip, cluster, cluster_num)
            filename = clip_file.split('/')[-1].split('.')[0]
            np.savez(os.path.join(features_dir, str(cluster_num), filename), feature=np.array(feature))
            print clip_file


def generate_list_file(video_dir, label):
    label_video_dir = os.path.join(video_dir, label)
    list_file = open(video_dir + '/' + label + '.txt', 'wb')
    for root, dirs, files in os.walk(label_video_dir):
        for file_path in files:
            if file_path.split('.')[-1] != 'mp4':
                continue
            list_file.write(file_path.split('.')[-2] + '\n')
    list_file.close()


def generate_list(list_file):
    video_list = []
    for l in open(list_file):
        video_list.append(l)
    return video_list


def train_test_svm(features_dir, label1_list, label2_list):
    features = []
    label =[]
    index = []
    for l in label1_list:
        f = np.load(os.path.join(features_dir, l[:-1] + ".npz"))['feature']
        features.append(f)
        label.append(1)
        index.append(l[:-1])
    for l in label2_list:
        f = np.load(os.path.join(features_dir, l[:-1] + ".npz"))['feature']
        features.append(f)
        label.append(0)
        index.append(l[:-1])

    features = np.array(features)
    row_sums = features.sum(axis=1, dtype=np.float64)
    features = features / row_sums[:, np.newaxis]

    index_shuf = range(len(features))
    random.shuffle(index_shuf)
    x = []
    y = []
    tmp = []
    for i in index_shuf:
        x.append(features[i])
        y.append(label[i])
        tmp.append(index[i])
    index = tmp

    # clf = svm.SVC()
    clf = RandomForestClassifier()
    train_x = x[:len(x)*9/10]
    train_y = y[:len(y)*9/10]
    test_x = x[len(x)*9/10:]
    test_y = y[len(y)*9/10:]
    index = index[len(index)*9/10:]

    clf.fit(train_x, train_y)
    result = clf.predict(test_x)
    error = sum([int(result[i] != test_y[i]) for i in range(len(result))])
    print "Random Forest Error rate: ", float(error)/len(result)

    error = []
    for i in range(len(result)):
        if result[i] != test_y[i]:
            error.append(index[i])
    return clf, error


def video_to_scene(video, clfs, cluster, cluster_num):

    def shrink_img(img_mat, max_size):
        img_h, img_w = img_mat.shape[:2]
        ratio = min(float(max_size) / max(img_h, img_w), 1)
        new_h = int(img_h * ratio)
        new_w = int(img_w * ratio)
        return cv2.resize(img_mat, (new_w, new_h))

    try:
        cap = cv2.VideoCapture(video)
    except:
        print 'Cannot read file'
        exit()

    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = []
    scene = []

    for i in range(video_len):
        _, cur_frame = cap.read()
        result.append(cur_frame)

        cv2.imshow('frame', cur_frame)
        k = cv2.waitKey(30) & 0xff

        if len(result) == 72:
            features = np.array([clip_to_feature(result, cluster, cluster_num)])
            row_sums = features.sum(axis=1, dtype=np.float64)
            features = features / row_sums[:, np.newaxis]
            f = [clf.predict(features)[0] for clf in clfs] * 3
            scene.append(f)
            result = []
            if len(scene) == 1:
                print 0, 0
            else:
                print (len(scene) - 1) * 3, int(scene[-1] != scene[-2])

    change = []
    for i in range(len(scene)):
        if len(change) == 0:
            change.append(0)
        else:
            change.append(int(scene[i] != scene[i-1]))
    return change

    cv2.destroyAllWindows()

# def train_test_kmeans(features_dir, label1_list, label2_list):
#     features = []
#     label = []
#     for l in label1_list:
#         f = np.load(os.path.join(features_dir, l[:-1] + ".npz"))['feature']
#         features.append(f)
#         label.append(1)
#     for l in label2_list:
#         f = np.load(os.path.join(features_dir, l[:-1] + ".npz"))['feature']
#         features.append(f)
#         label.append(0)
#
#     index_shuf = range(len(features))
#     random.shuffle(index_shuf)
#     x = []
#     y = []
#     for i in index_shuf:
#         x.append(features[i])
#         y.append(label[i])
#
#     cluster = KMeans(n_clusters=2, max_iter=300).fit(x)
#     result = cluster.predict(x)
#     error = sum([int(result[i] != y[i]) for i in range(len(result))])
#     print "Kmeans Error rate: ", min(float(error) / len(result), 1 - float(error) / len(result))


def show_error(clips_dir, error):
    for e in error:
        clip = np.load(clips_dir + '/' + str(e) + '.npz')['clip']
        for frame in clip:
            cv2.imshow('frame', frame)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        cv2.destroyWindow('frame')

    k = cv2.waitKey()
    time.sleep(3)


start_time = time.time()

max_iter = 500
cluster_num = 200
kmeans_sample = 50000

# print "sampling sift key points ..."
# key_points, descriptors, index = sift_sample("/mnt/scratch/tongjiang/_car_frames")
# print descriptors.shape, len(key_points), len(index)

# np.savez("/mnt/scratch/tongjiang/bag_of_words/key_points", key_points=np.array(key_points))
# np.savez("/mnt/scratch/tongjiang/bag_of_words/descriptors", descriptors=np.array(descriptors))
# np.savez("/mnt/scratch/tongjiang/bag_of_words/index", index=np.array(index))
# print "done"
# print


# print "clustering sift descriptors ..."
# descriptors = np.load("/mnt/scratch/tongjiang/bag_of_words/descriptors.npz")['descriptors']
# np.random.shuffle(descriptors)
# descriptors = descriptors[:kmeans_sample]
# descriptor_cluster(descriptors, cluster_num, max_iter, "/mnt/scratch/tongjiang/bag_of_words")
# print "done"
# print


# print "visualizing the results ..."
# cluster = pickle.load(open('/mnt/scratch/tongjiang/bag_of_words/' + str(cluster_num) + '.p', 'rb'))
# key_points = unpickle_keypoints( np.load('/mnt/scratch/tongjiang/bag_of_words/key_points.npz')['key_points'])
# descriptors = np.load('/mnt/scratch/tongjiang/bag_of_words/descriptors.npz')['descriptors']
# index = np.load('/mnt/scratch/tongjiang/bag_of_words/index.npz')['index']
# show_descriptor_cluster(cluster, cluster_num, key_points, descriptors, index,
#                         '/mnt/scratch/tongjiang/bag_of_words/visualization')
# print "done"
# print

# generate_list_file('/mnt/scratch/tongjiang/scene_label/', 'day')
# generate_list_file('/mnt/scratch/tongjiang/scene_label/', 'night')
# generate_list_file('/mnt/scratch/tongjiang/scene_label/', 'underground')
# generate_list_file('/mnt/scratch/tongjiang/scene_label/', 'onground')
# generate_list_file('/mnt/scratch/tongjiang/scene_label/', 'openwide')
# generate_list_file('/mnt/scratch/tongjiang/scene_label/', 'local')


# print
# print "generating features ..."
# cluster = pickle.load(open('/mnt/scratch/tongjiang/bag_of_words/' + str(cluster_num) + '.p', 'rb'))
# generate_features('/mnt/scratch/tongjiang/scene_label/clips',
#                   '/mnt/scratch/tongjiang/scene_label/features', cluster, cluster_num)
# print "done"


print
print "day/night"
clf, error = train_test_svm("/mnt/scratch/tongjiang/scene_label/features/1000",
               generate_list('/mnt/scratch/tongjiang/scene_label/day.txt'),
               generate_list('/mnt/scratch/tongjiang/scene_label/night.txt'))
pickle.dump(clf, open('/mnt/scratch/tongjiang/scene_classifier/day.p', 'wb'))

print
print "onground/underground"
clf, error = train_test_svm("/mnt/scratch/tongjiang/scene_label/features/1000",
               generate_list('/mnt/scratch/tongjiang/scene_label/onground.txt'),
               generate_list('/mnt/scratch/tongjiang/scene_label/underground.txt'))
pickle.dump(clf, open('/mnt/scratch/tongjiang/scene_classifier/ground.p', 'wb'))

print
print "openwide/local"
clf, error = train_test_svm("/mnt/scratch/tongjiang/scene_label/features/1000",
               generate_list('/mnt/scratch/tongjiang/scene_label/openwide.txt'),
               generate_list('/mnt/scratch/tongjiang/scene_label/local.txt'))
pickle.dump(clf, open('/mnt/scratch/tongjiang/scene_classifier/openwide.p', 'wb'))

# print error
# show_error('/mnt/scratch/tongjiang/scene_label/clips', error)

# for l in open("/mnt/scratch/tongjiang/scene_classifier/test_file_path.txt"):
#     if len(l) <= 1:
#         break
#     print l[:-1]
#     video_to_clip(l[:-1], None, True, "/mnt/scratch/tongjiang/scene_classifier/test_clips")
#

# day = pickle.load(open('/mnt/scratch/tongjiang/scene_classifier/day.p', 'rb'))
# ground = pickle.load(open('/mnt/scratch/tongjiang/scene_classifier/ground.p', 'rb'))
# openwide = pickle.load(open('/mnt/scratch/tongjiang/scene_classifier/openwide.p', 'rb'))
# cluster = pickle.load(open('/mnt/scratch/tongjiang/bag_of_words/' + str(cluster_num) + '.p', 'rb'))
#
#
# for l in open("/mnt/scratch/tongjiang/scene_classifier/test_file_path.txt"):
#     if len(l) <= 1:
#         break
#     print l
#     change = video_to_scene(l[:-1], [day, ground, openwide], cluster, cluster_num)
#     print change
#     plt.plot(change)
#     plt.show()


print
print("--- %s seconds ---" % (time.time() - start_time))