import numpy as np
import h5py
import cv2


def set_background_zero(img, exclude_background):
    result = -100*np.ones(img.shape)
    sz = img.shape[:2]
    # replace the uniform background values with -100, so that we then can ignore those for further processing
    for i in range(sz[0]):
        if i%(sz[0]//20) == 0:
            print(f'{i*100/sz[0]:.2f}% is done..')
        for j in range(sz[0]):
            # around_idxs = [-1, 0, 1]
            # around_vals = []
            # for k in around_idxs:
            #     for l in around_idxs:
            #         not_around = k == 0 and l == 0
            #         negative = i+k < 0 or j+l < 0
            #         too_big = i+k >= sz[0] or j+l >= sz[1]
            #         if not_around or negative or too_big:
            #             continue
            #         around_vals.append(img[i+k, j+l])
            # around_check = np.array([(v == exclude_background).all() for v in around_vals])
            # if not around_check.any() or not (img[i,j] == exclude_background).all():
            #     result[i,j] = img[i,j]
            if not (np.isclose(img[i,j], exclude_background, atol=3)).all():
                result[i, j] = img[i, j]
    return result


def create_face_array(img_path, save_dir, exclude_background=(128, 128, 128)):
    img = cv2.imread(img_path)
    img = set_background_zero(img, exclude_background)
    img[img > 205] = 205
    non_background = img[img != -100]
    img[img != -100] -= non_background.mean()
    non_background -= non_background.mean()
    img[img != -100] /= (non_background.std()/0.7071)
    non_background /= (non_background.std()/0.7071)
    img = np.mean(img, axis=2)
    # set -100 vals to 0, so that they don't affect the pattern/signal
    img[img == -100] = 0
    # resize to 512x512 using bilinear interpolation
    img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
    # from matplotlib import pyplot as plt
    # plt.hist(non_background)
    # plt.hist(non_background[non_background>100])
    with h5py.File(save_dir, 'w') as f:
        dset = f.create_dataset(name='face_mat', data=img)


if __name__ == '__main__':
    face_path = r'C:\Users\Fabian\Documents\data\faces\face_guy_green.png'
    save_dir =  r'C:\Users\Fabian\Documents\data\faces\face_guy_green.h5'
    create_face_array(face_path, save_dir, exclude_background=(0,255,0))

