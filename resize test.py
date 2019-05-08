import cv2
import numpy as np
import random
import pickle

paintings = {'Vincent_van_Gogh_': 877, 'Edgar_Degas_': 702, 'Pablo_Picasso_': 439, 'Pierre-Auguste_Renoir_': 336}
labels = {'Vincent_van_Gogh_':0, 'Edgar_Degas_':1, 'Pablo_Picasso_':2, 'Pierre-Auguste_Renoir_':3}

train_rate = 0.8
def get_train():
    Xs = []
    Ys = []
    paint_nums = []

    for painter in paintings:
        end = int((paintings[painter] + 1)*train_rate)
        for n in range(1, end):
            path = './resized/resized/' + painter + str(n) + '.jpg'
            print(path)
            img = cv2.imread(path)
            # resize--128*...
            shape = img.shape
            width = shape[0]
            length = shape[1]
            multiplier = min(width, length) / 128
            pic = cv2.resize(img, (int(width / multiplier), int(length / multiplier)), interpolation=cv2.INTER_CUBIC)
            # crop--64*64
            length = pic.shape[1] // 64
            width = pic.shape[0] // 64
            print(length, width)
            for l in range(length):
                for w in range(width):
                    cropped = pic[w*64:(w+1)*64, l*64:(l+1)*64]
                    print(cropped.shape)
                    # path = './resized/resized/cropped/temp.jpg'
                    # cv2.imwrite(path, cropped)
                    # cropped = cv2.imread(path)
                    b, g, r = cv2.split(cropped)
                    print(r.shape)
                    r_p = r.reshape(1, 64*64)
                    g_p = g.reshape(1, 64*64)
                    b_p = b.reshape(1, 64*64)
                    x = np.concatenate((r_p, g_p, b_p), axis=None).reshape(1, 3*64*64)
                    Xs.append(x)
                    Ys.append(labels[painter])
                    paint_nums.append(n)

    print(len(Ys), len(Xs), len(paint_nums))
    newX = []
    newY = []
    new_paint_nums = []
    idxs = [i for i in range(len(Xs))]
    random.shuffle(idxs)
    for idx in idxs:
        newX.append(Xs[idx])
        newY.append(Ys[idx])
        new_paint_nums.append(paint_nums[idx])
    npX = np.concatenate(newX, axis=0)
    npY = np.array(newY)
    print('npX:', npX.shape)
    print('npY:', npY.shape)
    print('len new paint_numbers', len(new_paint_nums))
    print('npY:',npY)

    dic = {'labels': npY, 'data': npX, 'paint_number': new_paint_nums}
    with open('train_batch_05072005.pkl', 'wb') as f:
        pickle.dump(dic, f)


def get_test():
    Xs = []
    Ys = []
    paint_nums = []
    for painter in paintings:
        start = int((paintings[painter] + 1)*train_rate)
        for n in range(start, paintings[painter] + 1):
            crops = []
            path = './resized/resized/' + painter + str(n) + '.jpg'
            print(path)
            img = cv2.imread(path)
            # resize--128*...
            shape = img.shape
            width = shape[0]
            length = shape[1]
            multiplier = min(width, length) / 128
            pic = cv2.resize(img, (int(width / multiplier), int(length / multiplier)), interpolation=cv2.INTER_CUBIC)
            # crop--64*64
            length = pic.shape[1] // 64
            width = pic.shape[0] // 64
            print(length, width)
            for l in range(length):
                for w in range(width):
                    cropped = pic[w*64:(w+1)*64, l*64:(l+1)*64]
                    print(cropped.shape)
                    # path = './resized/resized/cropped/temp.jpg'
                    # cv2.imwrite(path, cropped)
                    # cropped = cv2.imread(path)
                    b, g, r = cv2.split(cropped)
                    print(r.shape)
                    r_p = r.reshape(1, 64*64)
                    g_p = g.reshape(1, 64*64)
                    b_p = b.reshape(1, 64*64)
                    x = np.concatenate((r_p, g_p, b_p), axis=None).reshape(1, 3*64*64)
                    crops.append(x)
            whole_pic = np.concatenate(crops, axis=None).reshape(len(crops), 3 * 64 * 64)
            Xs.append(whole_pic)
            Ys.append(labels[painter])
            paint_nums.append(n)

    print(len(Ys), len(Xs), len(paint_nums))
    dic = {'labels': Ys, 'data': Xs, 'paint_number': paint_nums}
    with open('test_batch_05072005.pkl', 'wb') as f:
        pickle.dump(dic, f)


get_test()
