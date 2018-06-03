import os
import csv
import sys
import cv2
import numpy as np
from dateutil.parser import *
from PIL import Image, ImageOps
from random import sample
from datetime import datetime, date, timedelta
from voxel_flow_train import gen_img

MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']


def crops(src, dest, ids):

    img = cv2.imread(src)
    img_ = img[82:724, 31:673]

    reflect = cv2.copyMakeBorder(img_, 63, 63, 63, 63, cv2.BORDER_REFLECT)
    assert(reflect.shape[0] == 768 and reflect.shape[1] == 768)

    img_name = os.path.basename(src)[:-4]
    a = 0  # to distinguish the 9 parts, used in filename
    names = []

    for i in range(0, 3):
        for j in range(0, 3):
            cropped_img = reflect[j*256:(j+1)*256, i*256:(i+1)*256]
            s = cropped_img.shape
            assert(s[0] == 256 and s[1] == 256)
            name = os.path.join(dest, '{0}_{1}.jpg'.format(img_name, a))
            if a in ids:
                cv2.imwrite(name, cropped_img)
                names.append(os.path.basename(name))
            a += 1
    return names


def joins(paths, dest):

    paths.sort(key=lambda x: int(x.split('_')[-1][0]))

    imgs = [cv2.imread(p) for p in paths]

    joint = np.zeros((768, 768, 3))

    a = 0
    for i in range(0, 3):
        for j in range(0, 3):
            joint[j*256:(j+1)*256, i*256:(i+1)*256] = imgs[a]
            a += 1

    img_name = os.path.basename(paths[0])
    img_path = os.path.join(dest, '_'.join(img_name.split('_')[:-1]) + '.jpg')
    # print(img_path)
    joint = joint[63:-63, 63:-63]
    cv2.imwrite(img_path, joint)


def generate(src_img1, src_img2, dest):
    import tempfile

    with tempfile.TemporaryDirectory(dir=os.path.dirname(src_img1)) as tmpdir1, \
          tempfile.TemporaryDirectory(dir=os.path.dirname(src_img2)) as tmpdir2:

        p1 = crops(src_img1, tmpdir1, range(9))
        p2 = crops(src_img2, tmpdir2, range(9))
        p1 = [os.path.join(tmpdir1, p) for p in p1]
        p2 = [os.path.join(tmpdir2, p) for p in p2]
        gen_img(p1, p2, dest)
        p3 = [os.path.join(dest, p) for p in os.listdir(dest) if 'out' in p]
        print(p3)
        joins(p3, dest)


def do_stuff(src_dir, dst_dir):

    f1, f2, f3 = [], [], []
    global MONTHS

    f1_path = os.path.join(dst_dir, 'frame1')
    f2_path = os.path.join(dst_dir, 'frame2')
    f3_path = os.path.join(dst_dir, 'frame3')
    try:
        os.makedirs(dest)
    except OSError as e:
        pass

    try:
        os.makedirs(f1_path)
        os.makedirs(f2_path)
        os.makedirs(f3_path)
    except OSError as e:
        pass

    img_datetime = []
    times = []
    for img_name in os.listdir(src_dir):
        try:
            path = os.path.join(src_dir, img_name)
            img = Image.open(path)
        except OSError as e:
            continue
        tokens = img_name.split('_')
        hour = int(tokens[2][:2])

        minute = int(tokens[2][2:])
        day = int(tokens[1][:2])
        assert(1 <= day <= 31)
        year = int(tokens[1][5:])
        assert(2017 <= year <= 2018)

        for i in range(12):
            if MONTHS[i] in img_name:
                month = i + 1
                break

        parsed_date = parse('{0}-{1}-{2} {3}:{4}'.format(year, MONTHS[month-1],
                            day, hour, minute))

        times.append([parsed_date, path])

    times.sort(key=lambda x: x[0])

    for stime1, stime2, stime3 in zip(times, times[2:], times[4:]):
        # print(stime1[0].strftime('%Y-%m-%d %H:%M'))
        # print(stime2[0].strftime('%Y-%m-%d %H:%M'))
        if (stime2[0]-stime1[0] == timedelta(minutes=60)
           and stime3[0]-stime2[0] == timedelta(minutes=60)):

            ids = sample(range(9), 6)  # only save 6 images

            f1.extend(crops(stime1[1], f1_path, ids))
            f2.extend(crops(stime2[1], f2_path, ids))
            f3.extend(crops(stime3[1], f3_path, ids))

    with open(os.path.join(dest, 'f1.txt'), 'a') as f:
        f.write('\n'.join(f1))

    with open(os.path.join(dest, 'f2.txt'), 'a') as f:
        f.write('\n'.join(f2))

    with open(os.path.join(dest, 'f3.txt'), 'a') as f:
        f.write('\n'.join(f3))


if __name__ == '__main__':
    src1, src2, dest = sys.argv[1], sys.argv[2], sys.argv[3]
    # files = os.listdir(src)
    # files = [os.path.join(src, f) for f in files]
    # joins(files, dest)
    generate(src1, src2, dest)
