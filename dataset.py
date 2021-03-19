import os
import random
import cv2
import numpy as np

class Dataset(object):
    def __init__(self, cfg, mode='train'):
        self.dataset = cfg['dataset']
        self.input_size = cfg['input_size']
        self.batch_size = cfg['batch_size'] if mode == 'train' else 1
        self.mode = mode
        self.data_dirs = []

        # load data path
        if self.dataset.startswith('kitti'):
            self.dataset = self.dataset[:5]
            self.data_dirs.append(os.path.join('./datasets', self.dataset))

        elif self.dataset == 'multipie':
            self.data_dirs.append(os.path.join('./datasets', self.dataset))
        elif self.dataset == 'multipie_larger':
            self.data_dirs.append(os.path.join('./datasets', self.dataset))
        else:
            self.data_dirs.append('./datasets/multipie')
            self.data_dirs.append('./datasets/multipie_larger')
        
        self.load_filenames()

        self.image_num = len(self.image_paths) // 3
        self.shuffle_flag = True if self.mode == 'train' else False

    def load_filenames(self):
        self.image_paths = []

        if self.dataset.startswith('multipie'):
            if not self.dataset.startswith('multipie_asym'):
                for data_dir in self.data_dirs:
                    image_names = os.listdir(os.path.join(data_dir, self.mode))
                    for name in image_names:
                        self.image_paths.append(os.path.join(data_dir, self.mode, name))
                    self.image_paths.sort()

            else:
                objects = []
                image_names = os.listdir(os.path.join(self.data_dirs[0], self.mode))
                image_names.sort()
                for name in image_names:
                    if name[3:-10] not in objects:
                        objects.append(name[3:-10])

                views = ['15', '30', '45', '60', '75', '90']
                self.view_ids = {'0': ['051', ], '15': ['050', '140'], '30': ['041', '130'], 
                                 '45': ['190', '080'], '60': ['200', '090'],
                                 '75': ['010', '120'], '90': ['240', '110']}

                data_types = ['_lr', '_l', '_r']
                data_type = ''
                if self.dataset.endswith('_asym'):
                    data_type = random.choice(data_types)

                if self.dataset.endswith('_lr') or data_type == '_lr':
                    """1. left and right input views with center gt view"""
                    l_view = random.choice(views)
                    while(True):
                        r_view = random.choice(views)
                        if l_view != r_view:
                            break

                    for obj in objects:
                        self.append_paths([l_view, '0', r_view], obj, [0, 0, 1])
                        
                elif self.dataset.endswith('_l') or data_type == '_l':
                    """2. both left input views with middle gt view"""
                    l_view = random.choice(views[2:])
                    r_view = random.choice(views[:views.index(l_view)-1])
                    gt_view = random.choice(views[min(views.index(l_view), views.index(r_view))+1:\
                                                  max(views.index(l_view), views.index(r_view))])

                    for obj in objects:
                        self.append_paths([l_view, gt_view, r_view], obj, [0, 0, 0])
            
                elif self.dataset.endswith('_r') or data_type == '_r':
                    """3. both right input views with middle gt view"""
                    l_view = random.choice(views[:-2])
                    r_view = random.choice(views[views.index(l_view)+2:])
                    gt_view = random.choice(views[min(views.index(l_view), views.index(r_view))+1:\
                                                  max(views.index(l_view), views.index(r_view))])

                    for obj in objects:
                        self.append_paths([l_view, gt_view, r_view], obj, [1, 0, 1])


        elif self.dataset.startswith('kitti'):
            # from datasets.kitti import gen_list
            # gen_list.run(self.mode)
            with open(os.path.join(self.data_dirs[0], "split", self.mode + self.dataset[5:] + ".csv")) as f:
                for line in f:
                    trip = line.split(',')
                    trip[3] = trip[3][0:-1]
                    for i in range(1, 4):
                        self.image_paths.append(os.path.join(
                            self.data_dirs[0], "data", trip[0], trip[i]))
                            

    def append_paths(self, views, obj, types):
        l_view, gt_view, r_view = views
        l_type, gt_type, r_type = types
        self.image_paths.append(os.path.join(self.data_dirs[0 if int(l_view) <= 45 else 1], self.mode, 
                                "%s_%s_%s_07.png" % (l_view, obj, self.view_ids[l_view][l_type]) ) )
        self.image_paths.append(os.path.join(self.data_dirs[0], self.mode, 
                                "%s_%s_%s_07.png" % (gt_view, obj, self.view_ids[gt_view][gt_type]) ) )
        self.image_paths.append(os.path.join(self.data_dirs[0 if int(r_view) <= 45 else 1], self.mode, 
                                "%s_%s_%s_07.png" % (r_view, obj, self.view_ids[r_view][r_type]) ) )

    def shuffle(self):
        # shuffle list of data path
        inds = [i for i in range(self.image_num)]
        shuffled_inds = random.sample(inds, len(inds))

        image_paths_shuffle = []
        for i in shuffled_inds:
            for j in range(3):
                image_paths_shuffle.append(self.image_paths[i * 3 + j])
        self.image_paths = image_paths_shuffle

    def load_batch(self, batch_id):
        if batch_id == 0 and self.shuffle_flag:
            self.shuffle()
        batch = self.process(batch_id)
        return batch

    def process(self, batch_id):
        batch_sz = min(self.batch_size, self.image_num - batch_id)
        images = [[], [], []]
        for i in range(batch_id * 3, (batch_id + batch_sz) * 3, 3):
            angle = self.image_paths[i].split('/')[-1].split('_')[0]
            if not self.dataset.startswith('multipie_asym') and (angle == '45' or angle == '60' or angle == '90'):
                images[0].append(cv2.imread(self.image_paths[i + 2]))
                images[1].append(cv2.imread(self.image_paths[i + 1]))
                images[2].append(cv2.imread(self.image_paths[i]))
            else:
                images[0].append(cv2.imread(self.image_paths[i]))
                images[1].append(cv2.imread(self.image_paths[i + 2]))
                images[2].append(cv2.imread(self.image_paths[i + 1]))

            # center-crop and resize
            for j in range(3):
                img = images[j][-1]
                h, w = img.shape[:2]
                sz = min(h, w)

                if h != w:  # if is not squared
                    pad_h = int((h - sz) / 2.)
                    pad_w = int((w - sz) / 2.)
                    img = img[pad_h:-pad_h, pad_w:-pad_w]
                if sz != self.input_size:
                    img = cv2.resize(img, (self.input_size, self.input_size))
                images[j][-1] = img

        batch = (np.array(images, dtype=np.float64) - float(127.5)) / float(127.5)
        return batch