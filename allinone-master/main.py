import h5py
import pickle
import scipy.io as sio
import os.path as osp
import os
import numpy as np
import cv2
import time
# opencv.imread() BGR mode !!

base_path = '/home/peter/extra/dataset/jm'


class DataLoader():
    def __init__(self):
        self.train_data_counter = 0
        self.test_data_counter = 0
        self.label_counter = 0
        db = self.get_dataset()

        self.trnx = db['train']['X']
        self.trny = db['train']['Y']
        self.tstx = db['test']['X']
        self.tsty = db['test']['Y']

        self.do = True
        self.sp_name = 15
        self.sp_mode = 13
        self.sp_size = 13
        self.sp_label = 13

        print('DB name'.ljust(self.sp_name), '|',
              'Mode'.ljust(self.sp_mode), '|',
              '# samples'.ljust(self.sp_size), '|',
              '# labels'.ljust(self.sp_label), '|', 'new label range')
        print('-------------------------------------------------------------------')

    def get_dataset(self):
        path = os.path.join(base_path, 'jmdb.h5')

        if os.path.exists(path):
            db = h5py.File(path, 'r+')
        else:
            db = h5py.File(path, 'w')

            train_grp = db.create_group('train')
            test_grp = db.create_group('test')
            n_test_samples = 90000
            n_train_samples = 280000

            train_grp.create_dataset('X', (n_train_samples, 64, 64, 3), dtype='uint8')
            train_grp.create_dataset('Y', (n_train_samples, 1), dtype='uint8')

            test_grp.create_dataset('X', (n_test_samples, 64, 64, 3), dtype='uint8')
            test_grp.create_dataset('Y', (n_test_samples, 1), dtype='uint8')

        return db

    def run(self):
        self.cifar10()
        self.cifar100()
        self.flower12()
        self.cuhk03()
        self.lfw()
        self.mnist()
        self.stanfoard()
        self.svhn()

    def cifar10(self):
        """
        keys: dict_keys([b'data', b'labels', b'filenames', b'batch_label'])

        data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.

        1024 R + 1024 G + 1024 B (RGB)

        :return:
        """
        path = osp.join(base_path, 'cifar10')

        def train():
            """
            read dats from every train_batch 1~5
            eacth train_batch contains 10,000 samples

            :return:
            """

            anchor = self.train_data_counter  # pointer to trainX
            for i in range(1, 6):
                fname = osp.join(path, 'data_batch_%d' % i)
                dict = pickle.load(open(fname, 'rb'), encoding='bytes')
                data = dict[b'data']  # raw shape  10000x3072
                labels = dict[b'labels']

                images = data.reshape([-1, 3, 32, 32])
                images = images.transpose([0, 2, 3, 1])

                for x, y in zip(images, labels):
                    if self.do:
                        img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                        img = cv2.resize(img, (64, 64))
                        self.trnx[self.train_data_counter] = img
                        self.trny[self.train_data_counter] = y + self.label_counter  # re-labeling

                    self.train_data_counter += 1

            self.print_stat('cifar10', anchor, self.train_data_counter, isTrain=True)

        def test():
            """
            read datas from single test_batch file.

            same as train()
            :return:
            """
            anchor = self.test_data_counter
            fname = osp.join(path, 'test_batch')
            dict = pickle.load(open(fname, 'rb'), encoding='bytes')
            data = dict[b'data']
            labels = dict[b'labels']

            images = data.reshape([-1, 3, 32, 32])
            images = images.transpose([0, 2, 3, 1])

            new_label = []
            for x, y in zip(images, labels):
                if self.do:
                    img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, (64, 64))
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = y + self.label_counter
                new_label.append(y + self.label_counter)
                self.test_data_counter += 1  # should be last

            label_size = len(np.unique(new_label))  # counts unique labels
            self.label_counter += label_size  # add to label_counter for re-labeling on next dataset
            self.print_stat_test('cifar10', anchor, self.test_data_counter, label_size, new_label)

        train()
        test()

    def cifar100(self):
        """
        keys: dict_keys([b'fine_labels', b'coarse_labels', b'filenames', b'data', b'batch_label'])

        data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
        labels -- a list of 10000 numbers in the range 0-99. The number at index i indicates the label of the ith image in the array data.

        RAW DATA IS RGB!!

        coarse_labels: super-class in the range 0-19
        fine_labels: sub-class in the range 0-99

        :return:
        """
        path = osp.join(base_path, 'cifar100')

        def train():
            anchor = self.train_data_counter
            fname = osp.join(path, 'train')
            dict = pickle.load(open(fname, 'rb'), encoding='bytes')

            data = dict[b'data']
            labels = dict[b'fine_labels']
            images = data.reshape([-1, 3, 32, 32])
            images = images.transpose([0, 2, 3, 1])

            for x, y in zip(images, labels):
                if self.do:
                    img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, (64, 64))
                    self.trnx[self.train_data_counter] = img
                    self.trny[self.train_data_counter] = y + self.label_counter
                self.train_data_counter += 1

            self.print_stat('cifar100', anchor, self.train_data_counter, isTrain=True)

        def test():
            anchor = self.test_data_counter
            fname = osp.join(path, 'test')
            dict = pickle.load(open(fname, 'rb'), encoding='bytes')
            data = dict[b'data']
            labels = dict[b'fine_labels']

            images = data.reshape([-1, 3, 32, 32])
            images = images.transpose([0, 2, 3, 1])
            new_label = []
            for x, y in zip(images, labels):
                if self.do:
                    img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, (64, 64))
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = y + self.label_counter
                new_label.append(y + self.label_counter)
                self.test_data_counter += 1

            label_size = len(np.unique(new_label))
            self.label_counter += label_size
            self.print_stat_test('cifar100', anchor, self.test_data_counter, label_size, new_label)

        train()
        test()

    def flower12(self):
        path = osp.join(base_path, 'flower')
        mat = sio.loadmat(osp.join(path, 'setid.mat'))
        mat2 = sio.loadmat(osp.join(path, 'imagelabels.mat'))
        labels = mat2['labels'][0]  # 'labels' objects is contained in the list.
        test_id = mat['tstid'][0]  # same as above
        train_id = mat['trnid'][0]  # same as above

        def test():
            anchor = self.test_data_counter  # pointer to trainX
            new_label = []  # re-labeling
            for id in test_id:
                if self.do:
                    image_name = 'image_%s.jpg' % str(id).zfill(5)  # filename starts from 1
                    img = cv2.imread(osp.join(path, 'jpg', image_name))
                    img = cv2.resize(img, (64, 64))
                    label = labels[id - 1] - 1 + self.label_counter  # org-label statrs from 1, new-label starts from 0.
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = label

                new_label.append(labels[id - 1] - 1 + self.label_counter)  # apply chagne
                self.test_data_counter += 1

            label_size = len(np.unique(labels))
            self.label_counter += label_size
            self.print_stat_test('flower12', anchor, self.test_data_counter, label_size, new_label)

        def train():
            anchor = self.train_data_counter
            for id in train_id:
                if self.do:
                    image_name = 'image_%s.jpg' % str(id).zfill(5)
                    img = cv2.imread(osp.join(path, 'jpg', image_name))
                    img = cv2.resize(img, (64, 64))
                    label = labels[id - 1] - 1 + self.label_counter  # same as above
                    self.trnx[self.train_data_counter] = img
                    self.trny[self.train_data_counter] = label
                self.train_data_counter += 1
            self.print_stat('flower12', anchor, self.train_data_counter, isTrain=True)

        train()
        test()

    def cuhk03(self):

        path = osp.join(base_path, 'cuhk03/open-reid/images')
        train_anchor = self.train_data_counter
        test_anchor = self.test_data_counter

        pid_list = []
        for f in os.listdir(path):
            '''
            read all person_id from files.
            file name format is "personid_camid_imgeid"
            '''
            raws = f.split('.')[0]
            raw = raws.split('_')
            pid = raw[0]
            pid_list.append(pid)
        pid_list = np.unique(pid_list) # pid is duplicated since each person has multiple images.

        new_label = []
        for idx, pid in enumerate(pid_list):  # unique id, person
            image_list_pid = []
            for ff in os.listdir(path):
                if pid in ff:
                    image_list_pid.append(ff)

            n = len(image_list_pid)
            anchor = int(n * 0.7)  # split images for train and test

            # print('# Total samples:', n , 'Train: 0 ~ %d'%(anchor-1), 'Test %d ~ %d'%(anchor, n-1))
            # time.sleep(1)
            train_pid_list = image_list_pid[0:anchor]
            test_pid_list = image_list_pid[anchor:]

            new_label.append(idx + self.label_counter) # re-labeling

            for img_name in train_pid_list:  # for train images
                if self.do:
                    img = cv2.imread(os.path.join(path, img_name))
                    img = cv2.resize(img, (64, 64))
                    self.trnx[self.train_data_counter] = img
                    self.trny[self.train_data_counter] = idx + self.label_counter
                self.train_data_counter += 1

            for img_name in test_pid_list:  # for test images
                if self.do:
                    img = cv2.imread(os.path.join(path, img_name))
                    img = cv2.resize(img, (64, 64))
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = idx + self.label_counter

                self.test_data_counter += 1

        label_size = len(np.unique(new_label))
        self.label_counter += label_size

        self.print_stat('cuhk03', train_anchor, self.train_data_counter, isTrain=True)
        self.print_stat_test('cuhk03', test_anchor, self.test_data_counter, label_size, new_label)

    def lfw(self):
        path = osp.join(base_path, 'lfw', 'lfw_funneled')
        person_dir_list = []

        test_anchor = self.test_data_counter
        train_anchor = self.train_data_counter
        for f in os.listdir(path):
            if os.path.isdir(osp.join(path, f)):
                if len(os.listdir(os.path.join(path, f))) > 1:
                    person_dir_list.append(f)

        print(len(os.listdir(path)))
        print(len(person_dir_list))


        new_label = []
        for idx, person_dir in enumerate(person_dir_list[30:]):
            person_img_list = os.listdir(os.path.join(path, person_dir))
            n = len(person_img_list)
            anchor = int(n * 0.7)
            # print('# Total samples:', n , 'Train: 0 ~ %d'%(anchor-1), 'Test %d ~ %d'%(anchor, n-1))
            # time.sleep(1)
            train_pid_list = person_img_list[0:anchor]
            test_pid_list = person_img_list[anchor:]
            new_label.append(idx + self.label_counter)

            for pid in train_pid_list:
                if self.do:
                    img = cv2.imread(os.path.join(path, person_dir, pid))
                    img = cv2.resize(img, (64, 64))
                    self.trnx[self.train_data_counter] = img
                    self.trny[self.train_data_counter] = idx + self.label_counter
                self.train_data_counter += 1

            for pid in test_pid_list:
                if self.do:
                    img = cv2.imread(os.path.join(path, person_dir, pid))
                    img = cv2.resize(img, (64, 64))
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = idx + self.label_counter
                self.test_data_counter += 1

        label_size = len(new_label)
        self.label_counter += label_size

        self.print_stat('lfw', train_anchor, self.train_data_counter, isTrain=True)
        self.print_stat_test('lfw', test_anchor, self.test_data_counter, label_size, new_label)

    def mnist(self):
        from mnist import MNIST
        """
        int64 >> uint8
        """
        mndata = MNIST(osp.join(base_path, 'mnist'))

        def train():
            anchor = self.train_data_counter
            images, labels = mndata.load_training()  # (60000, 784) >> (28, 28)
            images = np.array(images).reshape((-1, 28, 28))
            images = images.astype(np.uint8)

            for x, y in zip(images, labels):
                if self.do:
                    img = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(img, (64, 64))
                    self.trnx[self.train_data_counter] = img
                    self.trny[self.train_data_counter] = y + self.label_counter
                self.train_data_counter += 1

            self.print_stat('mnist', anchor, self.train_data_counter, isTrain=True)

        def test():
            anchor = self.test_data_counter
            images, labels = mndata.load_testing()  # (10000, 784) >> (28, 28)
            images = np.array(images).reshape((-1, 28, 28))
            images = images.astype(np.uint8)
            new_label = []
            for x, y in zip(images, labels):
                if self.do:
                    img = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(img, (64, 64))
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = y + self.label_counter
                new_label.append(y + self.label_counter)
                self.test_data_counter += 1

            label_size = len(np.unique(new_label))
            self.label_counter += label_size
            self.print_stat_test('mnist', anchor, self.test_data_counter, label_size, new_label)

        train()
        test()

    def stanfoard(self):
        path = osp.join(base_path, 'stanford_dog')
        img_path = os.path.join(path, 'Images')

        def test():
            testlist = sio.loadmat(osp.join(path, 'test_list.mat'))
            labels = np.squeeze(testlist['labels'])
            files = np.squeeze(testlist['file_list'])
            labels -= 1

            anchor = self.test_data_counter
            new_label = []
            for _fname, y in zip(files, labels):
                if self.do:
                    fname = _fname[0]
                    img = cv2.imread(os.path.join(img_path, fname))
                    img = cv2.resize(img, (64, 64))
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = y + self.label_counter
                self.test_data_counter += 1
                new_label.append(y + self.label_counter)

            label_size = len(np.unique(new_label))
            self.label_counter += label_size
            self.print_stat_test('stanford', anchor, self.test_data_counter, label_size, new_label)

        def train():
            trainlist = sio.loadmat(osp.join(path, 'train_list.mat'))
            labels = np.squeeze(trainlist['labels'])
            files = np.squeeze(trainlist['file_list'])
            anchor = self.train_data_counter
            labels -= 1  # label starts from 1

            for _fname, y in zip(files, labels):
                if self.do:
                    fname = _fname[0]
                    img = cv2.imread(os.path.join(img_path, fname))
                    img = cv2.resize(img, (64, 64))
                    self.trnx[self.train_data_counter] = img
                    self.trny[self.train_data_counter] = y + self.label_counter
                self.train_data_counter += 1

            self.print_stat('stanford', anchor, self.train_data_counter, isTrain=True)

        train()
        test()

    def svhn(self):  # crooped
        path = osp.join(base_path, 'svhn')

        def test():
            testmat = sio.loadmat(osp.join(path, 'test_32x32.mat'))  # (32, 32, 3, 26032)
            anchor = self.test_data_counter
            Xset = testmat['X']
            Yset = testmat['y']
            new_label = []
            for i in range(Xset.shape[3]):
                x = Xset[:, :, :, i]
                y = Yset[i]

                if self.do:
                    img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, (64, 64))
                    self.tstx[self.test_data_counter] = img
                    self.tsty[self.test_data_counter] = y + self.label_counter

                new_label.append(y + self.label_counter)
                self.test_data_counter += 1

            label_size = len(np.unique(new_label))
            self.label_counter += label_size

            self.print_stat_test('svhn', anchor, self.test_data_counter, label_size, new_label)

        def train():
            anchor = self.train_data_counter
            trainmat = sio.loadmat(osp.join(path, 'train_32x32.mat'))  # (32, 32, 3, 73257)

            Xset = trainmat['X']
            Yset = trainmat['y']

            for i in range(Xset.shape[3]):
                x = Xset[:, :, :, i]
                y = Yset[i]
                if self.do:
                    img = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
                    img = cv2.resize(img, (64, 64))
                    self.trnx[self.train_data_counter] = img
                    self.trny[self.train_data_counter] = y + self.label_counter

                self.train_data_counter += 1

            self.print_stat('svhn', anchor, self.train_data_counter, isTrain=True)

        train()
        test()

    def print_stat_test(self, dbname, before, after, label_size, yset):
        mode = '| test'
        size = after - before
        minl = np.min(yset)
        maxl = np.max(yset)

        print(dbname.ljust(self.sp_name), mode.ljust(self.sp_name), '|', str(size).ljust(self.sp_size), '|',
              str(label_size).ljust(self.sp_label), '|', minl, '~', maxl)
        print('----------------------------------------------------------------------------')

    def print_stat(self, dbname, before, after, label_size=0, isTrain=True):
        mode = '| train'
        size = after - before
        print(dbname.ljust(self.sp_name), mode.ljust(self.sp_name), '|', str(size).ljust(self.sp_size), '|',
              '-'.ljust(self.sp_label), '|')



data = DataLoader()
data.run()

print('Total train # samples: ', data.train_data_counter)
print('Total test # samples: ', data.test_data_counter)
