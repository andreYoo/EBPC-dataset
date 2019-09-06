
Cifar10
	- raw type
	- data shape is 3072 ( 1024 red value, 1024 blue value, 1024 green value)
	- transpose: 3,32,32 >> 32,32,3 
	- convert: RGB >> BGR
	
	
Cifar100
	- raw type(RGB)
	- data shape is 3072 ( 1024 red value, 1024 blue value, 1024 green value)
	- transpose: 3,32,32 >> 32,32,3 
	- convert: RGB >> BGR
	

Flower102
	- jpg type(BGR)
	- label starts from 1

CUHK03
	- jpg type(BGR)
	- index of person name is label.
	- images of each person are divied into 2 sub-group: train 70% and test 30%
		Examples:
		# Total samples: 20 Train: 0 ~ 13 Test 14 ~ 19
		# Total samples: 20 Train: 0 ~ 13 Test 14 ~ 19
		# Total samples: 18 Train: 0 ~ 11 Test 12 ~ 17
		# Total samples: 16 Train: 0 ~ 10 Test 11 ~ 15

	
lfw:
	- jpg type(BGR)
	- index of person name is label.
	- 5786 people, only 1680 people has more than two picutres.
	- images of each person are divied into 2 sub-group: train 70% and test 30%
		Examples:
		# Total samples: 3 Train: 0 ~ 1 Test 2 ~ 2
		# Total samples: 2 Train: 0 ~ 0 Test 1 ~ 1
		# Total samples: 4 Train: 0 ~ 1 Test 2 ~ 3
		# Total samples: 6 Train: 0 ~ 3 Test 4 ~ 5
	- first 30 people are excluded.

mnist:
	- raw type(GRAY)
	- data shape is 784
	- reshape 784 >> 28,28
	- convert GRAY >> BGR

stanford:
	- jpg type(BGR)
	- label starts from 1
	
svhn:
	- raw type(RGB)
	- data shape is (32,32,3,n)
	- convert RGB >> BGR



cifar10
├── batches.meta
├── cifar-10-python.tar.gz
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── readme.html
└── test_batch


cifar100
├── cifar-100-python.tar.gz
├── file.txt~
├── meta
├── test
└── train


cuhk03/
├── cuhk-03.mat
├── cuhk03_release.zip
├── open-reid
│?? ├── docs
│?? ├── examples
│?? ├── images
│?? ├── LICENSE
│?? ├── meta.json
│?? ├── raw
│?? ├── README.md
│?? ├── reid
│?? ├── setup.cfg
│?? ├── setup.py
│?? ├── splits.json
│?? ├── test
│?? └── tmp.py
└── README.md


flower/
├── 102flowers.tgz
├── imagelabels.mat
├── jpg
└── setid.mat

lfw/
├── lfw_funneled
└── lfw-funneled.tgz


mnist/
├── t10k-images-idx3-ubyte
├── t10k-labels-idx1-ubyte
├── train-images-idx3-ubyte
└── train-labels-idx1-ubyte

stanford_dog/
├── annotation.tar
├── file_list.mat
├── Images
├── images.tar
├── lists.tar
├── test_list.mat
└── train_list.mat


svhn/
├── test_32x32.mat
└── train_32x32.mat
