disp('hello');

experiment('cifar10-layer3.config', 'data/cifar10-5000.mat', 'data/cifar10.t.mat', 3, 32, 32,'-s 2 -epoch_max 500 -C 0.01 -lr 0.1 -decay 0 -bsize 128 -momentum 0.9',10, 0.1);