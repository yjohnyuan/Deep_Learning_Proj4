disp('newton mnist 21')
experiment('cifar10-layer3.config', 'data/cifar10-train.mat', 'data/cifar10-test.mat', 3, 32, 32,'-s 1 -SR 0.2 -iter_max 1 -C 0.01 -xi 0.1 -CGmax 250 -lambda 1 -drop 2/3 -boost 3/2 -eta 0.0001',10, 21);

disp('newton mnist 51')
experiment('cifar10-layer3.config', 'data/cifar10-train.mat', 'data/cifar10-test.mat', 3, 32, 32,'-s 1 -SR 0.5 -iter_max 1 -C 0.01 -xi 0.1 -CGmax 250 -lambda 1 -drop 2/3 -boost 3/2 -eta 0.0001',10, 51);

disp('newton mnist 20')
experiment('cifar10-layer3.config', 'data/cifar10-train.mat', 'data/cifar10-test.mat', 3, 32, 32,'-s 1 -SR 0.2 -iter_max 1 -C 0.01 -xi 0.1 -CGmax 250 -lambda 0 -drop 2/3 -boost 3/2 -eta 0.0001',10, 20);

disp('newton mnist 50')
experiment('cifar10-layer3.config', 'data/cifar10-train.mat', 'data/cifar10-test.mat', 3, 32, 32,'-s 1 -SR 0.5 -iter_max 1 -C 0.01 -xi 0.1 -CGmax 250 -lambda 0 -drop 2/3 -boost 3/2 -eta 0.0001',10, 50);
