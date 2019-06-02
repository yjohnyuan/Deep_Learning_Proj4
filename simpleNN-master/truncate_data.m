load('data_old/mnist.t.mat','y','Z');

idx = y==[0,1,2,3];

idx = idx(:,1) + idx(:,2) + idx(:,3) + idx(:,4);

sum(idx(:) == 1);

y = (y(idx==1,:));

Z = (Z(idx==1,:));

save('data/mnist-test.mat','y','Z');