function experiment(config_file, datapath, datapath_t, d, a, b, options, seed)

if nargin == 6
    options = '';
end

addpath(genpath('simpleNN'));

%% Train
% ------
% Read train data sets
load(datapath,'y','Z');
y = y - min(y) + 1;
Z = [full(Z) zeros(size(Z,1), a*b*d - size(Z,2))];

% Rearrange data from row-wise to col-wise
Z = reshape(permute(reshape(Z, [],b,a,d), [1,3,2,4]), [], a*b*d);

% Max-min normalization
tmp_max = max(Z, [], 2);
tmp_min = min(Z, [], 2);
Z = (Z - tmp_min) ./ (tmp_max - tmp_min);

% Zero mean
mean_tr = mean(Z);
Z = Z - mean_tr;

model = cnn_train(y, Z, config_file, options, seed, datapath, datapath_t, d, a, b);

%% Test
% -----
% Read test data sets
load(datapath_t,'y','Z');
y = y - min(y) + 1;
Z = [full(Z) zeros(size(Z,1), a*b*d - size(Z,2))];

% Rearrange data from row-wise to col-wise
Z = reshape(permute(reshape(Z, [],b,a,d), [1,3,2,4]), [], a*b*d);

% Max-min normalization
tmp_max = max(Z, [], 2);
tmp_min = min(Z, [], 2);
Z = (Z - tmp_min) ./ (tmp_max - tmp_min);

% Zero mean
Z = Z - mean_tr;

[predicted_label, acc] = cnn_predict(y, Z, model);
fprintf('Seed Number: %g\n', seed);
fprintf('test_acc: %5.5f\n',acc);

