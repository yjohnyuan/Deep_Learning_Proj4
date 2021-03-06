function model = sgd(prob, param, model, net, datapath, datapath_t, d, a, b, lr)

lr = param.lr;
batch_size = param.bsize;
decay = param.decay;

v = cell(param.L,1);
v(:) = {0};

step = 1;
matplot = zeros (500,1);

% Read test data sets-------------------
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
mean_tr = mean(Z);
Z = Z - mean_tr;
%---------------------------------------

epochs = [];
accuracies = [];
times = [];
unix_time = num2str(posixtime(datetime('now')) * 1e6);

start_tic = tic;
for k = 1 : param.epoch_max
    epoch_tic = tic;
    
    %a = 0;
	for j = 1 : ceil(prob.l/batch_size)
        %dt = datestr(now,'HH:MM:SS.FFF;');
        %disp(['First Loop dt ' dt] );
        %disp(['a time ' char(a)] );
        %t1 = datevec(now,'mmmm dd, yyyy HH:MM:SS.FFF');
        batch_idx = randsample(prob.l, batch_size);
		[net, loss] = lossgrad_subset(prob, param, model, net, batch_idx, 'fungrad');

		for m = 1 : param.L
			Grad = [net.dlossdW{m} net.dlossdb{m}]/batch_size;
            %I put this line back from proj4 to proj 5 so the C term is
            %activated - John
			Grad = Grad + [model.weight{m} model.bias{m}]/param.C;
			v{m} = param.momentum*v{m} - lr*Grad;
			model.weight{m} = model.weight{m} + v{m}(:,1:end-1);
			model.bias{m} = model.bias{m} + v{m}(:,end);
		end
		lr = param.lr/(1 + decay*step);
		step = step + 1;
        %t2=clock;
        %e=etime(t1,t2);
        %a=a+e;
	end
	%fprintf('%d-epoch avg. loss: %g\n', k, loss/batch_size);
    %matplot(k)=loss/batch_size;
    %disp(matplot)
    %toc;
    
    model.param = param;
    if mod(k,5) == 0 || k==1
        [predicted_label, acc] = cnn_predict(y, Z, model);
        fprintf('test_acc: %5.5f\n',acc);
        accuracies = [accuracies acc];

        epochs = [epochs k];

        total_elapsed_toc = toc(start_tic);
        times = [times total_elapsed_toc];        
        
        l_rate = num2str(lr);
        result_name = strcat('results/results_sgd_',extractAfter(datapath,5),'_',l_rate,'_',unix_time,'.mat');
        save(result_name,'epochs','accuracies','times');
    end    
    
    disp("This epoch:");
    toc(epoch_tic);
    disp("Total:");
    toc(start_tic);
end

Dtimes=char(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z'));
disp(['End Time:  ' Dtimes] );
%disp(matplot)
model.param = param;
