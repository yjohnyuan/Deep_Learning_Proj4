function model = sgd(prob, param, model, net)

lr = param.lr;
batch_size = param.bsize;
decay = param.decay;

v = cell(param.L,1);
v(:) = {0};

step = 1;
matplot = zeros (500,1);

for k = 1 : param.epoch_max

    Dtimes=char(datetime('now','TimeZone','local','Format','HH:mm:ss Z'));
    disp([' Starting Epoch ' Dtimes] );
    a = 0 ;
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
	fprintf('%d-epoch avg. loss: %g\n', k, loss/batch_size);
    matplot(k)=loss/batch_size;
    %disp(matplot)
    
    
end

Dtimes=char(datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss Z'));
disp(['End Time:  ' Dtimes] );
disp(matplot)
model.param = param;
