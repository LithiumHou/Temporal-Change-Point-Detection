% study the influence of reconnection probability
clear
clc
maxstep=100; % steps to predict
result=zeros(500,maxstep);
global uini
global adjmat0;
global adjmat1;
global M
[adjmat0,M] = NWnetwork(10,1,0);
count = 0;
difference = zeros(1,500);
for round = 1:500
    [adjmat1,M] = NWnetwork(10,1,0.01); %Set the reconnection probability
    
    uini=rand(1,30);  %initial values
    
    if adjmat0-adjmat1==zeros(M,M)  %To ensure the matrices before and after the CP are different
        disp('Identical matrices')
        
    else
        difference(round) = sum(sum(adjmat1-adjmat0));
        [t, u] = Rk4(@lorenz30,uini',0.01,0,40);  %Use the RK4 algorithm to solve the equation
        record = u(:,1:4000)';
        X = record;
        timelag=2950-30-1;
        xx=X(timelag+1:end,:)';
        j=1; % the index of target variable
        trainlength=30; % length of training data (observed data)
        
        s=200;% number of non-delay embedding
        L=4;% embedding dimension, which could be determined using FNN or set empirically
        
        for step=1:maxstep
            warning off
            predictions=zeros(1,s);
            traindata=xx(:,step:trainlength+step-1);
            real=xx(j,trainlength+step);
            D=size(traindata,1); % number of variables in the system
            cmb=combntns(1:D,L);
            r=randperm(size(cmb,1));
            B=cmb(r,:);
            
            
            parfor i=1:s
                predictions(i)=myprediction_gp(traindata(B(i,:),1:trainlength-1),traindata(j,2:trainlength),traindata(B(i,:),trainlength));% other kinds of fitting method could be used here
            end
            pp=outlieromit(predictions);% exclude the outliers
            [F,XI]=ksdensity(pp,linspace(min(pp),max(pp),10000));% use kernal density estimation to approximate the probability distribution
            prediction=sum(XI.*F/sum(F)); % use expectation as the final one-step predicted value
            result(round,step)=real-prediction; % the prediction error 
        end
        count = count+1;
        count
        
    end
    clear adjmat1 
end