clear
clc
close all
paraini = 8/3;    %beta is changed in the test
para_after_change = [8.3/3,8.6/3,8.9/3,9.2/3,9.5/3,9.8/3];
max_round = size(para_after_change,2);
LL = 4000;
maxstep=100; % steps to predict
allresults = zeros(max_round,maxstep);
[adjmat0,M]=NWnetwork(5,1,0);

xini=1;
yini=1;
zini=1;

for round = 1:max_round
    fprintf('data generation for round %d \n',round)
    x=zeros(M,LL);
    y=zeros(M,LL);
    z=zeros(M,LL);
    x(:,1)=xini;
    y(:,1)=yini;
    z(:,1)=zini;
    X=zeros(3*M,LL);
    changepoint = 2000;
    stepsize=0.01;
    C = 0.1;
    for i = 1:changepoint-1
        for j=1:M
            x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat0(j,:)*x(:,i));
            y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
            z(j,i+1)=z(j,i)+stepsize*(-1*paraini*z(j,i)+x(j,i)*y(j,i));
        end
    end
    for i = changepoint:LL-1
        for j=1:M
            x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat0(j,:)*x(:,i));
            y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
            z(j,i+1)=z(j,i)+stepsize*(-1*para_after_change(round)*z(j,i)+x(j,i)*y(j,i));
        end
    end
    for j=1:M
        X(3*j-2:3*j,:)=[x(j,:);y(j,:);z(j,:)];
    end
	record=X';
    %Prediction
    Y = eemdbeforepredict(record,0); % dynamic system
    noisestrength=0*10^(-4);  %external noise up to 2e-4
    X=Y+noisestrength*rand(size(Y));% noise could be added
    timelag=1950-20-1;
    xx=X(timelag+1:end,:)';

    j=1; % the index of target variable
    trainlength=20+1; % length of training data (observed data)  

    s=50;% number of non-delay embedding
    L=4;% embedding dimension, which could be determined using FNN or set empirically
    %dis=zeros(6,10000);

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
        allresults(round,step)=real - prediction;
    end
    clear record x y z X
end