allresults = zeros(150,700);
para = [9.85,9.9,9.95,10.05,10.1,10.15];
paraset = combntns(para,2);
for choosepara = 1:15
allintervals = [8,10,12,15,20,24,30,40,50,60];
allnoint = [75,60,50,40,30,25,20,15,12,10];
[adjmat0,M]=NWnetwork(5,1,0);
LL = 4000;
xini=rand(M,1);
yini=rand(M,1);
zini=rand(M,1);

for round = 1:10
    fprintf('data generation for paraset %d round %d \n',choosepara,round)
    x=zeros(M,LL);
    y=zeros(M,LL);
    z=zeros(M,LL);
    x(:,1)=xini;
    y(:,1)=yini;
    z(:,1)=zini;
    X=zeros(3*M,LL);
    interval = allintervals(round);
    noch = allnoint(round);
    changepoints = 1000:interval:1000+interval*(noch-1);
    stepsize=0.01;
    C = 0.1;
    for i = 1:999
        for j=1:M
            x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat0(j,:)*x(:,i));
            y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
            z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
        end
    end
    for chs = 1:noch-1
        for i = changepoints(chs):changepoints(chs+1)-1
            if mod(chs,2)==1
                alpha=paraset(choosepara,1);
            else
                alpha=paraset(choosepara,2);
            end
            for j=1:M
                x(j,i+1)=x(j,i)+stepsize*(alpha*(y(j,i)-x(j,i))+C*adjmat0(j,:)*x(:,i));
                y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
                z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
            end
        end
    end
    for i =changepoints(noch):LL-1
        for j=1:M
            x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat0(j,:)*x(:,i));
            y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
            z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
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
    timelag=950-15-1;
    xx=X(timelag+1:end,:)';

    maxstep=700; % steps to predict
    j=1; % the index of target variable
    trainlength=15+1; % length of training data (observed data)  

    s=30;% number of non-delay embedding
    L=3;% embedding dimension, which could be determined using FNN or set empirically
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
        allresults(round+(choosepara-1)*10,step)=std(pp);
    end
    clear record x y z X
end
clearvars -Except allresults paraset adjmat2 choosepara
end