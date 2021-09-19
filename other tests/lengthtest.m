clear
maxstep=100; % steps to predict
result=zeros(200,maxstep);
for leng=10:5:30
    for ini = 2:2
    %data generation
    [adjmat0,M]=NWnetwork(10,1,0);
    L = 4000;
    stepsize = 0.01;
    C = 0.1;
    x=zeros(M,L);
    y=zeros(M,L);
    z=zeros(M,L);
    xini=1*ones(M,1);
    yini=0*ones(M,1);
    zini=0*ones(M,1);
    x(:,1)=xini;
    y(:,1)=yini;
    z(:,1)=zini;
    change1=2500;   %The preset change point
    for i=1:change1-1
    for j=1:M
    x(j,i+1)=x(j,i)+stepsize*((10+ini*0.5)*(y(j,i)-x(j,i))+C*adjmat0(j,:)*x(:,i));
    y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
    z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
    end
    end
    for i=change1:L-1

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
    Y = record;
    j=1; % the index of target variable
    s=100;% number of non-delay embedding
    L=4;% embedding dimension, which could be determined using FNN or set empirically
    timelag=2450-leng-1;  %preset change points: 2500
    xx=Y(timelag+1:end,:)';
    trainlength=leng;
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
        result((leng-5)*2+ini,step)=std(pp);
    end
    clearvars -except result maxstep leng ini adjmat2
    end
    leng  %print the current length
end