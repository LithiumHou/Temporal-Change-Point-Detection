clearvars -Except record
% Data generation
%record = table2array(record);
Y = eemdbeforepredict(record,0); % dynamic system
%Y = matsurrogate(Y,1,6,1);
%Y = record;
noisestrength=0*10^(-1);  %external noise up to 2e-4
X=Y+noisestrength*rand(size(Y));% noise could be added
timelag=0;
xx=X(timelag+1:end,:)';

maxstep=19000; % steps to predict
j=18; % the index of target variable
trainlength=30; % length of training data (observed data)  

s=80;% number of non-delay embedding
L=4;% embedding dimension, which could be determined using FNN or set empirically
result=zeros(3,maxstep);
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
result(1,step)=prediction; % use expectation as the final one-step predicted value 
result(2,step)=std(pp);           % the standard error of the randomly-picked predictions
result(3,step)=real - prediction; % the error between the prediction and the real data

if mod(step,100)==1
    step
end
end

c1=(trainlength+1):(trainlength+maxstep);
c2=result(1,1:maxstep);
figure
plot(xx(j,1:trainlength+maxstep),'-*'); % real data
hold on;
plot(c1,c2,'ro');

figure
plot(result(2,1:maxstep));

figure
plot(result(3,1:maxstep));
