% The main function to make predictions on the target record

% Data generation
[adjmat1,adjmat2,adjmat3,change1,change2,record] = NWnetdata(10,1,0.2); % Two cps are set at 2500 and 2700
clearvars -Except record result adjmat1 adjmat2 adjmat3 change1 change2

Y = record;
noisestrength=0*10^(-4);  %external noise up to 2e-4
X=Y+noisestrength*rand(size(Y));% noise could be added
trainlength=30; % length of training data (observed data)  
timelag=2400-trainlength-1; % Skip the first 2400 points, for data alignment we need to cut off 30 more points (equal to the trainlength)
xx=X(timelag+1:end,:)';

maxstep=400; % steps to predict
j=1; % the index of target variable

s=600;% number of non-delay embedding
L=4;% embedding dimension, which could be determined using FNN or set empirically
result=zeros(3,maxstep);

for step=1:maxstep
warning off
predictions=zeros(1,s);
traindata=xx(:,step:trainlength+step-1);
real=xx(j,trainlength+step);
D=size(traindata,1); % number of variables in the system
cmb=nchoosek(1:D,L);
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

% Output the std.dev data for changepoint detection
writematrix(result(2,1:maxstep),'/Users/houjiawen/Desktop/stderror.csv');
