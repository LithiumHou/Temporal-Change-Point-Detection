% fit ks in the bionet model
% prediction parameters:
% timelag=3500-15-1;maxstep=1000;j=1; trainlength=15+1; 
% s=15;L=3;
% data = result(1,:);
clearvars -Except data record   %data: 1~1000
clc
global init
init = 601;   % before: 1;  inter: 301   after:601
global iniy
iniy = record(3500+init,[1,4,7]);
maxstep = 6;
paraini = 1;
par=zeros(1,maxstep+1);
par(1,1) = paraini;
options=optimoptions('lsqcurvefit','FiniteDifferenceType','central','Display','Iter','FunctionTolerance',1e-12,'StepTolerance',1e-10,'MaxFunctionEvaluations',2000);
% options = statset('Display','Iter');
for step=1:maxstep
    [par(1,step+1),resnorm]=lsqcurvefit(@bio,par(1,step),[0:0.25:99.75],data(init:init+399),[],[],options);
end