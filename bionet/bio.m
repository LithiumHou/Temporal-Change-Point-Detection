function R = bio(pararesult,~)
global iniy
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
[t,x]=ode45('bionet',[0,2000],iniy,[],pararesult,opts);
R_t = interp1(t,x(:,1),[0:0.25:99.75],'spline');
R = R_t;
end
% x2w=35.97853385199156;
% y2w=84.76342980083892;
% z2w=0.10416547870579615;