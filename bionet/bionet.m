function dy = bionet(t,y,~,k,~)
c1 = 0.45;
sigma1 = 10; sigma2 = 10;
q1 = 50; q2 = 0.02;
L1 = 5e8; L2 = 100;
dy = zeros(3,1);
phi = y(1)*(1+y(1))*(1+y(2))^2/(L1+(1+y(1))^2*(1+y(2))^2);
eta = y(2)*(1+y(3))^2/(L2+(1+y(3))^2);
dy = [c1-sigma1*phi; q1*sigma1*phi-sigma2*eta; q2*sigma2*eta-k*y(3)];
end
