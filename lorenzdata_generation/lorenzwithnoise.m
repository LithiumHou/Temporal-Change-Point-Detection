function [change1,change2,record] = lorenzwithnoise(ns,adjmat1,adjmat2,adjmat3)
stepsize=0.01;
L=4000;
change1=2500;
change2=2700;

M=5;
x=zeros(M,L);
y=zeros(M,L);
z=zeros(M,L);
x(:,1)=rand(M,1);
y(:,1)=rand(M,1);
z(:,1)=rand(M,1);
% Lorenz system
C=0.1;
A=ns;   %noise strength up to 2e-4
X=zeros(3*M,L);

for i=1:change1-1

for j=1:M
x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat1(j,:)*x(:,i))+A*randn(1,1)*sqrt(stepsize);
y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
end
end


for i=change1:change2-1

for j=1:M
x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat2(j,:)*x(:,i))+A*randn(1,1)*sqrt(stepsize);
y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
end
end
for i=change2:L-1

for j=1:M
x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat3(j,:)*x(:,i))+A*randn(1,1)*sqrt(stepsize);
y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
end
end


for j=1:M
X(3*j-2:3*j,:)=[x(j,:);y(j,:);z(j,:)];
end


record=X';