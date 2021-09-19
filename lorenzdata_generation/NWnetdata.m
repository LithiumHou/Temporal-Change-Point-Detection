function [adjmat1,adjmat2,adjmat3,change1,change2,record] = NWnetdata(N,K,p)
stepsize=0.01;
L=4000;
change1=2500;
change2=change1+200;
[adjmat1,M]=NWnetwork(N,K,0);
x=zeros(M,L);
y=zeros(M,L);
z=zeros(M,L);
x(:,1)=rand(M,1);
y(:,1)=rand(M,1);
z(:,1)=rand(M,1);
% Lorenz system
C=0.1;
X=zeros(3*M,L);

for i=1:change1-1

for j=1:M
x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat1(j,:)*x(:,i));
y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
end
end

[adjmat2,M]=NWnetwork(N,K,p);
for i=change1:change2-1

for j=1:M
x(j,i+1)=x(j,i)+stepsize*(10.2*(y(j,i)-x(j,i))+C*adjmat1(j,:)*x(:,i));
y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
end
end

[adjmat3,M]=NWnetwork(N,K,p);
for i=change2:L-1

for j=1:M
x(j,i+1)=x(j,i)+stepsize*(10*(y(j,i)-x(j,i))+C*adjmat3(j,:)*x(:,i));
y(j,i+1)=y(j,i)+stepsize*(28*x(j,i)-y(j,i)-x(j,i)*z(j,i));
z(j,i+1)=z(j,i)+stepsize*(-8/3*z(j,i)+x(j,i)*y(j,i));
end
end

for j=1:M
X(3*j-2:3*j,:)=[x(j,:);y(j,:);z(j,:)];
end


record=X';