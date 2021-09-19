% Generate the small-world network
function [matrix,node] = NWnetwork(N,m,p)
% matrix=sparse([]);
matrix = zeros(N,N);
for i=m+1:N- m
    matrix(i,i- m:i+m)=1;
end
for i=1:m
    matrix(i,1:i+m)=1;
end
for i=N- m+1:N
    matrix(i,i- m:N)=1;
end
for i=1:m
    matrix(i,N- m+i:N)=1;
    matrix(N- m+i:N,i)=1;
end
% Random add edge
kk=(rand(N,N)<p);
matrix = logical(matrix + kk);
matrix = matrix -diag(diag(matrix));
node = N;