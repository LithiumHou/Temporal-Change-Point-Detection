tot = [];
succ = 0;
count = 0;
for i = 1:500
   if difference(i)~=0
       [m, n]=max(result(i,:));
       count = count+1;
       tot = [tot,n-1];
       if (50<=n)&&(n<59)
           succ=succ+1;
       end
   end

end
accuracy = succ/count;