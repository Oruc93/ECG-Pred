%Z2
N = 10000;
wn2=sqrt(0.1)*randn(1,N);
R2=randn(1,N);
for ii=2:N
R2(ii)=0.5-R2(ii-1)^2+wn2(ii);
end

plot(R2)