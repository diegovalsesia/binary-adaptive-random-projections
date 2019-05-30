%
% ROC test adaptive embedding
%

clear all
close all

dham = @(a,B) sum(bsxfun(@ne,a,B));

n = 8192;
N1 = 1000;
N2 = N1;
mpool = 8192;
m = 512;
m_na = 2969;
%m_na = 3269;
phi = randn(mpool,n);

% Reference signal
u = randn(n,1);
u = u/norm(u);

% Class 1: non-matching (expected correaltion 0)
v1 = randn(n,N1);
parfor ii=1:N1
    v1(:,ii) = v1(:,ii)/norm(v1(:,ii));
end

% Class 2: matching (expected correaltion rho)
sigma = 0.15;
v2 = zeros(n,N2);
parfor ii=1:N2
    v2(:,ii) = u + sigma*randn(n,1);
    v2(:,ii) = v2(:,ii)/norm(v2(:,ii));
end

iter=1;
Niter=300;
Pd = zeros(Niter,1);
Pfa = zeros(Niter,1);
for tau=linspace(-1,1,Niter)
    Pd(iter) = mean(corr(u,v2)>tau);
    Pfa(iter) = mean(corr(u,v1)>tau);
    iter=iter+1;
end

histogram(corr(u,v1))
hold on
histogram(corr(u,v2))
title('Uncompressed')


% Non-adaptive (eq. complexity)
y_JL = (sign( phi(1:m,:)*u )+1)/2;
z1_JL = (sign( phi(1:m,:)*v1 )+1)/2;
z2_JL = (sign( phi(1:m,:)*v2 )+1)/2;

Pd_JL = zeros(m,1);
Pfa_JL = zeros(m,1);
parfor iter=1:m
    Pd_JL(iter) = mean(dham(y_JL,z2_JL)<iter);
    Pfa_JL(iter) = mean(dham(y_JL,z1_JL)<iter);
end

% figure
% histogram(dham(y_JL,z1_JL))
% hold on
% histogram(dham(y_JL,z2_JL))
% title('Nonadaptive JL')


% Adaptive 1-bit embedding
y_ada = phi*u;
[~,pp] = sort(abs(y_ada),'descend');
y_ada = (sign(y_ada(pp(1:m)))+1)/2;
z1_ada = (sign( phi(pp(1:m),:)*v1 )+1)/2;
z2_ada = (sign( phi(pp(1:m),:)*v2 )+1)/2;

Pd_ada = zeros(m,1);
Pfa_ada = zeros(m,1);
parfor iter=1:m
    Pd_ada(iter) = mean(dham(y_ada,z2_ada)<iter);
    Pfa_ada(iter) = mean(dham(y_ada,z1_ada)<iter);
end

% figure
% histogram(dham(y_ada,z1_ada))
% hold on
% histogram(dham(y_ada,z2_ada))
% title('Adaptive JL')


% Non-adaptive (eq. storage)
y_JL = (sign( phi(1:m_na,:)*u )+1)/2;
z1_JL = (sign( phi(1:m_na,:)*v1 )+1)/2;
z2_JL = (sign( phi(1:m_na,:)*v2 )+1)/2;

Pd_JL2 = zeros(m,1);
Pfa_JL2 = zeros(m,1);
parfor iter=1:m_na
    Pd_JL2(iter) = mean(dham(y_JL,z2_JL)<iter);
    Pfa_JL2(iter) = mean(dham(y_JL,z1_JL)<iter);
end

% figure
% histogram(dham(y_JL,z1_JL))
% hold on
% histogram(dham(y_JL,z2_JL))
% title('Stupid JL')


% Universal (eq. complexity)
D = 2;
q_uni = @(a) mod(floor(a/D),2);
y_uni = q_uni(phi(1:m,:)*u);
z1_uni = q_uni( phi(1:m,:)*v1 );
z2_uni = q_uni( phi(1:m,:)*v2 );

Pd_uni = zeros(m,1);
Pfa_uni = zeros(m,1);
parfor iter=1:m
    Pd_uni(iter) = mean(dham(y_uni,z2_uni)<iter);
    Pfa_uni(iter) = mean(dham(y_uni,z1_uni)<iter);
end

% figure
% histogram(dham(y_uni,z1_uni))
% hold on
% histogram(dham(y_uni,z2_uni))
% title('Universal')

% Universal (eq. storage)
D = 2;
q_uni = @(a) mod(floor(a/D),2);
y_uni = q_uni(phi(1:m_na,:)*u);
z1_uni = q_uni( phi(1:m_na,:)*v1 );
z2_uni = q_uni( phi(1:m_na,:)*v2 );

Pd_uni2 = zeros(m_na,1);
Pfa_uni2 = zeros(m_na,1);
parfor iter=1:m_na
    Pd_uni2(iter) = mean(dham(y_uni,z2_uni)<iter);
    Pfa_uni2(iter) = mean(dham(y_uni,z1_uni)<iter);
end

% figure
% histogram(dham(y_uni,z1_uni))
% hold on
% histogram(dham(y_uni,z2_uni))
% title('Universal')


figure
plot(Pfa,Pd)
hold on
plot(Pfa_JL,Pd_JL)
plot(Pfa_JL2,Pd_JL2)
plot(Pfa_ada,Pd_ada)
plot(Pfa_uni,Pd_uni)
plot(Pfa_uni2,Pd_uni2)
legend('Uncompressed','Nonadaptive (eq. complexity)','Nonadaptive (eq. storage)','Adaptive','Universal (eq. complexity)','Universal (eq. storage)','Location','SouthEast')
