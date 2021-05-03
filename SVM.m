%% Artificial Intelligence Homework#3
clear
clc
%% ��ƾ�z
load('iris.txt')
featureSel = [3 4];
cP = iris(iris(:,5)==2,[3,4]);
cN = iris(iris(:,5)==3,[3,4]);
trnP = cP( 1:25,:);
trnN = cN( 1:25,:);
tstP = cP(26:50,:);
tstN = cN(26:50,:);
trnSet = [trnP;trnN];
tstSet = [tstP;tstN];
trnY   = [ones(1,25),ones(1,25)*-1]';
tstY   = trnY;
%% �ѼƳ]�w
ker = 'Liner RBF Polynomial';
C = 10;%�վ�C = 10, 100
S = 5;%�վ�S = 5, 1, 0.5, 0.1, 0.05
p = 7;%�վ�P = 2, 4, 6 
%% Quadratic Programming
L = length(trnY);
for a = 1:L
    u = trnSet(a,:);
    for b = 1:a
        v = trnSet(b,:);
        kerVal = (1+u*v')^p;%  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        H(a,b) = trnY(a)*trnY(b)*kerVal;
        H(b,a) = H(a,b);
    end
end
f   =  -ones(L,1) ;
Aeq =    trnY'    ;
beq =      0      ;
lb  =  zeros(L,1) ;
ub  = C*ones(L,1) ;
alpha1 = quadprog(H,f,[],[],Aeq,beq,lb,ub);
alpha1(alpha1<   sqrt(eps) ) = 0;
alpha1(alpha1>(C-sqrt(eps))) = C;
%% Support Vector Machine
%=========================================================================%
%   ����G���W���ҨD�o��alpha�p��bias�A�Y�i�o��SVM���ҫ�
%   �����q�{���X�Цۦ��{
%=========================================================================%
SV1 = find((alpha1 > 0) & (alpha1 < C));

bias1 = 0;
for a = 1:length(SV1)
    sumtemp = 0;
    u = trnSet(SV1(a),:);
    for b = 1:length(alpha1)
        v = trnSet(b,:);
        kerVal = (1+u*v')^p;%  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        sumtemp = sumtemp + alpha1(b)*trnY(b)*kerVal;
    end
    biasList1(a,1) = 1/trnY(SV1(a))-sumtemp;
    
end

if isempty(SV1)==0
    bias1 = mean(biasList1);
end

for a =1:size(tstSet,1)
    sumtemp = 0;
    u = tstSet(a,:);
    for b = 1:length(alpha1)
        v = trnSet(b,:);
        kerVal = (1+u*v')^p; %  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        sumtemp = sumtemp + alpha1(b)*trnY(b)*kerVal;
    end
    D1(a,1) = sign(sumtemp + bias1);
    
end

CR1 = length(find(D1(1:50,:) == tstY(1:50,:)))/50;



%% Scatter plot and Hyperplane
[xx,yy] = meshgrid(linspace(min([cP(:,1);cN(:,1)]),max([cP(:,1);cN(:,1)]),201),...
    linspace(min([cP(:,2);cN(:,2)]),max([cP(:,2);cN(:,2)]),201));
xy = [xx(:),yy(:)];
for a =1:size(xy,1)
    sumtemp = 0;
    u = xy(a,:);
    for b = 1:length(alpha1)
        v = trnSet(b,:);
        kerVal = (1+u*v')^p;%  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        sumtemp = sumtemp + alpha1(b)*trnY(b)*kerVal;
    end
    D(a,1) = sign(sumtemp + bias1);
end
        
%=========================================================================%
% Ū��demo�Ϊ�SVM���G�A�����G���W�zxy�a�JSVM�ҫ���P�O��decision results
% �ӵ��G�ĥ�rbf-based SVM�AC = 10�Asigma = 0.1
% �����q�{���X�Цۦ��{
%load('demoDecisionResult_rbf_C10_S1E-1')
%=========================================================================%
colorClass = D*-0.5+1.5;
Hyperplane = reshape(colorClass,size(xx));
figure(1)
clf
image(xx(1,:),yy(:,1),Hyperplane)
colormap([1,.4,.4;.4,.4,1]);
set(gca,'YDir','normal');
title(['�Ĥ@������Panelty weight = ',num2str(C),', kernal parameter (sigma/p) = ',num2str(p)])
hold on
plot(trnP(:,1),trnP(:,2),'ks','markerface','r','LineWidth',1,'MarkerSize',10)
plot(trnN(:,1),trnN(:,2),'ks','markerface','b','LineWidth',1,'MarkerSize',10)
plot(tstP(:,1),tstP(:,2),'yo','markerface','r','LineWidth',1,'MarkerSize',5)
plot(tstN(:,1),tstN(:,2),'yo','markerface','b','LineWidth',1,'MarkerSize',5)
legend('Class2 - Training set','Class3 - Training set',...
    'Class2 - Test set','Class3 - Test set','Location','southeast')


%%=============================��e����==================================%%
%% ��ƾ�z
load('iris.txt')
featureSel = [3 4];
cP = iris(iris(:,5)==2,[3,4]);
cN = iris(iris(:,5)==3,[3,4]);
trnP = cP(26:50,:);
trnN = cN(26:50,:);
tstP = cP( 1:25,:);
tstN = cN( 1:25,:);
trnSet = [trnP;trnN];
tstSet = [tstP;tstN];
trnY   = [ones(1,25),ones(1,25)*-1]';
tstY   = trnY;

%% Quadratic Programming
L = length(trnY);
for a = 1:L
    u = trnSet(a,:);
    for b = 1:a
        v = trnSet(b,:);
        kerVal = (1+u*v')^p;%  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        H(a,b) = trnY(a)*trnY(b)*kerVal;
        H(b,a) = H(a,b);
    end
end
f   =  -ones(L,1) ;
Aeq =    trnY'    ;
beq =      0      ;
lb  =  zeros(L,1) ;
ub  = C*ones(L,1) ;
alpha2 = quadprog(H,f,[],[],Aeq,beq,lb,ub);
alpha2(alpha2<   sqrt(eps) ) = 0;
alpha2(alpha2>(C-sqrt(eps))) = C;

%% Support Vector Machine
SV2 = find((alpha2 > 0) & (alpha2 < C));

bias2 = 0;
for a = 1:length(SV2)
    sumtemp = 0;
    u = trnSet(SV2(a),:);
    for b = 1:length(alpha2)
        v = trnSet(b,:);
        kerVal = (1+u*v')^p;%  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        sumtemp = sumtemp + alpha2(b)*trnY(b)*kerVal;
    end
    biasList2(a,1) = 1/trnY(SV2(a))-sumtemp;
    
end

if isempty(SV2)==0
    bias2 = mean(biasList2);
end

for a =1:size(tstSet,1)
    sumtemp = 0;
    u = tstSet(a,:);
    for b = 1:length(alpha2)
        v = trnSet(b,:);
        kerVal = (1+u*v')^p;%  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        sumtemp = sumtemp + alpha2(b)*trnY(b)*kerVal;
    end
    D2(a,1) = sign(sumtemp + bias2);
    
end

CR2 = length(find(D2(1:50,:) == tstY(1:50,:)))/50;

%% Scatter plot and Hyperplane
[xx,yy] = meshgrid(linspace(min([cP(:,1);cN(:,1)]),max([cP(:,1);cN(:,1)]),201),...
    linspace(min([cP(:,2);cN(:,2)]),max([cP(:,2);cN(:,2)]),201));
xy = [xx(:),yy(:)];
for a =1:size(xy,1)
    sumtemp = 0;
    u = xy(a,:);
    for b = 1:length(alpha2)
        v = trnSet(b,:);
        kerVal = (1+u*v')^p;%  [liear = u*v']   [RBF = exp(-(u-v)*(u-v)'/(2*S^2))]   [Polynomial = (1+u*v')^p
        sumtemp = sumtemp + alpha2(b)*trnY(b)*kerVal;
    end
    D(a,1) = sign(sumtemp + bias2);
end
        
%=========================================================================%
% Ū��demo�Ϊ�SVM���G�A�����G���W�zxy�a�JSVM�ҫ���P�O��decision results
% �ӵ��G�ĥ�rbf-based SVM�AC = 10�Asigma = 0.1
% �����q�{���X�Цۦ��{
%load('demoDecisionResult_rbf_C10_S1E-1')
%=========================================================================%
colorClass = D*-0.5+1.5;
Hyperplane = reshape(colorClass,size(xx));
figure(2)
clf
image(xx(1,:),yy(:,1),Hyperplane)
colormap([1,.4,.4;.4,.4,1]);
set(gca,'YDir','normal');
title(['�ĤG������Panelty weight = ',num2str(C),', kernal parameter (sigma/p) = ',num2str(p)])
hold on
plot(trnP(:,1),trnP(:,2),'ks','markerface','r','LineWidth',1,'MarkerSize',10)
plot(trnN(:,1),trnN(:,2),'ks','markerface','b','LineWidth',1,'MarkerSize',10)
plot(tstP(:,1),tstP(:,2),'yo','markerface','r','LineWidth',1,'MarkerSize',5)
plot(tstN(:,1),tstN(:,2),'yo','markerface','b','LineWidth',1,'MarkerSize',5)
legend('Class2 - Training set','Class3 - Training set',...
    'Class2 - Test set','Class3 - Test set','Location','southeast')



fprintf('�Ĥ@�������vCR1 = %2.4f%%\n', CR1*100);
fprintf('�ĤG�������vCR2 = %2.4f%%\n', CR2*100);

CR = (CR1+CR2)/2;
fprintf('���������vCR = %2.4f%%\n', CR*100);






