

%Third hypothesis of assignment 2 using the next 3 features with the same
%function of x in all other hypotheses

clc
clear all
close all
 
ds = tabularTextDatastore('heart_DD.csv','TreatAsMissing','NA',.....
    'MissingValue',0,'ReadSize',250);
T = read(ds);

m= length(T{:,1});
m_60 = ceil(length(T{:,1})*0.6);
m_20 = ceil(length(T{:,1})*0.2);
 

% Using the first 60% of the data

x=T{1:m_60,7:9}; %selected features
xrest=T{1:m_60,[1:6 10:13]};
y=T{1:m_60,14};
 
pos=find(y==1);
neg=find(y==0);
 
%Plotting
plot(x(pos,1), x(pos, 2), 'kx', 'MarkerSize', 5);
hold on
plot(x(neg,1), x(neg, 2), 'ko', 'MarkerSize', 5, 'Color', 'r');
 
%compute cost and gradient
Alpha=0.01;
lambda=100;
 
X=[ones(m_60,1) x xrest.^2 exp(-x.^2)];
Y=T{1:m_60,14}/mean(T{1:m_60,14});


n=length(X(1,:)); 
 for w=2:n
    if max(abs(X(:,w)))~=0
    X(:,w)=(X(:,w)-mean((X(:,w))))./std(X(:,w));
    end
 end
 
 theta=zeros(n,1);
 
 h=1./(1+exp(-X*theta));  %sigmoid function
 
 k=1;
 J(k)=-(1/m_60)*sum(Y.*log(h)+(1-Y).*log(1-h))+(lambda/(2*m_60))*sum((theta).^2);  %cost function
 
 grad=zeros(size(theta,1),1);     %gradient vector
  
 for i=1:size(grad)
     grad(i)=(1/m_60)*sum((h-Y)'*X(:,i));
 end
 

R=1;
while R==1
Alpha=Alpha*1;
theta=theta-(Alpha/m_60)*X'*(h-Y);
h=1./(1+exp(-X*theta));  %sigmoid function
k=k+1

J(k)=(-1/m_60)*sum(Y.*log(h)+(1-Y).*log(1-h))+(lambda/(2*m_60))*sum((theta).^2);
if J(k-1)-J(k) <0 
    break
end 
q=(J(k-1)-J(k))./J(k-1);
if q <.00001
    R=0;
end
end
 
% Using the second 20% of the data

x1=T{m_60+1:m_60+m_20,7:9}; %selected features
xrest1=T{m_60+1:m_60+m_20,[1:6 10:13]};
y1=T{m_60+1:m_60+m_20,14};
 
pos=find(y1==1);
neg=find(y1==0);
 
%Plotting
figure(2)
plot(x1(pos,1), x1(pos, 2), 'kx', 'MarkerSize', 5);
hold on
plot(x1(neg,1), x1(neg, 2), 'ko', 'MarkerSize', 5, 'Color', 'r');
 
 
X1=[ones(m_20,1) x1 xrest1.^2 exp(-x1)];
Y1=y1;


n=length(X1(1,:)); 
 for w=2:n
    if max(abs(X1(:,w)))~=0
    X1(:,w)=(X1(:,w)-mean((X1(:,w))))./std(X1(:,w));
    end
 end
 
 theta1=theta;
 
 h1=1./(1+exp(-X1*theta1));  %sigmoid function
 
 k=1;
 J1(k)=-(1/m_20)*sum(Y1.*log(h1)+(1-Y1).*log(1-h1))+(lambda/(2*m_20))*sum((theta1).^2);  %cost function
 
 grad1=zeros(size(theta1,1),1);     %gradient vector
  
 for i=1:size(grad1)
     grad1(i)=(1/m_20)*sum((h1-Y1)'*X1(:,i));
 end
 

 % Using the last 20% of the data

m_last=m_60+m_20+1;
x2=T{m_last:end,7:9}; %selected features
xrest2=T{m_last:end,[1:6 10:13]};
y2=T{m_last:end,14};
 
pos=find(y2==1);
neg=find(y2==0);
 
%Plotting
figure(3)
plot(x2(pos,1), x2(pos, 2), 'kx', 'MarkerSize', 5);
hold on
plot(x2(neg,1), x2(neg, 2), 'ko', 'MarkerSize', 5, 'Color', 'r');
 
 
X2=[ones(length(y2),1) x2 xrest2.^2 exp(-x2)];
Y2=y2;


n=length(X2(1,:)); 
 for w=2:n
    if max(abs(X2(:,w)))~=0
    X2(:,w)=(X2(:,w)-mean((X2(:,w))))./std(X2(:,w));
    end
 end
 
 theta2=theta1;
 
 h2=1./(1+exp(-X2*theta2));  %sigmoid function
 
 k=1;
 J2(k)=-(1/m_20)*sum(Y2.*log(h2)+(1-Y2).*log(1-h2))+(lambda/(2*m_20))*sum((theta2).^2);  %cost function
 
 grad2=zeros(size(theta2,1),1);     %gradient vector
  
 for i=1:size(grad2)
     grad2(i)=(1/m_20)*sum((h2-Y2)'*X2(:,i));
 end
 

