close all;clear;clc
cd H:\Emily\2021-闭环刺激迷走神经课题\subpicture;
load('H:\Emily\2021-闭环刺激迷走神经课题\subpicture\orignaldata\data20220127.mat')

signal = data(240001:300000);%2275530
figure;plot(signal);
I70_data = signal;

data = signal(12532:13865);


signal = data(540001:600000);%2275530
figure;plot(signal);
I63_data = signal;


data2 = signal(47173:49233);
figure;plot(data2);

data3 = signal(47173:47777); 

data4 = [data;data2;data3];
figure;plot(data4);
ylim([-20000 20000]);

for i=1:4000
    if(data4(i,1)==50000)data4(i,1) = 5000;
    elseif(data4(i,1)==-50000)data4(i,1)=-5000;
    end
end

data4 = resample(data4,1,5);
figure;plot(data4);


L1 = size(signal);
L = L1(1,1);
Fs = 125; % 采样频率
T = 1 / Fs; % 采样时间
t = (0:L-1)*T; % 时间向量
figure;plot(signal);
ylim([-100000 100000])

signalsi = signal(48800:49000);
L1 = size(signalsi);
L = L1(1,1);
Fs = 125; % 采样频率
T = 1 / Fs; % 采样时间
t = (0:L-1)*T; % 时间向量
figure;plot(t,signalsi);

