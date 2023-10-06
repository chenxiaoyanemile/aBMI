function [features] = ndfeatures(Signal)

features = [
mean(instfreq(Signal,125)),...  %瞬时频率的均值
mean(pentropy(Signal,125)),...  %谱熵的均值
norm(Signal),...  %欧几里得范数
getcl(Signal),... % 海岸线参数
std(Signal),... %标准差
wentropy(Signal) %熵
];
end