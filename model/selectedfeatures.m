function [features] = selectedfeatures(Signal)

features = [
getcl(Signal),... % 海岸线参数
std(Signal),... %标准差
wentropy(Signal,'log energy') %熵
];
end