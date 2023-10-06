%closed-loop BMI fig2 : 2023.6.4
close all;clear;clc
cd H:\Emily\2021-闭环刺激迷走神经课题\Fig2大礼包;
load('epilepse_label.mat')
load('epilepsedataset.mat')
load('normal_label.mat')
load('normaldataset.mat')
gpuDevice(1);
fs = 125;
epilepse_label = categorical(epilepse_label);
normal_label = categorical(normal_label);
%将标签与数据集匹配
normalX = normaldataset(normal_label=='0');
normalY = normal_label(normal_label=='0');
GTCSX = epilepsedataset(epilepse_label=='1');
GTCSY = epilepse_label(epilepse_label=='1');

%按照8：2的比例分配训练集和测试集
[trainIndn,~,testIndn] = dividerand(100,0.8,0.0,0.2);
[trainIndG,~,testIndG] = dividerand(100,0.8,0.0,0.2);

%将不同标签下的训练集和测试集合为整体Train和Test
XTrainN = normalX(trainIndn);
YTrainN = normalY(trainIndn);
XTestN = normalX(testIndn);
YTestN = normalY(testIndn);

XTrainG = GTCSX(trainIndG);
YTrainG = GTCSY(trainIndG);
XTestG = GTCSX(testIndG);
YTestG = GTCSY(testIndG);

XTrain = [XTrainN;XTrainG];
YTrain = [YTrainN;YTrainG];

XTest = [XTestN;XTestG];
YTest = [YTestN;YTestG];

XTrainG = cellfun(@(x) x',XTrainG,'UniformOutput',false);
XTest= cellfun(@(x) x',XTest,'UniformOutput',false);

for i = 1:80
    featuresTrain(i,:) = selectedfeatures(XTrainG{i}');
end
for i = 1:40
    featuresTest(i,:) = selectedfeatures(XTest{i}');
end

CLThreshold1 =min(featuresTrain(:,1)); %  1892
CLThreshold2 = mean(featuresTrain(:,1));
STDThreshold1 =min(featuresTrain(:,2)); % 3358
STDThreshold2 = mean(featuresTrain(:,2));
WentropyThreshold1 = min(featuresTrain(:,3));% 4971
WentropyThreshold2 = mean(featuresTrain(:,3));


for i=1:40
    if(featuresTest(i,1)> CLThreshold1 && featuresTest(i,2)>STDThreshold1 && featuresTest(i,3)>WentropyThreshold1) YResult(i,1) = 1;% epilepsy
    elseif(featuresTest(i,1)<CLThreshold1 && featuresTest(i,2)>STDThreshold2 && featuresTest(i,3)>WentropyThreshold1) YResult(i,1) = 1;% epilepsy
     elseif(featuresTest(i,1)>CLThreshold1 && featuresTest(i,2)>STDThreshold2 && featuresTest(i,3)<WentropyThreshold1) YResult(i,1) = 1;% epilepsy
     elseif(featuresTest(i,1)<CLThreshold1 && featuresTest(i,2)>STDThreshold1 && featuresTest(i,3)>WentropyThreshold2) YResult(i,1) = 1;% epilepsy
     elseif(featuresTest(i,1)>CLThreshold1 && featuresTest(i,2)<STDThreshold1 && featuresTest(i,3)>WentropyThreshold2) YResult(i,1) = 1;% epilepsy
     elseif(featuresTest(i,1)>CLThreshold2 && featuresTest(i,2)<STDThreshold1 && featuresTest(i,3)>WentropyThreshold1) YResult(i,1) = 1;% epilepsy
     elseif(featuresTest(i,1)>CLThreshold2 && featuresTest(i,2)>STDThreshold1 && featuresTest(i,3)<WentropyThreshold1) YResult(i,1) = 1;% epilepsy
    else YResult(i,1) = 0;% normal        
    end  
end
YResult = categorical(YResult);
figure
confusionchart(YTest,YResult,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for Threshold')


