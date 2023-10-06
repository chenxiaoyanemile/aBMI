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
XTrain = cellfun(@(x) x',XTrain,'UniformOutput',false);
XTest= cellfun(@(x) x',XTest,'UniformOutput',false);

cd G:\code
for i = 1:160
    featuresTrain(i,:) = ndfeatures(XTrain{i}');
end
for i = 1:40
    featuresTest(i,:) = ndfeatures(XTest{i}');
end
   
c = cvpartition(160,'KFold',10);
%优化器参数
opts = struct('Optimizer','bayesopt','ShowPlots',true,'CVPartition',c,...
    'AcquisitionFunctionName','expected-improvement-plus');
%进行训练
svmmod = fitcsvm(featuresTrain,YTrain,'KernelFunction','rbf',...
    'OptimizeHyperparameters','auto','HyperparameterOptimizationOptions',opts)
%分类器的交叉验证
CVSVMModel = crossval(svmmod); 
%评价损失
lossnew = kfoldLoss(fitcsvm(featuresTrain,YTrain,'CVPartition',c,'KernelFunction','rbf',...
    'BoxConstraint',svmmod.HyperparameterOptimizationResults.XAtMinObjective.BoxConstraint,...
    'KernelScale',svmmod.HyperparameterOptimizationResults.XAtMinObjective.KernelScale))

 
%模型使用 label是判断的标签，score是预测分数
[YResult,scores] = predict(svmmod,featuresTest);
table(YTest,YResult,scores(1:40,2),'VariableNames',...
    {'TrueLabel','PredictedLabel','Score'})

YResult=YResult';%这个为预测的标签，转化之后是1X34061矩阵
YTest=YTest';%这个为原标签，转化之后也是1X34061矩阵
figure
confusionchart(YTest,YResult,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for SVM')


