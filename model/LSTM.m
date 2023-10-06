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


%按照9：1的比例分配训练集和测试集
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

%XTest =f3;
XTest = [XTestN;XTestG];
YTest = [YTestN;YTestG];
XTrain = cellfun(@(x) x',XTrain,'UniformOutput',false);
XTest= cellfun(@(x) x',XTest,'UniformOutput',false);

%定义layer
layers = [ ...
    sequenceInputLayer(1)
    bilstmLayer(250)
    dropoutLayer
    bilstmLayer(250,'OutputMode','last')
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer
    ];
options = trainingOptions('adam', ...
    'MaxEpochs',200, ...%定义轮数
    'MiniBatchSize', 50, ...
    'InitialLearnRate', 0.001, ...
    'GradientThreshold', 1, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);
%训练深度学习神经网络
net = trainNetwork(XTrain,YTrain,layers,options);

trainPred = classify(net,XTest);
histogram(trainPred);
%stem(trainPred);

LSTMAccuracy = sum(trainPred == YTest)/numel(YTest)*100;
figure
confusionchart(YTest,trainPred,'ColumnSummary','column-normalized',...
              'RowSummary','row-normalized','Title','Confusion Chart for LSTM');