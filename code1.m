%Program for prediction on new data to get trends


%Clear old stuff and create folder named networks if it doesn't 
%already  exist
clear all; fclose all; clc;
if isdir('networks')==0
    mkdir('networks');
end

%Import data
inputs=xlsread('newinput.xlsx');  %input data
targets=xlsread('newtarget.xlsx');%target data
predicts=xlsread('siminput.xlsx');

%transpose the data
inputs = inputs';   
targets = targets';
predicts = predicts';

trainFcn = 'trainbr';  %use the Bayesian Regularisation algorithm
%Train the networks
for i=34  %vary number of hidden layer neurons from 1 to 100 
    hiddenLayerSize = i; %number of hidden layer neurons
    net = feedforwardnet(hiddenLayerSize,trainFcn)
    net.divideParam.trainRatio = 100/100;  %use 100% of data for predictions
    net.divideParam.testRatio = 0/100; %0% for testing as model is ready from previous program
    [net,tr] = train(net,inputs,targets); % train the network
    outputs = net(predicts); %simulate 0% test data
end