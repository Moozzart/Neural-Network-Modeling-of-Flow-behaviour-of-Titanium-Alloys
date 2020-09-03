%just for plotting the error histogram
%link to the code found on matlab website - https://in.mathworks.com/help/deeplearning/ref/ploterrhist.html



%Clear old stuff and create folder named networks if it doesn't already exist
%clear all; fclose all; clc;
%if isdir('Networks')==0
   % mkdir('Networks');
%end

%Import data
inputs=xlsread('Inputs.xlsx');  %input data
targets=xlsread('ExperimentalOutput.xlsx');  %target experimental data

inputs = inputs';   %transposing the data for matrix multiplication
targets = targets';

%change the training algorithm
trainFcn = 'trainbr';  %use the Bayesian Regularisation algorithm

%Train the networks
for i=34  %vary number of hidden layer neurons from 20 to 40 
    hiddenLayerSize = i; %number of hidden layer neurons (only 1 hidden layer in network)
    net = feedforwardnet(hiddenLayerSize,trainFcn) %create a cascade forward network
    net.divideParam.trainRatio = 70/100;  %use 70% of data for training
    net.divideParam.testRatio = 30/100; %use 30% for testing
    [net,tr] = train(net,inputs,targets); % train the network
    outputs = net(inputs(:,tr.testInd)); %simulate 30% test data
    
    %calculate all errors
    %mpe(i-19)=((mean(abs(outputs-targets(tr.testInd))))*100)/mean(targets(tr.testInd)); %MPE for 30% random test data
    %mae(i-19)= (mean(abs(outputs-targets(tr.testInd)))); %MAE for 30% random test data
    %rmse(i-19)=sqrt(mean((outputs-targets(tr.testInd)).^2)); %RMSE for 30% random test data
    %aare(i-19)=mean(((((abs(outputs-targets(tr.testInd))))*100)/targets(tr.testInd))); %AARE for 30% random test data
    %si(i-19)=rmse(i-19)/mean(targets(tr.testInd)); %SI for 30% random test data
    %nmbe(i-19)= (mean((targets(tr.testInd)-outputs))/mean(targets(tr.testInd)))*100;
    
    %plooting regression plots
    r(i)= regression(targets(tr.testInd), outputs);
    plotregression(targets(tr.testInd),outputs)
    %error = targets(tr.testInd) - outputs;      %absolute error histogram plot
    %ploterrhist(error,'bins',20)
    %hold on
    rel_error = ((targets(tr.testInd) - outputs)*100)/(targets(tr.testInd))
    ploterrhist(rel_error,'bins',20)              %percentage relative error histogram plot
    %save(['Networks\net' num2str(i)],'net');  %save the network in networks folder
end