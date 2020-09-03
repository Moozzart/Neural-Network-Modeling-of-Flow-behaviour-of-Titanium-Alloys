%Artificial Neural Network Training Program for prediction of flow stress in Titanium alloys 
%Developed by Aniket Niranjan Mishra, Utkarsh Pratap Singh, Saurabh Singh
%under the guidance of Dr. Sumantra Mandal, Associate Professor,
%Department of Metallurgical and Materials Engineering, IIT Kharagpur
%----------------------------------------------------
%This program trains several neural networks (that vary in their number of hidden layer neurons,
%the exact range of number of neurons can be varied) and enables the user to pick the best. 
%The program trains the networks using input and target data contained in files named
%____Inputs.xlsx____ and ____ExperimentalOutput.xlsx____ respectively.
%The program randomly splits the supplied data into 2 portions: 
%70% for training and 30% for testing
%There is an option to change this proportioning in lines 59 to 61
%Also there is an option to use his/her own test data:
%These should be saved in same format as Inputs.xlsx and ExperimentalOutput.xlsx
%New predictions from input data can be made using Prediction trends program.
%Further info about predicting provided in the Prediction trends file.
%When the program runs, it trains many different networks:
%The networks differ in their number of hidden layer neurons
%The networks generated are saved in in a folder named:
%networks, within the same folder where this program is run:
%The network saved as net1 has 1 hidden layer neuron,
%net2 has 2, net3 has 3, net4 has 4,....
%User may change the values 20:40 in line 57 to the desired value
%Training is done using the Beyesian Regularisation algorithm, trainbr 
%The desired algorithm can be changed in line 55
%The program also computes and saves root-mean-squared-errors (RMSEs), Mean
%percentage error, Mean average error, Average absolute relative error and
%Scatter index. in (.txt) files of respective names.
%Data in the file is in 3 columns: 
%column 1 for the number of hidden layer neurons, 
%column 2 for the respective error computed on the random test data
%The data on this file guides the user to decide which of the 
%networks is best. Networks with smaller RMSEs are better, especially
%for the RMSEs computed on the test data which is outside the range of data used for the training.



%Clear old stuff and create folder named networks if it doesn't already exist
clear all; fclose all; clc;
if isdir('Networks')==0
    mkdir('Networks');
end

%Import data
inputs=xlsread('Inputs.xlsx');  %input data
targets=xlsread('ExperimentalOutput.xlsx');  %target experimental data

inputs = inputs';   %transposing the data for matrix multiplication
targets = targets';

%change the training algorithm
trainFcn = 'trainbr';  %use the Bayesian Regularisation algorithm

%Train the networks
for i=20:21  %vary number of hidden layer neurons from 20 to 40 
    hiddenLayerSize = i; %number of hidden layer neurons (only 1 hidden layer in network)
    net = feedforwardnet(hiddenLayerSize,trainFcn) %create a cascade forward network
    net.divideParam.trainRatio = 70/100;  %use 70% of data for training
    net.divideParam.testRatio = 30/100; %use 30% for testing
    [net,tr] = train(net,inputs,targets); % train the network
    outputs = net(inputs(:,tr.testInd)); %simulate 30% test data
    
    %calculate all errors
    mpe(i-19)=((mean(abs(outputs-targets(tr.testInd))))*100)/mean(targets(tr.testInd)); %MPE for 30% random test data
    mae(i-19)= (mean(abs(outputs-targets(tr.testInd)))); %MAE for 30% random test data
    rmse(i-19)=sqrt(mean((outputs-targets(tr.testInd)).^2)); %RMSE for 30% random test data
    aare(i-19)=mean(((((abs(outputs-targets(tr.testInd))))*100)/targets(tr.testInd))); %AARE for 30% random test data
    si(i-19)=rmse(i-19)/mean(targets(tr.testInd)); %SI for 30% random test data
    nmbe(i-19)= (mean((targets(tr.testInd)-outputs))/mean(targets(tr.testInd)))*100;
    
    %plooting regression plots
    r(i)= regression(targets(tr.testInd), outputs);
    plotregression(targets(tr.testInd),outputs)
    save(['Networks\net' num2str(i)],'net');  %save the network in networks folder
end

%Save the values in text files
fid=fopen('mae.txt', 'wt');
fprintf(fid, 'Nh\t MAE\n');
fprintf(fid, '%4.0f\t %f\n', [20:21; mae]);
fclose all;

fid=fopen('nmbe.txt', 'wt');
fprintf(fid, 'Nh\t NMBE\n');
fprintf(fid, '%4.0f\t %f\n', [20:21; nmbe]);
fclose all;


fid=fopen('mpe.txt', 'wt');
fprintf(fid, 'Nh\t MPE\n');
fprintf(fid, '%4.0f\t %f\n', [20:21; mpe]);
fclose all;

fid=fopen('rmse.txt', 'wt');
fprintf(fid, 'Nh\t RMSE\n');
fprintf(fid, '%4.0f\t %f\n', [20:21; rmse]);
fclose all;

fid=fopen('aare.txt', 'wt');
fprintf(fid, 'Nh\t AARE\n');
fprintf(fid, '%4.0f\t %f\n', [20:21; aare]);
fclose all;

fid=fopen('si.txt', 'wt');
fprintf(fid, 'Nh\t SI\n');
fprintf(fid, '%4.0f\t %f\n', [20:21; si]);
fclose all;