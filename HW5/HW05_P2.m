%% Equalizer
close all;
clear;

%% Generate signals
% enough to represent one cycle of the sinusoid signal
n = 1:100;
signal = 2*sin(2*pi*n/20);

Xtrain = signal + 0.2*signal.^2;
Dtrain = signal;
% Define scaling function
scaleTo01 = @(x) (x - min(x)) / (max(x) - min(x));
% Scale D to [0,1]
dtrain = scaleTo01(Dtrain);

% Scale X to [0,1]
xtrain = scaleTo01(Xtrain);

%% learning
K= size(Xtrain, 2);

learnrate= [0.001, 0.010];
momentum= 0;
maxstep= 20000*K;
batchsize= 1;
l1_NPE= [80, 50];
tol = 0.0015;

set= [learnrate(2);
       momentum;
       maxstep;
       tol;
       batchsize;
       l1_NPE(2);
       false];


mfreq= 10000;

% BP learning
[MSEtrain, ~, weightsInputHidden, weightsHiddenOutput] = BPlearn(xtrain, xtrain, dtrain, dtrain, set);

% BP recall 
[y, MSErecall] = BPrecall(xtrain, dtrain, weightsInputHidden, weightsHiddenOutput, false);

%% Scale back and report performance
figure(1);
subplot(2,1,1);
plot(1:length(MSEtrain), MSEtrain, "r-*");
title("Learning History: training MSE set");
xlabel("Learning steps/"+num2str(mfreq));
ylabel("MSE")

% Scale back actual output
Dmax= max(Dtrain);
Dmin= min(Dtrain);
Y= y/1*(Dmax-Dmin)+Dmin;

subplot(2,1,2);
p= plot(1:length(Dtrain), Dtrain, 1:length(Y), Y, 1:length(Xtrain), Xtrain);
p(1).LineWidth= 2;
p(2).LineWidth= 2;
xlim([0,100]);
legend("Desired", "Actual", "Distorted Input");
title("Desired VS. Actual Outputs signal: training set");
xticks([]);
ylabel("s(n)");
xlabel("MSE of Actual Ouput: "+num2str(MSErecall));

%% 2 Test sets
% 1
signal1 = 0.8*sin(2*pi*n/10)+0.25*cos(2*pi*n/25);
Xtest1= signal1 + 0.2*signal1.^2;

% 2
signal2= normrnd(0,1,1,length(n));
Xtest2 = signal2 + 0.2*signal2.^2;


% scale input to [0,1]
xtest1 = scaleTo01(Xtest1);

% scale input to [0,1]
xtest2 = scaleTo01(Xtest2);

% recall
[y1, MSErecall1] = BPrecall(xtest1, signal1, weightsInputHidden, weightsHiddenOutput, false);
[y2, MSErecall2] = BPrecall(xtest2, signal2, weightsInputHidden, weightsHiddenOutput, false);
% scale back 
Dmin1 = min(signal1);
Dmax1 = max(signal1);
Dmin2 = min(signal2);
Dmax2 = max(signal2);

Y1= y1*(Dmax1-Dmin1)+Dmin1;
Y2= y2*(Dmax2-Dmin2)+Dmin2;

%Plots
figure(2);
p2= plot(1:length(signal1), signal1, 1:length(Y1), Y1, 1:length(Xtest1), Xtest1);
p2(1).LineWidth= 2;
p2(2).LineWidth= 2;
xlim([0,100]);
legend("Desired", "Actual", "Distorted Input");
title("Desired VS. Actual Outputs signal: test set 1");
xticks([]);
ylabel("s(n)");
xlabel("Actual MSE Ouput: "+num2str(MSErecall1));

figure(3);
p3= plot(1:length(signal2), signal2, 1:length(Y2), Y2, 1:length(Xtest2), Xtest2);
p3(1).LineWidth= 2;
p3(2).LineWidth= 2;
xlim([0,100]);
legend("Desired", "Actual", "Distorted Input");
title("Desired VS. Actual Outputs signal: test set 2");
xticks([]);
ylabel("s(n)");
xlabel("Actual MSE Ouput: "+num2str(MSErecall2));
