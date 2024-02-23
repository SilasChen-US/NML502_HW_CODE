%% Definitions

% Function to approximate
f = @(x) 1./x;

% Training parameters
mu = 0.0002; % learning rate
error_threshold = 0.01; % mean absolute error on network scale 
batch_size = 100; % or 'K'
learn_steps = 5000 * batch_size;

% Defining network topology:
input_neurons = 1;
hidden_neurons = 10;
output_neurons = 1;

% Defining variables to store learning history
training_errors = [];
test_errors = [];
plotting_interval = 2000; % Adjusted as per advice to every 1000 or 2000 steps

% Initialize weights
w1 = rand(hidden_neurons, 1) * 0.2 - 0.1;
b1 = rand(hidden_neurons, 1) * 0.2 - 0.1;
w2 = rand(1, hidden_neurons) * 0.2 - 0.1;
b2 = rand(1, 1) * 0.2 - 0.1;

%% Generating training/data pairs

% 200 training data pairs (x between 0.1 and 1.0)
x_train = linspace(0.1, 1.0, 200); % we're generating 200 evenly spaced points between 0.1 and 1.0 to ensure we capture the entire range of the function. 
y_train_pre = f(x_train); % we're computing the corresponding y values using the function f(x) = 1/x.

% 100 test data pairs
x_test = [];
while numel(x_test) < 100
    %Generating 1000 random x values b/w 0.1 and 1.0
    rand_x = linspace(0.1, 1.0, 1000);
    rand_x = datasample(rand_x, 100, 'Replace', false);

    %Remove any values overlapping with x_train
    non_overlap_values = setdiff(rand_x, x_train);

    %Add non-overlapping values to x_test
    x_test = [x_test, non_overlap_values];
end

%Re-adjust x_test list length to 100 elements
x_test = x_test(1:100);

%Scaling the output data so it fits within the range of (0,1)
y_train = y_train_pre / max(y_train_pre);
y_test = f(x_test) / max(f(x_test)); % Scaling test data accordingly

%% Training loop:
% ... (No changes needed in the training loop setup)

for step = 1:learn_steps
    % ... (Training code remains the same until checking the plotting interval)

    if mod(step, plotting_interval) == 0
        % ... (Code to calculate training and test errors remains the same)

        % Store errors in arrays using a more frequent interval for monitoring
        training_errors(end+1) = mean_absolute_error_training;
        test_errors(end+1) = mean_absolute_error_test;
    end
end


if mean_absolute_error > error_threshold
    disp('Learning terminated. Maxiumum number of learn steps reached.');
end

%% Plotting the learning history
% Adjusted to include mean absolute error for both training and test set
figure(1);
plot(1:numel(training_errors), training_errors, 'b', 'LineWidth', 2);
hold on;
plot(1:numel(test_errors), test_errors, 'r', 'LineWidth', 2);
xlabel('Learning steps (x2000)');
ylabel('Mean Absolute Error');
legend('Training Error', 'Test Error');
title('Learning History');

%% Actual output vs target output with error display
% Calculate the absolute errors for plotting
training_absolute_errors = abs(f(x_train) - tanh(w2 * tanh(w1 * x_train' + b1) + b2)');
test_absolute_errors = abs(f(x_test) - tanh(w2 * tanh(w1 * x_test' + b1) + b2)');

% Plot actual vs target outputs with consistent color scheme
figure(2);
scatter(x_train, f(x_train), 'b');
hold on;
scatter(x_train, tanh(w2 * tanh(w1 * x_train' + b1) + b2)', 'r');
scatter(x_test, f(x_test), 'b', 'filled');
scatter(x_test, tanh(w2 * tanh(w1 * x_test' + b1) + b2)', 'r', 'filled');
xlabel('Input (x)');
ylabel('Output');
legend('Training Target', 'Training Prediction', 'Test Target', 'Test Prediction');
title('Actual Output vs Target Output');
grid on;

% Display Mean Absolute Error for training and test set
disp(['Training MAE: ', num2str(mean(training_absolute_errors))]);
disp(['Test MAE: ', num2str(mean(test_absolute_errors))]);
