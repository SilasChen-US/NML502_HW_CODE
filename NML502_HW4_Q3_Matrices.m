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
plotting_interval = 10000; 

% Momentum coefficient
alpha = 0.9;
% Initialize previous weight updates for momentum
prev_dw1 = zeros(size(w1));
prev_db1 = zeros(size(b1));
prev_dw2 = zeros(size(w2));
prev_db2 = zeros(size(b2));

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

for step = 1:learn_steps

    % ------ disp('step:'), disp(step)

    % Shuffle around/randomize the values from the training dataset
        idx = randperm(numel(x_train));
        x_batch = x_train(idx);
        y_batch = y_train(idx);
        % ------ disp('x batch:'), disp(x_batch)
        % ------ disp('y batch:'), disp(y_batch)


    for batch_start = 1:batch_size:numel(x_train) % Looping through each batch (2 batches) %don't hardwirw; change 100
        
        % ------ disp('batch number:'), disp(batch_start)
        % ------ disp('weight of layer 2 before each batch:'), disp(w2)
        
        % Initializing error for batch
        batch_error = zeros(size(batch_size));
        % ------ disp('displaying batch_error:'),

        % Accumulating all delta values over batch
        dw1_sum = zeros(size(w1));
        db1_sum = zeros(size(b1));
        dw2_sum = zeros(size(w2));
        db2_sum = zeros(size(b2));
        

        for i = batch_start:min(batch_start+batch_size-1, numel(x_train)) %Looping through each of 100 samples


            % ----- disp('Running through 100 samples of x:'), disp(i), disp('displaying x_batch(i):'), disp(x_batch(i))
      

            % Forward pass
            x = x_batch(i);
            desired_output = y_batch(i);

            hidden_output = tanh(w1 * x + b1);
            output = tanh(w2 * hidden_output + b2);


            % Calulcate error
            error = desired_output - output;
            absolute_error = abs(error);
            batch_error(i) = absolute_error;
    
    
            % Backpropagation
            
            %Calculating delta values

            dw2 = error * (1 - output.^2) * hidden_output'; %try moving 'hidden_output' down in dw_sum lines
            db2 = error * (1 - output.^2);
            dw1 = ((1 - hidden_output.^2) .* (w2' * (error * (1 - output.^2)))) * x;%x'; %try moving 'x' down in dw_sum lines
            db1 = (1 - hidden_output.^2) .* (w2' * (error * (1 - output.^2)));


            % Summing up delta values

            dw1_sum = dw1_sum + dw1;
            db1_sum = db1_sum + db1;
            dw2_sum = dw2_sum + dw2;
            db2_sum = db2_sum + db2;
  
        end

        %Calculate mean absolute error
        % ----- disp('batch error for a given batch:'), disp(batch_error)
        mean_absolute_error = sum(batch_error) / batch_size;
        % ----- disp('Printing mean absolute error:'), disp(mean_absolute_error)
        %disp('mean absolute error after a given batch:'), disp(mean_absolute_error)

        % Updating the weights after each batch with momentum
        w1_update = mu * dw1_sum + alpha * prev_dw1;
        b1_update = mu * db1_sum + alpha * prev_db1;
        w2_update = mu * dw2_sum + alpha * prev_dw2;
        b2_update = mu * db2_sum + alpha * prev_db2;
        % Updating the weights after each batch
        w1 = w1 + mu * dw1_sum;
        b1 = b1 + mu * db1_sum;
        w2 = w2 + mu * dw2_sum;
        b2 = b2 + mu * db2_sum;
        % Store the updates for next iteration's momentum calculation
        prev_dw1 = w1_update;
        prev_db1 = b1_update;
        prev_dw2 = w2_update;
        prev_db2 = b2_update;
        

    end

    % ------- Check if error threshold is reached
    if mean_absolute_error <= error_threshold
        disp(['Error threshold reached at step ', num2str(step)])
        break;
    end

    % BP RECALL SECTION HERE
     % Recording errors after each plotting_error intervals 
    if mod(step, plotting_interval) == 0

        
        %Calculate training error
        summed_absolute_training_error = zeros(size(x_train));
        for i = 1:numel(x_train) %% --------- Definitely convert the 'forward pass' and 'calculating output' and 'errors' all into one function since I'm constantly re-using them
            % ----- disp('printing i of training data:'), disp(i)
            
            % Forward pass
            x = x_train(i);
            desired_output = y_train(i);


            hidden_output = tanh(w1 * x + b1);
            output = tanh(w2 * hidden_output + b2);


            % Calulcate error               can be converted into a
            % functionfunction that takes as it's input (the thing to
            % iterate over (e.g. either batch_error or
            % summed_abolsute_training_error or summed_absolute_test_error
            error = desired_output - output;
            absolute_error = abs(error);
    

            %Calculating mean absolute error-- summed up across all 200 training patterns
            summed_absolute_training_error(i) = absolute_error;
            % ------ disp('printing the summed_absolute_errors of training:'), disp(summed_absolute_training_error)


        end
        %Calculating mean absolute error for training set
        mean_absolute_error_training = sum(summed_absolute_training_error) / numel(x_train);
        disp('printing mean absolute error of training:'), disp(mean_absolute_error_training)

        
        % Calculate test error
        summed_absolute_test_error = zeros(size(x_test));
        for i = 1:numel(x_test)

            % -----disp('printing i of testing data:'), disp(i)

            % Forward pass
            x = x_test(i);
            desired_output = y_test(i);

            hidden_output = tanh(w1 * x + b1);
            output = tanh(w2 * hidden_output + b2);

            %calculate error
            error = desired_output - output;
            absolute_error = abs(error);
    
           

            %Calculating mean absolute error-- summed up across all 100 test pairs
            summed_absolute_test_error(i) = absolute_error;
            % --------disp('printing the summed_absolute_errors of testing:'), disp(summed_absolute_test_error)
        end
        %Calcualting mean absolute error for test set
        mean_absolute_error_test = sum(summed_absolute_test_error) / numel(x_test);
        disp('printing mean absolute error of testing:'), disp(mean_absolute_error_test)

        
        % Store errors in arrays
        training_errors(end+1) = mean_absolute_error_training;
        %disp('training errors of learninig history:'), disp(training_errors)
        test_errors(end+1) = mean_absolute_error_test;
        %disp('test errors of learning history:'), disp(test_errors)

    end



end


if mean_absolute_error > error_threshold
    disp('Learning terminated. Maxiumum number of learn steps reached.');
end


%%

% Ploting learning history

figure(1);

plot(1:numel(training_errors), training_errors, 'b', 'LineWidth', 2);
hold on;
plot(1:numel(test_errors), test_errors, 'r', 'LineWidth', 2);
xlabel('Learning steps (x10000)');
ylabel('Mean Absolute Error');
legend('Traning Error', 'Test Error');
title('Learning History');


%% Testing network on test data after total learning steps

predictions = zeros(100, 1); %100 pairs of test data
for i = 1:100
    
    % Forward pass
    x = x_test(i);
    desired_output = y_test(i);

    hidden_output = tanh(w1 * x + b1);
    output = tanh(w2 * hidden_output + b2);

    predictions(i) = output;
    
    fprintf('Input x: %f, Target: %f, Output: %f\n', x, desired_output, predictions(i));


end


%% Plotting the trained network

figure(2);
% Plot y_test
plot(x_test, y_test * 10, 'b', 'LineWidth', 2); % Multiply y_test by 10 to match the y-axis range
hold on;

% Plot predictions
plot(x_test, predictions * 10, 'r--', 'LineWidth', 2); % Multiply predictions by 10 to match the y-axis range

% Add labels and title
xlabel('Input (x)');
ylabel('Output');
title('Predictions vs. Targets');
legend('Target', 'Predictions');


%% Write the BPlearn funcion
function [w1, b1, w2, b2, training_errors, test_errors] = BPlearn(input_neurons, hidden_neurons, output_neurons, mu, alpha, error_threshold, batch_size, learn_steps, x_train, y_train, x_test, y_test)
    % Initialize weights and biases
    w1 = rand(hidden_neurons, input_neurons) * 0.2 - 0.1;
    b1 = rand(hidden_neurons, 1) * 0.2 - 0.1;
    w2 = rand(output_neurons, hidden_neurons) * 0.2 - 0.1;
    b2 = rand(output_neurons, 1) * 0.2 - 0.1;

    % Initialize previous updates for momentum
    prev_dw1 = zeros(size(w1));
    prev_db1 = zeros(size(b1));
    prev_dw2 = zeros(size(w2));
    prev_db2 = zeros(size(b2));

    % Initialize variables to store learning history
    training_errors = [];
    test_errors = [];

    % Training loop
    for step = 1:learn_steps
        % Shuffle and batch data...
        
        % Loop over batches and perform backpropagation...
        
        % Update weights with momentum...
        w1 = w1 + mu * dw1_sum + alpha * prev_dw1;
        b1 = b1 + mu * db1_sum + alpha * prev_db1;
        w2 = w2 + mu * dw2_sum + alpha * prev_dw2;
        b2 = b2 + mu * db2_sum + alpha * prev_db2;

        % Store the updates for momentum
        prev_dw1 = mu * dw1_sum;
        prev_db1 = mu * db1_sum;
        prev_dw2 = mu * dw2_sum;
        prev_db2 = mu * db2_sum;

        % Calculate error for stopping criteria...
        
        % Monitor and record errors every specified number of steps...
        
        % Check for stopping criteria...
    end
end

% BPrecall function
function [predictions, mean_absolute_error] = BPrecall(w1, b1, w2, b2, x_data, y_data)
    predictions = zeros(size(x_data));
    for i = 1:numel(x_data)
        % Forward pass
        hidden_output = tanh(w1 * x_data(i) + b1);
        output = tanh(w2 * hidden_output + b2);
        predictions(i) = output;
    end
    errors = abs(predictions - y_data);
    mean_absolute_error = mean(errors);
end

% Write a main program that performs data input
% Define the function to approximate
f = @(x) 1./x;

% Define network parameters
network_params = struct(...
    'input_neurons', 1, ...
    'hidden_neurons', 10, ...
    'output_neurons', 1, ...
    'learning_rate', 0.0002, ...
    'momentum', 0.9, ...
    'error_threshold', 0.01, ...
    'batch_size', 100, ...
    'learn_steps', 5000 * 100 ...
);

% Generate or input data
x_train = linspace(0.1, 1.0, 200); 
y_train = f(x_train);
x_test = linspace(0.1, 1.0, 100); 
y_test = f(x_test);

% Scale data to fit the network
y_train_scaled = y_train / max(y_train);
y_test_scaled = y_test / max(y_test);

% Call the BPlearn function
[w1, b1, w2, b2, training_errors] = BPlearn(x_train, y_train_scaled, network_params);

% Call the BPrecall function
[predictions_scaled, recall_errors] = BPrecall(w1, b1, w2, b2, x_test, y_test_scaled);

% Inverse scaling of the outputs
predictions = predictions_scaled * max(y_train);

% Post-processing and evaluation
% Compute error metrics
mae_train = mean(abs(y_train - predictions));
mae_test = mean(abs(y_test - predictions_scaled * max(y_test)));

% Plotting learning history
figure(1);
plot(training_errors);
title('Training Error over Epochs');
xlabel('Epoch');
ylabel('Error');

% Plot actual vs. predicted outputs
figure(2);
plot(x_test, y_test, 'o', x_test, predictions, 'x');
legend('Actual Outputs', 'Predicted Outputs');
title('Comparison of Actual and Predicted Outputs');
xlabel('Input');
ylabel('Output');

% Display Mean Absolute Error for training and test set
disp(['Training MAE: ', num2str(mae_train)]);
disp(['Test MAE: ', num2str(mae_test)]);