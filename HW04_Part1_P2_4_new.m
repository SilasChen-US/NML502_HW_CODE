clc; clear all; close all;
hidden_neurons = 10; 
learning_rate = 0.05;
batch_size = 100; 
training_epochs = 5000;
input_range = [0.1, 1.0]; 

train_inputs = input_range(1) + (input_range(2)-input_range(1)) .* rand(200, 1);
train_targets = 1 ./ train_inputs;

test_inputs = input_range(1) + (input_range(2)-input_range(1)) .* rand(100, 1);
test_targets = 1 ./ test_inputs;

test_inputs = setdiff(test_inputs, train_inputs, 'stable');
test_inputs = test_inputs(1:min(100, numel(test_inputs))); 

rng(1); 
W1 = rand(hidden_neurons, 1) * 0.2 - 0.1;
b1 = rand(hidden_neurons, 1) * 0.2 - 0.1;
W2 = rand(1, hidden_neurons) * 0.2 - 0.1;
b2 = rand(1, 1) * 0.2 - 0.1;

fprintf("The weight of the first layer is in range [%.4g, %.4g]\n", min(W1), max(W1));
fprintf("The bias of the first layer is in range [%.4g, %.4g]\n", min(b1), max(b1));
fprintf("The weight of the second layer is in range [%.4g, %.4g]\n", min(W2), max(W2));
fprintf("The bias of the second layer is in range [%.4g, %.4g]\n", min(b2), max(b2));

epochs = zeros(1, training_epochs);
RMSEs_train = zeros(1, training_epochs);
RMSEs_test = zeros(1, training_epochs);

for learn_step = 1:training_epochs
    dW1_accum = zeros(size(W1));
    db1_accum = zeros(size(b1));
    dW2_accum = zeros(size(W2));
    db2_accum = zeros(size(b2));
    error_accum = 0;
    
    for i = 1:batch_size
        idx = randi([1, size(train_inputs, 1)]);
        input = train_inputs(idx);
        target = train_targets(idx);
        
        hidden_output = tanh(W1 * input + b1);
        output = tanh(W2 * hidden_output + b2);
        output = (output + 1) * (9 / 2) + 1;
        
        error = target - output;
        error_accum = error_accum + error^2;
        
        dW2 = error * (1 - tanh(W2 * hidden_output + b2).^2) * hidden_output';
        db2 = error * (1 - tanh(W2 * hidden_output + b2).^2);
        dW1 = ((1 - hidden_output.^2) .* (W2' * (error * (1 - tanh(W2 * hidden_output + b2).^2)))) * input';
        db1 = (1 - hidden_output.^2) .* (W2' * (error * (1 - tanh(W2 * hidden_output + b2).^2)));
        
        dW1_accum = dW1_accum + dW1;
        db1_accum = db1_accum + db1;
        dW2_accum = dW2_accum + dW2;
        db2_accum = db2_accum + db2;
    end
    
    W1 = W1 + learning_rate * dW1_accum / batch_size;
    b1 = b1 + learning_rate * db1_accum / batch_size;
    W2 = W2 + learning_rate * dW2_accum / batch_size;
    b2 = b2 + learning_rate * db2_accum / batch_size;
    
    % Calculate test error
    error_test = 0;
    for i=1:length(test_inputs)
        test_input = test_inputs(i);
        test_target = test_targets(i);
        hidden_output = tanh(W1 * test_input + b1);
        output = tanh(W2 * hidden_output + b2);
        output = (output + 1) * (9 / 2) + 1;
    
        error = test_target - output;
        error_test = error_test + error^2;
    end
    epochs(learn_step) = learn_step;
    RMSEs_train(learn_step) = sqrt(error_accum / batch_size);
    RMSEs_test(learn_step) = sqrt(error_test / length(test_inputs));
    % fprintf('Epoch: %d, RMSE: %f\n', learn_step, sqrt(error_accum / batch_size));
end

% Plot the training curve
figure(1);
hold on;
plot(epochs, RMSEs_train);
plot(epochs, RMSEs_test);
xlabel("Epoch");
ylabel("RMSE");
legend("Training", "Test");
title("Training curve");
grid on;


%% Difference
predictions_train = zeros(1, length(train_targets));
predictions_test = zeros(1, length(test_targets));

for i = 1:size(train_inputs, 1)
    input = train_inputs(i);
    hidden_output = tanh(W1 * input + b1);
    output = tanh(W2 * hidden_output + b2);
    output = (output + 1) * (9 / 2) + 1;
    predictions_train(i) = output;
end
for i = 1:size(test_inputs, 1)
    input = test_inputs(i);
    hidden_output = tanh(W1 * input + b1);
    output = tanh(W2 * hidden_output + b2);
    output = (output + 1) * (9 / 2) + 1;
    predictions_test(i) = output;
    % fprintf('Input: %f, Target: %f, Output: %f\n', input, test_targets(i), output);
end
figure(2);
scatter(train_targets, predictions_train);
hold on;
scatter(test_targets, predictions_test);
xlabel("Desired outputs");
ylabel("Actual outputs");
legend("Training", "Test");
title("Prediction Performance");
grid on;