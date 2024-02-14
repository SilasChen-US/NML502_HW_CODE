% Problem setup
hidden_neurons = 10; % Use 10 hidden PEs as specified
learning_rate = 0.05; % Set learning rate to 0.05
batch_size = 100; % Set batch size to 100
max_learn_steps = 5000 * batch_size; % Set maximum learning steps
input_range = [0.1, 1.0]; % Input range for the function
output_scale = 1.0; % Maximum value for scaling the output

% Generate training data
train_inputs = input_range(1) + (input_range(2)-input_range(1)) .* rand(200, 1);
train_targets = 1 ./ train_inputs;
train_targets = train_targets / output_scale; % Scale the targets

% Generate test data
test_inputs = input_range(1) + (input_range(2)-input_range(1)) .* rand(100, 1);
test_targets = 1 ./ test_inputs;
test_targets = test_targets / output_scale; % Scale the targets

% Ensure test data does not coincide with training data
test_inputs = setdiff(test_inputs, train_inputs, 'stable');
test_inputs = test_inputs(1:100); % Ensure we have 100 unique test samples

% Initialize weights and biases
rng(1); % Seed for reproducibility
W1 = rand(hidden_neurons, 1) * 0.2 - 0.1;
b1 = zeros(hidden_neurons, 1);
W2 = rand(1, hidden_neurons) * 0.2 - 0.1;
b2 = 0;

% Batch learning
for learn_step = 1:max_learn_steps
    % Accumulate gradients for a batch
    dW1_accum = zeros(size(W1));
    db1_accum = zeros(size(b1));
    dW2_accum = zeros(size(W2));
    db2_accum = zeros(size(b2));
    error_accum = 0;
    
    % Start of the batch
    for i = 1:batch_size
        % Randomly pick a sample from the training set
        idx = randi([1, size(train_inputs, 1)]);
        input = train_inputs(idx);
        target = train_targets(idx);
        
        % Forward pass
        hidden_output = tanh(W1 * input + b1);
        output = tanh(W2 * hidden_output + b2);
        
        % Compute error
        error = target - output;
        error_accum = error_accum + error^2; % Accumulate squared error
        
        % Backpropagation to accumulate gradients
        dW2 = error * (1 - output^2) * hidden_output';
        db2 = error * (1 - output^2);
        dW1 = ((1 - hidden_output.^2) .* (W2' * (error * (1 - output.^2)))) * input';
        db1 = (1 - hidden_output.^2) .* (W2' * (error * (1 - output.^2)));
        
        dW1_accum = dW1_accum + dW1;
        db1_accum = db1_accum + db1;
        dW2_accum = dW2_accum + dW2;
        db2_accum = db2_accum + db2;
    end
    
    % Update weights after each batch
    W1 = W1 + learning_rate * dW1_accum / batch_size;
    b1 = b1 + learning_rate * db1_accum / batch_size;
    W2 = W2 + learning_rate * dW2_accum / batch_size;
    b2 = b2 + learning_rate * db2_accum / batch_size;
    
    % Print progress every 1000 batches
    if mod(learn_step, 1000) == 0
        fprintf('Learning step: %d, MSE: %f\n', learn_step, error_accum / batch_size);
    end
end

% Test the trained network with the test set
for i = 1:size(test_inputs, 1)
    input = test_inputs(i);
    hidden_output = tanh(W1 * input + b1);
    output = tanh(W2 * hidden_output + b2);
    fprintf('Input: %f, Target: %f, Output: %f\n', input, test_targets(i), output);
end
