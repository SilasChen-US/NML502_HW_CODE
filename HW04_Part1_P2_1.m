% Define the architecture of the network
input_neurons = 2; % For XOR input
hidden_neurons = 2; % Can be varied
output_neurons = 1; % For XOR output

% Initialize weights and biases
% For reproducibility, set the random seed or use predetermined values
rng(1); % Seed random number generator for reproducibility
W1 = rand(hidden_neurons, input_neurons) * 2 - 1; % Input to hidden weights
b1 = rand(hidden_neurons, 1) * 2 - 1; % Hidden layer biases
W2 = rand(output_neurons, hidden_neurons) * 2 - 1; % Hidden to output weights
b2 = rand(output_neurons, 1) * 2 - 1; % Output layer biases

% Learning parameters
learning_rate = 0.1;
epochs = 10000;
epsilon = 0.01;

% XOR Problem Setup
X = [0 0; 0 1; 1 0; 1 1]; % Inputs
T = [0; 1; 1; 0]; % Targets

% Begin training
for epoch = 1:epochs
    for i = 1:size(X, 1)
        % Forward pass
        input = X(i, :)';
        target = T(i);

        % Calculate activations for hidden layer
        hidden_input = W1 * input + b1;
        hidden_output = tanh(hidden_input);

        % Calculate activations for output layer
        output_input = W2 * hidden_output + b2;
        output = tanh(output_input);

        % Compute error
        error = target - output;

        % Backpropagation
        % Output to hidden layer weights
        dW2 = learning_rate * error * (1 - output.^2) * hidden_output';
        db2 = learning_rate * error * (1 - output.^2);

        % Hidden to input layer weights
        % dW1 = learning_rate * ((1 - hidden_output.^2) * (W2' * error)) * input';
        % db1 = learning_rate * ((1 - hidden_output.^2) * (W2' * error));
        delta1 = (W2' * (error * (1 - output.^2))) .* (1 - hidden_output.^2);
        dW1 = learning_rate * delta1 * input';
        db1 = learning_rate * delta1;

        % Weights update
        W1 = W1 + dW1;
        b1 = b1 + db1;
        W2 = W2 + dW2;
        b2 = b2 + db2;
    end

    % Compute mean squared error for epoch
    mse = mean(error.^2);
    if mse < epsilon
        fprintf('Training completed at epoch %d with MSE: %f\n', epoch, mse);
        break;
    end
end

% Test the trained network with XOR input
for i = 1:size(X, 1)
    input = X(i, :)';
    hidden_output = tanh(W1 * input + b1);
    output = tanh(W2 * hidden_output + b2);
    fprintf('Input: [%d %d], Output: %f\n', X(i, 1), X(i, 2), output);
end
