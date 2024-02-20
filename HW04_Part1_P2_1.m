
input_neurons = 2;
hidden_neurons = 2;
output_neurons = 1;

rng(1);
W1 = rand(hidden_neurons, input_neurons) * 2 - 1;
b1 = rand(hidden_neurons, 1) * 2 - 1; 
W2 = rand(output_neurons, hidden_neurons) * 2 - 1; 
b2 = rand(output_neurons, 1) * 2 - 1; 

learning_rate = 0.1;
epochs = 10000;
epsilon = 0.01;

X = [0 0; 0 1; 1 0; 1 1]; 
T = [0; 1; 1; 0]; 

for epoch = 1:epochs
    for i = 1:size(X, 1)
        input = X(i, :)';
        target = T(i);

        hidden_input = W1 * input + b1;
        hidden_output = tanh(hidden_input);

        output_input = W2 * hidden_output + b2;
        output = tanh(output_input);

        error = target - output;

        dW2 = learning_rate * error * (1 - output.^2) * hidden_output';
        db2 = learning_rate * error * (1 - output.^2);

        delta1 = (W2' * (error * (1 - output.^2))) .* (1 - hidden_output.^2);
        dW1 = learning_rate * delta1 * input';
        db1 = learning_rate * delta1;

        W1 = W1 + dW1;
        b1 = b1 + db1;
        W2 = W2 + dW2;
        b2 = b2 + db2;
    end

    mse = mean(error.^2);
    if mse < epsilon
        fprintf('Training completed at epoch %d with MSE: %f\n', epoch, mse);
        break;
    end
end

for i = 1:size(X, 1)
    input = X(i, :)';
    hidden_output = tanh(W1 * input + b1);
    output = tanh(W2 * hidden_output + b2);
    fprintf('Input: [%d %d], Output: %f\n', X(i, 1), X(i, 2), output);
end
