for learn_step = 1:max_learn_steps

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
        
        error = target - output;
        error_accum = error_accum + error^2;
        
        dW2 = error * (1 - output^2) * hidden_output';
        db2 = error * (1 - output^2);
        dW1 = ((1 - hidden_output.^2) .* (W2' * (error * (1 - output.^2)))) * input';
        db1 = (1 - hidden_output.^2) .* (W2' * (error * (1 - output.^2)));
        
        dW1_accum = dW1_accum + dW1;
        db1_accum = db1_accum + db1;
        dW2_accum = dW2_accum + dW2;
        db2_accum = db2_accum + db2;
    end
    
    W1 = W1 + learning_rate * dW1_accum / batch_size;
    b1 = b1 + learning_rate * db1_accum / batch_size;
    W2 = W2 + learning_rate * dW2_accum / batch_size;
    b2 = b2 + learning_rate * db2_accum / batch_size;
    
    if mod(learn_step, 1000) == 0
        fprintf('Learning step: %d, MSE: %f\n', learn_step, error_accum / batch_size);
    end
end
