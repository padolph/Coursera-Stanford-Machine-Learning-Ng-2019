function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% Feedforward model

% input layer
X1 = [ones(m,1), X];

% hidden layer
z2 = X1 * Theta1';
a2tmp = sigmoid(z2);
a2 = [ones(m, 1), a2tmp];

% output layer
z3 = a2 * Theta2';
a3 = sigmoid(z3);
h = a3;

% output recoded as binary vectors collected into a matrix; use the value
% of y as the column index to set to 1
yBin = zeros(m, num_labels);
for i = 1:m
    yBin(i, y(i)) = 1;
end

%%  cost function
% old: J = (-1./m) * ( y' * log(h) + (1 - y') * log(1 - h) );
J = (1./m) * sum(sum(-yBin .* log(h) - (1 - yBin) .* log(1-h)));

% add the cost for the regularization terms

% first compute new theta matrices without the bias terms
Theta1nb = Theta1(:,2:end);
Theta2nb = Theta2(:,2:end);

r = (sum(sum(Theta1nb .^ 2)) + sum(sum(Theta2nb .^ 2))) * lambda / (2.*m);

J = J + r;

%% backprop gradient

% for each training example
for t = 1:m
    % step 1 feedforward pass
    %input
    a1 = X(t,:)';  % pull off the t'th row of X and make it a vector
    a1 = [1; a1];  % add bias term
    %hidden layer
    z2 = Theta1 * a1;
    a2 = sigmoid(z2);
    a2 = [1; a2];  % add bias term
    %output
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    %step 2 output error (layer 3)
    yt = yBin(t,:)'; % pull off the t'th answer and make it a vector
    d3 = a3 - yt;
    
    %step 3 hidden layer error (layer 2)
    d2 = (Theta2' * d3) .* sigmoidGradient([1;z2]);
    
    %step 4 accumulate the gradient from this example
    d2 = d2(2:end); % remove d0 bias term
    Theta2_grad = Theta2_grad + d3 * a2';
    Theta1_grad = Theta1_grad + d2 * a1';
end

%step 5 obtain the (unreg) gradient for the neural network cost function
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

% Add regularization to the gradient, skipping the bias column
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m) * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m) * Theta2(:,2:end);

%% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
