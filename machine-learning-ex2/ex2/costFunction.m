function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

n = size(X,2);  % number of features + 1

%%% Cost Function %%%

if (0)
    %make an m x 1 vector that is h evaluated for each row of X
    h = zeros(m,1);
    for i = 1:m
        x = X(i,:)';  % ith row of X, turned into a vector
        arg = theta' * x;
        h(i) = 1./(1. + exp(-arg));
    end

    %now make an m x 1 vector whose elements are each one value we want sum up
    addmeup = y.*log(h) + (ones(m,1)-y).*log(1-h);

    %ready to compute the cost function now
    J = (-1./m) * sum(addmeup);
else
    arg = X * theta;
    h = 1./(1. + exp(-arg));
    J = (-1./m) * ( y' * log(h) + (1 - y') * log(1 - h) );
end

%%% Gradient %%%

if (0)
    grad = zeros(n,1);
    for j = 1:n
        d = h-y;
        v = d'*X(:,j);
        grad(j) = (1.0/m)*sum(v);
    end
else  
    grad = (1./m) * X' * (h - y);  % Similar as ex1 gradDescentMulti.m
end

% =============================================================

end
