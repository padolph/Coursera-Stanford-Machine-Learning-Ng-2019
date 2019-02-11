function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

arg = X * theta;
h = 1./(1. + exp(-arg));
n = size(X,2); % number of features + 1
thetaLessTheta0 = theta(2:n,:);  % removes first 'row' from theta vector (theta-sub-0)
J = (-1./m) * ( y' * log(h) + (1 - y') * log(1 - h) ) + (lambda/(2.*m))*(thetaLessTheta0'*thetaLessTheta0);

thetaWithZeroTheta0 = [0;thetaLessTheta0];
grad = (1./m) * X' * (h - y) + (lambda/m) * thetaWithZeroTheta0;


% =============================================================

end
