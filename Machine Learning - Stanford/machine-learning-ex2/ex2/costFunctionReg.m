function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
% J = 0;
% grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


h = sigmoid(X * theta);
posCost = y' * log( h );
negCost = (1 - y)' * log(1 - h);

thetaX = theta(2:end);
regTerm = (lambda / (2*m)) * thetaX' * thetaX;

J = (1 / m) * (-posCost - negCost) + regTerm;

thetaSize = size(theta, 1);
addCost = ones(thetaSize, 1);
addCost(1) = 0;
addCost = (addCost .* theta) * lambda / m;

grad = (1 / m) * X' * (h - y) + addCost;




% =============================================================

end
