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

sum = 0;
for i = 1:m

sum = sum + (-y(i)*log(sigmoid(X(i, :)*theta)) - (1-y(i))*log(1-sigmoid(X(i,:)*theta)));

end
n = size(theta);
reg = 0;
for j = 2:n
    reg = reg +  theta(j) * theta(j);
end

J = 1/m * sum + lambda/(2*m)*reg;

for j = 1:size(theta)
    sum = 0;
    for k = 1:m
        sum = sum + (sigmoid(X(k,:)*theta)-y(k))*X(k, j);
    end
    if j == 1
        grad(j) = 1/m * sum;
    else
        grad(j) = 1/m * sum + lambda/m*theta(j);
end




% =============================================================

end
