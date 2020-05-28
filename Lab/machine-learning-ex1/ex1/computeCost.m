function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y


% ===================== Notes ====================
% X is a m by 2 matrix. Each row of X is [1, x_i]. 
% Note that this is different than the representation in the lecture slides,
% where a training sample x is represented by a column matrix [1, x_1, ..., x_n]^T

% theta is a 2 by 1 matrix. 
% Instead of doing theta^T x X^T, here I do X x theta
% ===============================================
% Initialize some useful values
m = length(y); % number of training examples
% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

for i=1:m
    % Here deal with X and y generally
    J = J + ( X(i, :)*theta - y(i, :) )^2;
end

% J is a scalar
J = J /(2*m);





% =========================================================================

end
