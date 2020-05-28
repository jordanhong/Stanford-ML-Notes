function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
for iter = 1:num_iters

    old_theta = theta;
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    %this is doing the arithmetic per row and summing over entire column
    delta_1 = sum((old_theta(1) + old_theta(2) .* X(:,2)) - y); % Un-Vectorized 
    theta(1) = old_theta(1) - alpha/m*delta_1;
    delta_2 = sum( ((old_theta(1) + old_theta(2).*X(:,2)) -y).*X(:,2) );
    theta(2) = old_theta(2) -alpha/m*delta_2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
    % Checking point: plot J_history to see if error plateus.
    % plot(J_history)
end
