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
K = num_labels;         
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

% Part 1: Compute cost without regularization
    % Construct y matrix
    Y = recode(y, m, K); 
    %fprintf("Size of Y\n")
    %size(Y)
    % Add 1 to each example
    X = [ones(m, 1) X];

    %fprintf("Size of X\n")
    %size(X)
    %fprintf("Size of Theta1\n")
    %size(Theta1)

    A_2 = sigmoid( X*(Theta1.'));
    A_2 = [ones(m, 1) A_2];
    

    %fprintf("Size of A_2")
    %size(A_2)
    %fprintf("Size of Theta2")
    %size(Theta2)

    H = sigmoid(A_2*(Theta2.'));
    %H = A_2*(Theta2.');

    
    % Cost matrix (each row is an example, with K elements
    cost = Y.*log(H) + (1-Y).*log(1-H);

    % Sum over each row to get a column of all examples (dim =m)
    cost_example = sum(cost, 2);

    % Sum the column to get a scalar 
    cost_total = sum(cost_example, 1);

    % Normalize 
    J = -cost_total/m;



    % Regularized
    Theta1_reg = Theta1 (:, 2:end); %remove bias term from each row
    Theta2_reg = Theta2 (:, 2:end); %remove bias term from each row
    sum_1 = sum( sum( Theta1_reg.^2 , 1), 2);
    sum_2 = sum( sum( Theta2_reg.^2 , 1), 2);
    reg = lambda/(2*m)* (sum_1+sum_2);
    J = J + reg;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end


function Y = recode (y, m,K)
    Y = zeros(m,K);
    for i = 1:m
        Y (i, y(i,1)) = 1;
    end
end
