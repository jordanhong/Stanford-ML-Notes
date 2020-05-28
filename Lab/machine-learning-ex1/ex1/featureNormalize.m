function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

%====================== A NOTE ABOUT SIZE()==================
% M = SIZE(X,DIM) returns the lengths of the specified dimensions in a 
%    row vector. DIM can be a scalar or vector of dimensions. For example, 
%    SIZE(X,1) returns the number of rows of X and SIZE(X,[1 2]) returns a 
%    row vector containing the number of rows and columns.
%
% Here, size(X, 2) returns a scalar with value = number of columns.
% The number of columns here denote the number of features.
%
%============================================================


% You need to set these values correctly
%X_norm = X;
%mu = zeros(1, size(X, 2)); % mu is a row vector. each column in the vector is the mean value of a feature 
%sigma = zeros(1, size(X, 2)); % sigma is a row vector with each column taking the value of the standard deviation of each feature. 

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       


% Vectorized 
% Get the mean of a 2xm matrix in a row
mu = mean(X);
sigma = std(X);

X_norm = (X-mu)./sigma;





% ============================================================

end
