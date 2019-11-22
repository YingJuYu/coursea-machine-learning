function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
% X = 97x2
% y = 97x1
% theta = 2x1
% h(theta, X)_i = 1 +theta*X_i;
% sum = sigma (h(theta, X)_i-y_i)*(h(theta, X)_i-y_i)
% J = 1/(2*m)*sum

%Solution1:
h = theta'*X'; %1x2 * 2x97 = 1x97
J = 1./(2*m)*sum((h' - y).*(h' - y));

%Solution2:
%h = theta(1)*X(:,1) + theta(2)*X(:,2); %97*1
%J = 1./(2*m)*sum((h - y).*(h - y));


% =========================================================================

end
