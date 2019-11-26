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

%size(X) % 5000 x 400 --> totalTraingingExample, m x # input features, X
%size(y) % 5000 x 1   --> totalTraingingExample, m x # output, y
%Training Sets : X(m,1:401);

% reorder y outputs to Y : 10 x 5000
Y = zeros(num_labels, m); %10 x 5000
for i = 1 : m
  Y(y(i),i) = 1;
endfor


%Adding a0^(1) = 1 to all trainging examples, then transpose a^(1)
a1 = [ones(m, 1) X]'; % a^(1): (input feature + 1) x totalTraingingExample
%Calculate Theta1 (25 x 401) * a^(1) (401 x 5000)
z2 = Theta1 * a1;
a2 = sigmoid(z2); %25 x 5000

a2 = [ones(m, 1) a2']'; % a^(2): 26 x 5000
%Calculate Theta2 (10 x 26) * a^(2) (26 x 5000)
z3 = Theta2 * a2;
a3 = sigmoid(z3); %10 x 5000 ===> h_theta(X)

h = a3;

J = sum(sum((-Y).*log(h) - (1-Y).*log(1-h), 2))/m % column-wise operation
%for i = 1 : m
%J = J + 1/m*(-Y(:,i)' * log(h(:,i)) - ...
%(1 - Y(:,i))' * log(1 - h(:,i)));
%endfor

%Regulazation term

Jreg = lambda/(2*m)*( sum(sum(Theta1(:,2:end).^2, 2)) + sum(sum(Theta2(:,2:end).^2, 2)));

J = J + Jreg;
% ==============================================================================
%Gradient

delta3 = a3 .- Y; %10 x 5000

z2 = z2'; %25 x 5000 --> 5000 x 25
z2 = [ones(size(z2,1), 1) z2]; %5000 x 26
delta2 = Theta2'*delta3 .* sigmoidGradient(z2)';
% size(delta2) 26 x 5000

Delta2 = zeros(size(delta3*a2'));
Delta2 = Delta2 + delta3*a2'; % 10x 26

Delta1 = zeros(size(delta2*a1'));
Delta1 = Delta1 + delta2*a1';
Delta1
Theta1_grad = Delta1./m + lambda.*Theta1';
Theta2_grad = Delta2./m + lambda.*Theta2';
% 2019/11/26

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
