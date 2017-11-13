function x = omp(A, b, k)
% OMP Solve the P0 problem via OMP
%
% Solves the following problem:
%   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
%
% The solution is returned in the vector x

% Reference implementation: slide 7 of SubSection_03_01_Part2.pdf

%% Initialization

% initialize the vector x with 0
m = size(A,2);
x = zeros(m,1);

% residual error
r = b-A*x;

% support
support = [];

%% Main iteration
for i = 1:k
% Step 1: compute all the errors
E = A'*r;

% Step 2: select the next atom,
[~, next_atom] = max(abs(E));

% Step 3: update the support
support = sort([support next_atom]);

% Step 4: solve the LS problem and update the provisional solution
A_s = A(:, support);
x_i = pinv(A_s)*b;
x(support,:)=x_i;

% Step 5: update the residual error
r = b-A*x;
end

end