function x = omp(A, b, iterations)
% OMP Solve the P0 problem via OMP
%
% Solves the following problem:
%   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
%

%% Initialization
fprintf('Initialization\n');
% solution vector
% size(A,1) = n rows
% size(A,2) = m columns
% x column vector of size m x 1
x = zeros(size(A,2),1)

% residual error
r = b-A*x

% support
S = []

for k=0:iterations-1
%% Main iteration

% Step 1&2: choose the next atom
[~, next_atom] = max(abs(A'*r))

% Step 3: update the support
S = [S next_atom]

% Step 4: LS / update the current solution
A_s = A(:, S)
x_k = pinv(A_s)*b
x(S,:)=x_k

% Step 5: update residual
r = b-A*x

% Debugging
fprintf('k=%d r=%f\n', k, norm(r));
end

end
