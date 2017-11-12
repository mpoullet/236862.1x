function x = omp(A, b, k)
% OMP Solve the P0 problem via OMP
%
% Solves the following problem:
%   min_x ||b - Ax||_2^2 s.t. ||x||_0 <= k
%
% The solution is returned in the vector x

% Precondition: we suppose that the columns of A are normalized
% Reference implementation: slide 7 of SubSection_03_01_Part2.pdf

%% Initialization
fprintf('OMP: initialization, k=%d\n', k);

% initialize the vector x with 0
n = size(A,1);
m = size(A,2);
x = zeros(m,1);

% residual error
r = b-A*x;

% support
support = [];

% iteration index
i = 1;

while i <= k
%% Main iteration

% Step 1&2: choose the next atom
[~, next_atom] = max(abs(A'*r));

% Step 3: update the support
support = sort([support next_atom]);

% Step 4: solve the LS problem and update the solution
A_s = A(:, support);
x_i = pinv(A_s)*b;
x(support,:)=x_i;

% Step 5: update the residual error
r = b-A*x;

% Debugging
fprintf('i=%d norm(r)=%f\n', i, norm(r));
i = i+1;
end

end