function x = lp(A, b, tol)
% LP Solve Basis Pursuit via linear programing
%
% Solves the following problem:
%   min_x || x ||_1 s.t. b = Ax
%
% The solution is returned in the vector x.

% Set the options to be used by the linprog solver
options = optimoptions('linprog','Algorithm','dual-simplex',...
    'Display','none','OptimalityTolerance',tol);

% Use Matlab's linprog function to solve the BP problem
m = size(A,2);
f = ones(2*m,1);
x = linprog(f,[],[],[A, -A],b,0*f,3*f,options);
x = x(1:m)-x(m+1:end);

end