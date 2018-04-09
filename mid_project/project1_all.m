% Cleanup
clear;
clc;

% In this project we demonstrate the OMP and BP algorithms, by running them
% on a set of signals and checking whether they provide the desired outcome

%% Parameters

% Set the length of the signal
n = 50;

% Set the number of atoms in the dictionary
m = 100;

% Set the maximum number of non-zeros in the generated vector
s_max = 15;

% Set the minimal entry value
min_coeff_val = 1;

% Set the maximal entry value
max_coeff_val = 3;

% Number of realizations
num_realizations = 200;

% Base seed: A non-negative integer used to reproduce the results
% Set an arbitrary value for base seed
base_seed = 2;

%% Create the dictionary

% Create a random matrix A of size (n x m)
A=randn(n,m);

% Normalize the columns of the matrix to have a unit norm
% The most elegant solution would be to use 'A_normalized=normc(A)' but it is part of the Neural Network Toolbox
A_normalized = A;
for j=1:m
    A_normalized(:,j) = A(:,j)/norm(A(:,j));
end

%% Create data and run OMP and BP

% Nullify the entries in the estimated vector that are smaller than eps
eps_coeff = 1e-4;
% Set the optimality tolerance of the linear programing solver
tol_lp = 1e-4;

% Allocate a matrix to save the L2 error of the obtained solutions
L2_error = zeros(s_max,num_realizations,2);
% Allocate a matrix to save the support recovery score
support_error = zeros(s_max,num_realizations,2);

% Loop over the sparsity level
for s = 1:s_max

    % Use the same random seed in order to reproduce the results if needed
    rng(s+base_seed)

    % Loop over the number of realizations
    for experiment = 1:num_realizations

        % In this part we will generate a test signal b = A_normalized*x by
        % drawing at random a sparse vector x with s non-zeros entries in
        % true_supp locations with values in the range of [min_coeff_val, max_coeff_val]
        x = zeros(m,1);

        % Draw at random a true_supp vector
        % i.e. we need s random indexes of the vector x
        permutation = randperm(m);
        true_supp = permutation(1:s);

        % Draw at random the coefficients of x in true_supp locations
        % i.e. the value of each entry is taken from a uniform distribution in [min_coeff_val, max_coeff_val] and multiplied by a random sign.
        % The most elegant solution for the sign would be to use randsrc but it is part of the Communication System Toolbox
        x(true_supp) = sign(randn(s,1)).*(min_coeff_val + (max_coeff_val - min_coeff_val)*rand(s,1));
        x = sparse(x);

        % Create the signal b
        b = A_normalized*x;

        % Run OMP for s iterations
        x_omp = sparse(omp(A_normalized, b, s));

        % Compute the relative L2 error
        L2_error(s,experiment,1) = norm(x_omp-x)^2/norm(x)^2;

        % Get the indices of the estimated support
        estimated_supp = find(x_omp);

        % Compute the support recovery error
        support_error(s,experiment,1) = 1-length(intersect(estimated_supp, true_supp))/max(length(estimated_supp), length(true_supp));

        % Run BP
        x_lp = sparse(lp(A_normalized, b, tol_lp));

        % Compute the relative L2 error
        L2_error(s,experiment,2) = norm(x_lp-x)^2/norm(x_lp)^2;

        % Get the indices of the estimated support, where the
        % coefficients are larger (in absolute value) than eps_coeff
        estimated_supp = find(abs(x_lp)>eps_coeff);

        % Compute the support recovery error
        support_error(s,experiment,2) = 1-length(intersect(estimated_supp, true_supp))/max(length(estimated_supp), length(true_supp));
    end

end

%% Display and print the results

% Plot the average relative L2 error, obtained by the OMP and BP versus the cardinality
figure(1); clf;
plot(1:s_max,mean(L2_error(1:s_max,:,1),2),'r','LineWidth',2); hold on;
plot(1:s_max,mean(L2_error(1:s_max,:,2),2),'g','LineWidth',2);
xlabel('Cardinality of the true solution');
ylabel('Average and relative L_2-error');
set(gca,'FontSize',14);
legend({'OMP','LP'});
axis([0 s_max 0 1]);
title('L_2-error vs. cardinality');
print('L2_vs_cardinality','-depsc');

% Plot the average support recovery score, obtained by the OMP and BP versus the cardinality
figure(2); clf;
plot(1:s_max,mean(support_error(1:s_max,:,1),2),'r','LineWidth',2); hold on;
plot(1:s_max,mean(support_error(1:s_max,:,2),2),'g','LineWidth',2);
xlabel('Cardinality of the true solution');
ylabel('Probability of error in support');
set(gca,'FontSize',14);
legend({'OMP','LP'});
axis([0 s_max 0 1]);
title('Average support recovery vs. cardinality');
print('support_vs_cardinality', '-depsc');
