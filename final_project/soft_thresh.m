function x = soft_thresh(z, lambda)

% Reference implementation: slide 8 of Subsection_04_01_Part6.pdf

% Find the elements that are less than -lambda
k = find(z <= -lambda);
x(k) = z(k) + lambda;

% Find then elements which absolute values are less than lambda
k = find(abs(z) < lambda);
x(k) = 0; 

% Find the elements that are larger than +lambda
k = find(z >= lambda);
x(k) = z(k) - lambda;

x = x(:);