% Generate uniform random points inside a circle
close all
clear all
clc

% c: center [cx cy]
% r: radius
% N: number of random samples

c1 = [0.5 0];   % example center
r1 = 1;       % example radius
N = 50;     % number of points to generate

% Random angle uniformly distributed between 0 and 2?
theta1 = 2*pi*rand(N,1);

% Correct uniform radius sampling: sqrt ensures uniform area distribution
rho1 = r1 * sqrt(rand(N,1));

% Generate points inside the circle
x1 = c1(1) + rho1 .* cos(theta1);
y1 = c1(2) + rho1 .* sin(theta1);


c2 = [-1 -1];   % example center
r2 = 0.5;       % example radius
N = 50;     % number of points to generate

% Random angle uniformly distributed between 0 and 2?
theta2 = 2*pi*rand(N,1);

% Correct uniform radius sampling: sqrt ensures uniform area distribution
rho2 = r2 * sqrt(rand(N,1));

% Generate points inside the circle
x2 = c2(1) + rho2 .* cos(theta2);
y2 = c2(2) + rho2 .* sin(theta2);



c3 = [-1 1];   % example center
r3 = 0.5;       % example radius
N = 50;     % number of points to generate

% Random angle uniformly distributed between 0 and 2?
theta3 = 2*pi*rand(N,1);

% Correct uniform radius sampling: sqrt ensures uniform area distribution
rho3 = r3 * sqrt(rand(N,1));

% Generate points inside the circle
x3 = c3(1) + rho3 .* cos(theta3);
y3 = c3(2) + rho3 .* sin(theta3);


c4 = [2 1];   % example center
r4 = 0.5;       % example radius
N = 50;     % number of points to generate

% Random angle uniformly distributed between 0 and 2?
theta4 = 2*pi*rand(N,1);

% Correct uniform radius sampling: sqrt ensures uniform area distribution
rho4 = r4 * sqrt(rand(N,1));

% Generate points inside the circle
x4 = c4(1) + rho4 .* cos(theta4);
y4 = c4(2) + rho4 .* sin(theta4);

% Visualization
figure; hold on; axis equal; grid on;
scatter(x1, y1, 20, 'filled');
viscircles(c1, r1, 'Color','r'); % circle outline for reference

scatter(x2, y2, 20, 'filled');
viscircles(c2, r2, 'Color','k'); % circle outline for reference

scatter(x3, y3, 20, 'filled');
viscircles(c3, r3, 'Color','g'); % circle outline for reference

scatter(x4, y4, 20, 'filled');
viscircles(c4, r4, 'Color','b'); % circle outline for reference



title('Uniform Random Points Inside a Circle');
xlabel('x'); ylabel('y');


%% ADDED CODE

% ------------------------------------------------------------------------
% Tüm noktaları tek bir veri matrisi halinde birleştir
% ------------------------------------------------------------------------
X = [x1 y1;
     x2 y2;
     x3 y3;
     x4 y4];     % 200 x 2

% K-means parametreleri
k = 4;
maxIter = 100;

[idx, C, J_hist] = my_kmeans(X, k, maxIter);

% Sonucu görselleştir
figure; hold on; grid on; axis equal;
gscatter(X(:,1), X(:,2), idx);     % kümelere göre renklendir
plot(C(:,1), C(:,2), 'kx', 'MarkerSize', 12, 'LineWidth', 2);
title('k-means clustering result');
xlabel('x'); ylabel('y');
legend('Cluster 1','Cluster 2','Cluster 3','Cluster 4','Centroids','Location','best');

% Maliyet fonksiyonu (within-cluster sum of squares) iterasyona göre
figure; 
plot(J_hist,'-o'); grid on;
xlabel('Iteration'); ylabel('J = \Sigma ||x_i - c_{k(i)}||^2');
title('k-means objective vs iteration');
