clear; clc; close all;

% Sistem matrisleri
A = [-1.0455  4.7727  -9.6364  -4.6818;
      1.1822  0.9081   1.5452  -4.2726;
     -3.4090  7.9542 -22.7274  -1.1363;
      4.5910  7.9542  -5.7274 -18.1363];

B = [0.5 0 0 0];
C = [1   0 0 0];

% X = [A  B'  C']  -> 4x6
X = [A B.' C.'];

% Ekonomik SVD: U 4x4, S 4x4, V 6x4
[U,S,V] = svd(X);

% (a) İlk iki anlamlı tekil değer
sigma = diag(S);
fprintf('Singular values:\n');
disp(sigma);

% Rekonstrüksiyon hatası e_X(r) için r = 1..4
rmax = size(S,1);   % = 4
eX = zeros(rmax,1);

for r = 1:rmax
    X_r = U(:,1:r) * S(1:r,1:r) * V(:,1:r).';
    eX(r) = norm(X - X_r, 'fro');
end

figure;
plot(1:rmax, eX, 'o-','LineWidth',1.5); grid on;
xlabel('r (kullanılan tekil değer sayısı)');
ylabel('e_X(r) = ||X - X_r||_F');
title('SVD based X reconstruction error');

% (b) Sadece ilk iki tekil değerle A tahmini
rA = 2;
X_2 = U(:,1:rA) * S(1:rA,1:rA) * V(:,1:rA).';
A_est = X_2(:,1:4);    % ilk 4 sütun A'ya karşılık geliyor

fprintf('A matrisinin 2 tekil değerle tahmini (A_est):\n');
disp(A_est);

% İstersen gerçek A ile farkını da yazdır:
fprintf('||A - A_est||_F = %.4f\n', norm(A - A_est, 'fro'));
