%% KON447E - Assignment I - Question 3
clear; clc; close all;

%% Sistem tanımı
% y(n+1) = 1.2 * (1 - 0.8*exp(-0.1*n)) * y(n)/(1 + y(n)^2) + u(n)

N = 1000;                % Toplam örnek sayısı
u = rand(N,1);           % u(n) ~ U[0,1]
y = zeros(N,1);          % y(1) = 0 başlangıç

%% (a) Sistemi u ve y'yi üret
for n = 1:N-1
    alpha = 1.2 * (1 - 0.8*exp(-0.1*n));
    y(n+1) = alpha * y(n)/(1 + y(n)^2) + u(n);
end

%% (b) Tüm veriyi karıştır, 250 train + 50 test örneği seç ve plot et
% Burada sadece (u,y) ikililerinin dağılımını göstermek için basit bir
% scatter çizimi (ödevdeki örnek grafiğe benzer).

all_data = [u y];
idx_all  = randperm(N);     % 1..N rastgele permütasyon
train_idx_raw = idx_all(1:250);
test_idx_raw  = idx_all(251:300);

train_samples = all_data(train_idx_raw,:);
test_samples  = all_data(test_idx_raw,:);

figure;
scatter(train_samples(:,1), train_samples(:,2), 'b','filled'); hold on;
scatter(test_samples(:,1),  test_samples(:,2),  'r','filled');
xlabel('u'); ylabel('y');
legend('Training samples','Test samples','Location','best');
title('Training vs Test samples (u,y)');
grid on;

%% Yardımcı fonksiyon: regresyon datasını oluştur (NARX)
% ny: y gecikme derecesi, nu: u gecikme derecesi
build_narx = @(y,u,ny,nu) build_regression_narx(y,u,ny,nu);

%% (c) ny = nu = 5 için regresyon datasını hazırla, Theta'yı bul
ny = 5; nu = 5;

[Phi_full, y_next_full] = build_narx(y, u, ny, nu);  % Phi: M x (ny+nu)

M = size(Phi_full,1);           % kullanılabilir örnek sayısı
perm = randperm(M);

train_idx = perm(1:250);
test_idx  = perm(251:300);

Phi_train = Phi_full(train_idx,:);
y_train   = y_next_full(train_idx);

Phi_test  = Phi_full(test_idx,:);
y_test    = y_next_full(test_idx);

% LS çözüm: Theta = (Phi'Phi)^(-1) Phi' y  yerine Phi\y kullanıyoruz
Theta = Phi_train \ y_train;

fprintf('Theta (ny=nu=5) parametre vektoru:\n');
disp(Theta.');

%% (d) Train datası üzerinde tahmin ve modelleme hatası plot
y_train_hat = Phi_train * Theta;
e_train     = y_train - y_train_hat;

figure;
subplot(2,1,1);
plot(y_train, 'r','LineWidth',1.2); hold on;
plot(y_train_hat, 'b--','LineWidth',1.2);
legend('y_{train}','y_{hat}','Location','best');
title('Training data: real vs predicted');
xlabel('Sample index'); grid on;

subplot(2,1,2);
plot(e_train,'k','LineWidth',1.2);
xlabel('Sample index');
ylabel('Error');
title('Training modeling error');
grid on;

%% (e) Test datası performansı
y_test_hat = Phi_test * Theta;
e_test     = y_test - y_test_hat;

figure;
subplot(2,1,1);
plot(y_test, 'r','LineWidth',1.2); hold on;
plot(y_test_hat, 'b--','LineWidth',1.2);
legend('y_{test}','y_{hat}','Location','best');
title('Test data: real vs predicted');
xlabel('Sample index'); grid on;

subplot(2,1,2);
plot(e_test,'k','LineWidth',1.2);
xlabel('Sample index');
ylabel('Error');
title('Test error');
grid on;

%% (f) Farklı ny = nu dereceleri icin train/test MSE ve en iyi modelin bulunması
orders = 1:25;
MSE_train = zeros(length(orders),1);
MSE_test  = zeros(length(orders),1);

for k = 1:length(orders)
    nyk = orders(k);
    nuk = orders(k);

    [Phi_k, y_next_k] = build_narx(y,u,nyk,nuk);
    Mk = size(Phi_k,1);

    perm_k = randperm(Mk);
    idx_train_k = perm_k(1:250);
    idx_test_k  = perm_k(251:300);

    Phi_tr_k = Phi_k(idx_train_k,:);
    y_tr_k   = y_next_k(idx_train_k);

    Phi_te_k = Phi_k(idx_test_k,:);
    y_te_k   = y_next_k(idx_test_k);

    Theta_k = Phi_tr_k \ y_tr_k;

    y_tr_hat_k = Phi_tr_k * Theta_k;
    y_te_hat_k = Phi_te_k * Theta_k;

    MSE_train(k) = mean((y_tr_k - y_tr_hat_k).^2);
    MSE_test(k)  = mean((y_te_k - y_te_hat_k).^2);
end

[~, best_idx] = min(MSE_test);
best_order = orders(best_idx);

fprintf('En iyi model derecesi (ny = nu) = %d\n', best_order);

figure;
plot(orders, MSE_train, 'bo-','LineWidth',1.2); hold on;
plot(orders, MSE_test,  'ro-','LineWidth',1.2);
xlabel('n_u = n_y'); ylabel('Mean squared error');
legend('mean(e_{train}^2)','mean(e_{test}^2)','Location','best');
title('Training & Test MSE vs regressor order');
grid on;

%% ------------------------------------------------------------------------
% Yardimci fonksiyon: NARX regresyon datasini üretir
function [Phi, y_next] = build_regression_narx(y,u,ny,nu)
    % y, u : uzunluk N
    % ny   : y gecikmesi
    % nu   : u gecikmesi
    % Phi  : M x (ny+nu) regresyon matrisi
    % y_next: M x 1 hedef (y(n+1))

    N = length(y);
    maxLag = max(ny,nu);
    % Kullanılabilir son index: N-1 (çünkü y(n+1) var)
    M = N - maxLag;

    Phi = zeros(M, ny+nu);
    y_next = zeros(M,1);

    row = 1;
    for n = maxLag : N-1
        % y(n), y(n-1), ..., y(n-ny+1)
        phi_y = y(n:-1:n-ny+1).';
        % u(n), u(n-1), ..., u(n-nu+1)
        phi_u = u(n:-1:n-nu+1).';
        Phi(row,:) = [phi_y phi_u];
        y_next(row) = y(n+1);
        row = row + 1;
    end
end
