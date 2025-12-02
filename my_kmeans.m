function [idx, C, J_hist] = my_kmeans(X, k, maxIter, tol)
% MY_KMEANS  Basit k-means clustering algoritmasi
%  [idx, C, J_hist] = my_kmeans(X, k, maxIter, tol)
%   X      : N x d veri matrisi (burada d = 2)
%   k      : küme sayısı
%   maxIter: maksimum iterasyon sayısı (opsiyonel, default=100)
%   tol    : centroid hareket toleransı (opsiyonel, default=1e-6)
%
%   idx    : N x 1 küme indeksleri (1..k)
%   C      : k x d centroidler
%   J_hist : her iterasyondaki toplam hata (within-cluster sum of squares)

    if nargin < 3
        maxIter = 100;
    end
    if nargin < 4
        tol = 1e-6;
    end

    [N, d] = size(X);

    % 1) Başlangıç centroid'leri: veriden rastgele k nokta seç
    perm = randperm(N, k);
    C = X(perm, :);         % k x d
    idx = zeros(N,1);
    J_hist = zeros(maxIter,1);

    for it = 1:maxIter
        % --------------------------------------------------------------
        % 2) Atama adımı: her noktayı en yakın centroid'e ata
        % --------------------------------------------------------------
        for i = 1:N
            diffs = C - X(i,:);                % k x d
            dist2 = sqrt(sum(diffs.^2, 2));          % k x 1 (Öklid^2)
            [~, idx(i)] = min(dist2);
        end

        % --------------------------------------------------------------
        % 3) Güncelleme adımı: her küme için centroid'i güncelle
        % --------------------------------------------------------------
        C_old = C;
        for j = 1:k
            pts = X(idx == j, :);
            if ~isempty(pts)
                C(j,:) = mean(pts, 1);
            else
                % Küme boş kalırsa yeniden rastgele başlat
                C(j,:) = X(randi(N), :);
            end
        end

        % --------------------------------------------------------------
        % 4) Maliyet fonksiyonu J(it) = sum ||x_i - c_{k(i)}||^2
        % --------------------------------------------------------------
        J = 0;
        for j = 1:k
            pts = X(idx == j, :);
            if ~isempty(pts)
                diffs = pts - C(j,:);
                J = J + sum(sum(diffs.^2));
            end
        end
        J_hist(it) = J;

        % --------------------------------------------------------------
        % 5) Yakınsama kontrolü: centroid'ler neredeyse hareket etmiyorsa dur
        % --------------------------------------------------------------
        centroid_move = max(vecnorm(C - C_old, 2, 2));  % her centroid için ||Δc_j||
        if centroid_move < tol
            J_hist = J_hist(1:it);  % kullanılmayan iterasyonları at
            break;
        end
    end
end
