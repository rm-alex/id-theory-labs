datasets = {'zad21','zad22'};
alpha = 0.05; % уровень значимости для F-теста

for k = 1:numel(datasets)
    name = datasets{k};
    data = eval(name);

    T = data.T(:);
    V = data.V(:);

    % idx = ~isnan(T) & ~isnan(V);
    % T = T(idx);
    % V = V(idx);

    % n = numel(T);
    % if n < 3
    %     fprintf('Набор %s: недостаточно точек (n=%d) для оценки.\n', name, n);
    %     continue;
    % end

    % Матрицы проектирования
    X1 = [T, ones(n,1)];             % H1: V = b*T + c
    X2 = [T.^2, T, ones(n,1)];       % H2: V = a*T^2 + b*T + c

    % Оценки МНК (нормальное уравнение)
    beta1 = X1 \ V;
    beta2 = X2 \ V;

    % Предсказания
    V_hat1 = X1 * beta1;
    V_hat2 = X2 * beta2;

    % Ошибки
    res1 = V - V_hat1;
    res2 = V - V_hat2;

    % SSE, SST, R2, adjR2, RMSE
    SSE1 = sum(res1.^2);
    SSE2 = sum(res2.^2);
    SST = sum((V - mean(V)).^2);

    R2_1 = 1 - SSE1 / SST;
    R2_2 = 1 - SSE2 / SST;

    p1 = size(X1,2); % число параметров в H1 (2)
    p2 = size(X2,2); % число параметров в H2 (3)

    adjR2_1 = 1 - (SSE1/(n-p1)) / (SST/(n-1));
    adjR2_2 = 1 - (SSE2/(n-p2)) / (SST/(n-1));

    RMSE1 = sqrt(SSE1 / (n - p1));
    RMSE2 = sqrt(SSE2 / (n - p2));

    % AIC (приближённая форма)
    AIC1 = n*log(SSE1/n) + 2*p1;
    AIC2 = n*log(SSE2/n) + 2*p2;

    % F-тест (H1 — вложенная, H2 — полная)
    dfn = p2 - p1;       % степени свободы числителя
    dfd = n - p2;        % степени свободы знаменателя
    F = ((SSE1 - SSE2)/dfn) / (SSE2/dfd);
    pF = 1 - fcdf(F, dfn, dfd); % p-value для F

    fprintf('------------------------------\n');
    fprintf('Данные: %s (n=%d)\n', name, n);
    fprintf('H1: V = b*T + c\n  b = %.6g, c = %.6g\n', beta1(1), beta1(2));
    fprintf('  SSE = %.6g, RMSE = %.6g, R2 = %.4f, adjR2 = %.4f, AIC = %.4f\n', SSE1, RMSE1, R2_1, adjR2_1, AIC1);
    fprintf('H2: V = a*T^2 + b*T + c\n  a = %.6g, b = %.6g, c = %.6g\n', beta2(1), beta2(2), beta2(3));
    fprintf('  SSE = %.6g, RMSE = %.6g, R2 = %.4f, adjR2 = %.4f, AIC = %.4f\n', SSE2, RMSE2, R2_2, adjR2_2, AIC2);
    fprintf('F-test (H1 vs H2): F = %.4f, p = %.4g (df = [%d,%d])\n', F, pF, dfn, dfd);

    % критерии: значимая F (p<palpha) --> H2 лучше; иначе предпочитаем
    % модель проще
    if pF < alpha && SSE2 < SSE1
        better = 'H2 (квадратичная) предпочтительнее по F-тесту (значимо).';
    else
        % если F незначимо, смотрим на AIC и adjR2
        if AIC2 < AIC1 && adjR2_2 > adjR2_1
            better = 'H2 предпочтительнее по AIC и скорректированному R^2, но F-тест не значим.';
        elseif AIC1 < AIC2 && adjR2_1 >= adjR2_2
            better = 'H1 (линейная) предпочтительнее по AIC и/или скорректированному R^2.';
        else
            better = 'Результаты смешанные: используйте графики ошибок и предметную интерпретацию.';
        end
    end
    fprintf('РЕКОМЕНДАЦИЯ: %s\n', better);

    figure('Name', ['Аппроксимации и Ошибки — ' name], 'NumberTitle', 'off', 'Position',[100,100,900,600]);

    subplot(2,2,1);
    hold on; box on;
    scatter(T, V, 40, 'filled');

    Tgrid = linspace(min(T), max(T), 200)';
    Vgrid1 = [Tgrid, ones(numel(Tgrid),1)] * beta1;
    Vgrid2 = [Tgrid.^2, Tgrid, ones(numel(Tgrid),1)] * beta2;
    plot(Tgrid, Vgrid1, 'LineWidth', 1.8);
    plot(Tgrid, Vgrid2, 'LineWidth', 1.8);
    legend('V(T)', 'V(T)=H1', 'V(T)=H2', 'Location','best');
    xlabel('T (°C)'); ylabel('V (T)');
    title(sprintf('%s: Данные и аппроксимации', name));
    grid on;
    hold off;

    % err H1
    subplot(2,2,2);
    plot(T, res1, 'o');
    hold on; plot([min(T), max(T)], [0,0], 'k--'); hold off;
    xlabel('T (°C)'); ylabel('V - \^V (H1)');
    title('Ошибки H1'); grid on; box on;

    % err H2
    subplot(2,2,3);
    plot(T, res2, 'o');
    hold on; plot([min(T), max(T)], [0,0], 'k--'); hold off;
    xlabel('T (°C)'); ylabel('V - \^V (H2)');
    title('Ошибки H2'); grid on; box on;

    % сравнение абсолютных ошибок
    subplot(2,2,4);
    scatter(res1, res2, 30, 'filled');
    xlabel('V - \^V (H1)'); ylabel('V - \^V (H2)');
    title('Сравнение ошибок H1 и H2'); grid on; box on;

    hold on; lims = axis; plot(lims(1:2), lims(1:2), 'k--'); axis(lims); hold off;

    % figure('Name', ['Диагностика ошибок — ' name], 'NumberTitle', 'off', 'Position',[150,150,800,400]);
    % subplot(1,2,1);
    % histogram(res1);
    % title('Гистограмма ошибок H1');
    % xlabel('Ошибка'); ylabel('count'); box on;
    % subplot(1,2,2);
    % histogram(res2);
    % title('Гистограмма ошибок H2');
    % xlabel('Ошибка'); ylabel('count'); box on;
end
