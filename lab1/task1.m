datasets = {'zad11','zad12'};
for i=1:numel(datasets)
    dsname = datasets{i};
    if ~exist(dsname,'var')
        warning('Переменная %s не найдена в workspace — пропускаю.', dsname);
        continue
    end
    fprintf('\n=== Обработка %s ===\n', dsname);
    zad = eval(dsname);

    % Собираем векторы
    Y = zad.y(:);
    X = [zad.x1(:), zad.x2(:), zad.x3(:)];
    [N,p] = deal(length(Y), size(X,2));

    % Проверка размеров
    if size(X,1) ~= N
        error('Размеры X и Y не согласованы в %s.', dsname);
    end

    % Оценка параметров (OLS) с учётом возможной вырожденности
    XtX = X' * X;
    if rcond(XtX) < 1e-12
        warning('Матрица X''*X плохо обусловлена или вырождена — использую псевдообратную (pinv).');
        theta_hat = pinv(XtX) * (X' * Y); % альтернативно: theta_hat = pinv(X) * Y;
    else
        theta_hat = XtX \ (X' * Y);
    end

    % Оцененное выходное значение и ошибка
    Y_hat = X * theta_hat;
    e = Y - Y_hat;

    % Оценка дисперсии шума и ковариации оценок
    dof = N - p; % degrees of freedom
    sigma2_hat = (e' * e) / dof;
    Cov_theta = sigma2_hat * inv(XtX);
    se = sqrt(diag(Cov_theta));
    t_stats = theta_hat ./ se;

    % Вывод в таблицу
    T = table((1:p)', theta_hat, se, t_stats, 'VariableNames', {'param_index','theta_hat','std_error','t_stat'});
    disp(T);
    fprintf('Оценка sigma^2 = %.4e, N = %d, p = %d, dof = %d\n', sigma2_hat, N, p, dof);

    % Диагностика остатков
    mean_e = mean(e);
    var_e = var(e);
    dw = sum(diff(e).^2) / sum(e.^2); % Durbin-Watson statistic
    fprintf('mean(residuals) = %.4e, var(residuals) = %.4e, Durbin-Watson = %.4f\n', mean_e, var_e, dw);

    % Плот графиков
    t = (1:N)';
    figure('Name',sprintf('%s: y и y_hat', dsname),'NumberTitle','off');
    plot(t,Y,'-o','MarkerSize',3); hold on;
    plot(t,Y_hat,'-x','MarkerSize',3); hold off;
    legend('y (измерения)','\hat{y} (оценка)','Location','best');
    xlabel('Номер измерения'); ylabel('y');
    title(sprintf('%s: исходный сигнал y и оценка \hat{y}', dsname));
    grid on;

    figure('Name',sprintf('%s: ошибка оценивания', dsname),'NumberTitle','off');
    plot(t,e,'-s','MarkerSize',3);
    xlabel('Номер измерения'); ylabel('Ошибка e = y - \hat{y}');
    title(sprintf('%s: ошибка оценивания', dsname));
    grid on;

    % Гистограмма остатков (быстрая проверка нормальности)
    figure('Name',sprintf('%s: гистограмма остатков', dsname),'NumberTitle','off');
    histogram(e); hold on;
    xx = linspace(min(e),max(e),100);
    % Нормальная плотность, используя оценённые параметры
    pd = (1/sqrt(2*pi*sigma2_hat)) * exp(-(xx-mean_e).^2/(2*sigma2_hat));
    % Масштабировать плотность, чтобы поместилась на гистограмму
    h = histogram(e); maxcount = max(h.Values); delete(h);
    scale = maxcount / max(pd);
    bar_data = histogram('BinEdges',linspace(min(e),max(e),21),'BinCounts',histcounts(e,linspace(min(e),max(e),21))); hold on;
    plot(xx, pd*scale, 'LineWidth',2);
    hold off;
    title(sprintf('%s: гистограмма остатков и нормальная плотность (масштабирована)', dsname));

    % Сохранение результатов в структуру для возможного дальнейшего использования
    results.(dsname).theta_hat = theta_hat;
    results.(dsname).Cov_theta = Cov_theta;
    results.(dsname).sigma2_hat = sigma2_hat;
    results.(dsname).Y_hat = Y_hat;
    results.(dsname).residuals = e;
    results.(dsname).mean_residual = mean_e;
    results.(dsname).dw = dw;

    % Сохранение фигур (опционально)
    % saveas(gcf, sprintf('%s_error.png', dsname)); % раскомментируйте при желании

end