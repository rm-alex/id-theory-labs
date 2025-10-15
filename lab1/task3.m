datasets = {'zad31', 'zad32'};

for i = 1:length(datasets)
    name = datasets{i};
    data = eval(name);

    fprintf('\n===== Обработка %s =====\n', name);

    x = data.x(:);
    y = data.y(:);

    switch name
        case 'zad31'
            % y = 7^(p1) * x^(p2)
            Y = log(y);
            X = [log(7)*ones(size(x)), log(x)];
        case 'zad32'
            % y = p1 * e^(x*p2)
            Y = log(y);
            X = [ones(size(x)), x];      
    end

    theta_hat = estim(X, Y);
    e = diff_anal(Y, X*theta_hat);
    correx(e, x);

    switch name
        case 'zad31'
            y_hat = 7.^theta_hat(1) .* x.^theta_hat(2);
            fprintf('p1_hat = %.4f, p2_hat = %.4f\n', theta_hat(1), theta_hat(2));

        case 'zad32'
            y_hat = exp(theta_hat(1)) .* exp(x .* theta_hat(2));
            fprintf('p1_hat = %.4f, p2_hat = %.4f\n', exp(theta_hat(1)), theta_hat(2));

            plot((1:length(Y))', x/10);
            hold on;
            plot((1:length(Y))', e/3*10^15);
            legend('x/10','e/3*10^15','Location','best');
            hold off;
    end

    depict(i, y, y_hat);
    
end


function theta_hat = estim(X, Y)
    fprintf('\n--- Оценка theta ---\n');
    N = length(Y);
    % Проверка размеров
    if size(X,1) ~= N
        error('Размеры X и Y не согласованы в %s.', dsname);
    end
    
    % Оценка параметров (OLS) с учётом возможной вырожденности
    XtX = X' * X;
    det_XtX = det(XtX);
    cond_XtX = rcond(XtX);
    inv_XtX = inv(XtX)
    
    fprintf('det(X^T X) = %.4e\n', det_XtX);
    fprintf('rcond(X^T X) = %.4e\n', cond_XtX);
    theta_hat = inv_XtX * (X' * Y);
end

function e = diff_anal(y, y_hat)
    fprintf('\n\n--- Параметры отклонений ---\n');
    e = y - y_hat;
    mean_e = mean(e);
    var_e = var(e);
    dw = sum(diff(e).^2) / sum(e.^2); % Durbin-Watson statistic
    fprintf('mean(residuals) = %.4e, var(residuals) = %.4e, Durbin-Watson = %.4f\n', mean_e, var_e, dw);
    % Ошибка
    RMSE = sqrt(mean((y - y_hat).^2));
    fprintf('RMSE = %.6e\n', RMSE);
end

function correx(e, X)
    % ---------------------------------------------------------------
    % Проверка корреляции помехи (остатков) с регрессорами
    % ---------------------------------------------------------------
    fprintf('\n\n--- Проверка корреляции остатков с регрессорами ---\n');

    p = size(X,2);
    corr_ex = zeros(1, p);
    p_corr = zeros(1, p);
    
    for j = 1:p
        [R, P] = corrcoef(e, X(:, j)); % 2x2 корреляционная матрица и p-value
        corr_ex(j) = R(1, 2);
        p_corr(j) = P(1, 2);
    end
    
    T_corr = table((1:p)', corr_ex', p_corr', ...
        'VariableNames', {'RegressorIndex', 'corr(e,x_i)', 'p_value'});
    
    disp(T_corr);
   
    % Интерпретация:
    % Если |corr(e, x_i)| < 0.1 и p_value > 0.05 → корреляции статистически нет.
    % Если корреляция значима (p<0.05) → нарушено одно из условий Гаусса–Маркова,
    % оценки могут быть смещёнными (или неэффективными).
end

function depict(i, Y, Y_hat)
    t = (1:length(Y))';
    figure('Name',sprintf('%d: y и y_hat', i),'Position',[100,100,600,300]);
    plot(t,Y,'-o','MarkerSize',5); hold on;
    plot(t,Y_hat,'-x','MarkerSize',5); 
    legend('y','y^{hat}','Location','best');
    xlabel('Номер измерения'); ylabel('y');
    title(sprintf('%d: y, y^{hat}', i));
    grid on;

    saves(3, 2, i)
    
    hold off;

    % figure('Name',sprintf('%d: ошибка оценивания', i),'NumberTitle','off');
    % plot(t,e,'-s','MarkerSize',3);
    % xlabel('Номер измерения'); 
    % % ylabel('Ошибка e = y - y^{hat}');
    % title(sprintf('%d: ошибка оценивания', i));
    % grid on;
    % 
    % saves(3, 3, i)
end

function saves(No, pt, i)
    filename = sprintf('Tid_LR1_No_%d_pt_%d_sys_%d.png', No, pt,i);
    set(gcf,'Color',[1 1 1]);
    exportgraphics(gcf,filename,'Resolution',300);
end