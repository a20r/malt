
function waitForEnd()
    for i = 1:10000
        k = waitforbuttonpress;
        if k == 1
            break;
        end
    end
endfunction

function graph_nn_experiments()
    data = load('sandbox/nn.txt');
    figure;
    plot(data(:, 1), data(:, 2), 'rx-', 'linewidth', 3);
    hold on;
    errorbar(data(:, 1), data(:, 2), data(:, 3), 'rx-');
    xlabel('Number of sensors');
    ylabel('Error [meters]');
    title('NN');
endfunction

function graph_in_experiments()
    figure;
    hold on;

    data = load('sandbox/in_10.txt');
    plot(data(:, 1), data(:, 2), 'rx-', 'linewidth', 3);

    data = load('sandbox/in_20.txt');
    plot(data(:, 1), data(:, 2), 'gx-', 'linewidth', 3);

    data = load('sandbox/in_30.txt');
    plot(data(:, 1), data(:, 2), 'bx-', 'linewidth', 3);

    legend('10 nodes', '20 nodes', '30 nodes', 'Location', 'BestOutside');
    xlabel('Amount of sensor noise [% error std]');
    ylabel('Error [meters]');
    title('IN');
endfunction

graph_nn_experiments();
graph_in_experiments();
waitForEnd()
