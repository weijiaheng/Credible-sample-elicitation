function key = generate_samples(idx, Nx, uttype)

% generate samples
if idx == 1
    % 900 case
    seed = 900;
    Sigma = [9.54519143, 9.43694302; 9.43694302, 10.08116575];
    mu = [-3.83100007, 2.17349291];
elseif idx == 2
    % 987 case
    seed = 987;
    Sigma = [10.54485165, 16.17829607; 16.17829607, 26.43088196];
    mu = [6.97831699, 8.38476761];
elseif idx == 3
    % 10086 case
    seed = 10086;
    Sigma = [1.2785415, 4.3921567; 4.3921567, 16.1871173];
    mu = [-2.97037504, 8.97720813];
end

x = mvnrnd(mu, Sigma, Nx);

if uttype == 1
    % random shift report
    utlevel = cat(2, unifrnd(0, 3, [Nx,1]), zeros(Nx,1));
elseif uttype == 2
    % totally random report
    % range 2sigma
    utlevel = cat(2, unifrnd(0, 2*sqrt(Sigma(1,1)), [Nx,1])-x(:,1), zeros(Nx,1));
end

if uttype ~= 0
    x = x + utlevel;
end

smp_1 = x(:,1);
smp_2 = x(:,2);
idx1 = randi([1, Nx], 1, Nx);
idx2 = randi([1, Nx], 1, Nx);
y = cat(2, smp_1(idx1), smp_2(idx2));

key.seed = seed;
key.x = x;
key.y = y;
return;
end