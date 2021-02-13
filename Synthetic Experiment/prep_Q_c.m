function key = prep_Q_c(x, y, Nx, Ny, sigma, lmd)

lambdaN = lmd / Ny;

ONE = ones(Nx, 1);

var = zeros(Nx, Ny);
for i = 1:Nx
    for j = 1:Ny
        delta = x(i,:) - y(j,:);
        mu2 = delta * delta';
        var(i,j) = -1.0 * mu2 / sigma;
    end
end
Kxy = exp(var);

for i = 1:Nx
    for j = i:Ny
        delta = y(i,:) - y(j,:);
        mu2 = delta * delta';
        var(i,j) = -1.0 * mu2 / sigma;
        var(j,i) = var(i,j);
    end
end
Kyy = exp(var);

Q = Kyy / lambdaN;
c = Kxy' * ONE / (-1.0 * lambdaN * Ny);

key.Q = Q;
key.c = c;

return;
end