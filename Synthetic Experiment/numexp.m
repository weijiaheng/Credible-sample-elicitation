function numexp(Nsample)

Nx = Nsample;
Ny = Nx;

% parameters for EVKL algorithm
sigma = 1;
lmd = 0.1;

% main loop
uttype = 1;
expidx_list = [3];
nrepeats = 10;

for expidx = expidx_list
for i = 1:nrepeats
    smp = generate_samples( expidx, Nx, uttype );
    prep = prep_Q_c(smp.x, smp.y, Nx, Ny, sigma, lmd);
    res = logbarrier(Nx, prep.Q, prep.c);
    fprintf('#%d With Seed = %d, Sample num = %d, KL estimation = %.5f \n', i, smp.seed, Nx, res.KLest);
end
end
    
end