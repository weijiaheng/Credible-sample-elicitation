function key = logbarrier(N, Q, c)
% with solver SeDuMi
cvx_begin
    cvx_precision best
    variable alp(N) nonnegative
    
    minimize( quad_form(alp, 0.5*Q) + c'*alp - 1/N*sum( log(alp) ) )

cvx_end

KLest = -log(N) - sum(log(alp))/N;

key.alp = alp;
key.KLest = KLest;

return;
end