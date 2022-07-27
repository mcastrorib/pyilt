% Inverse Laplace Transform
% time  -> time axis of the exponential decay
% decay -> magnetization 
% reg   -> regularizer
% supp  -> number of bins
% returns tt2 and dt2 for T2 distribution and the signal to noise ratio (snr).

function [t2,dt2,err] = ilt (time ,decay ,reg, supp)
  N=2048;
  id = round(logspace(log10(1),log10(length(time)),N));
  time = time(id);
  decay = decay(id);
  t2d = logspace(log10(0.01),log10(200000),supp); % Para gradi int
  G = length(t2d);
  M = zeros(N,G);
  for i0 = 1:G
    T2 = t2d(i0);
    yg = exp(-time./T2);
    M(:,i0)=yg;
  end
  if (reg == 0)
    y = lsqnonneg(M,seq);
  else
    Mreg = [M; reg*eye(G,G)];
    s    = [decay; zeros(G,1)]; 
    y    = lsqnonneg(Mreg,s);
  end
  t2 =t2d';
  dt2 = y;
  err = dot(decay-M*y,decay-M*y);
  

end


