
function [x,y,ss] = LCurve_grad(time,decay,supp)
  ss = 0;
  s = logspace(-1.5,1.5,30);
  N = length(s);
  x = zeros(N,1);
  y = zeros(N,1);
  errv = zeros(N,1);
  solv = zeros(N,1);
  for i = 1:N
    reg = s(i);
    if ((reg>0.1)&&(ss == 0))
      ss=i;
    end
    [t,dt,err] = ilt_grad(time,decay,reg,supp);
    errv(i) = err;
    solv(i) = sqrt(dot(dt,dt));    
    fprintf('Iteration %i',i);
    fprintf('Regularizer %f ',reg);
    fprintf('Solution %d \n',solv(i));
  
    figure(1); hold on
    subplot(5,6,i);
    h = semilogx(t,dt);
    set(gca, 'FontName', 'Times', 'FontSize', 10);
    xlim([0.0001 10]); %ylim([-0.01 1.01]);
    xlabel('T_2 (s)');
    set(h, 'linewidth', 1.5);

  end

  hold off

  figure(2);
  h = loglog(errv,solv,'*');
  set(gca, 'FontName', 'Times', 'FontSize', 14);
  title('L Curve criterion');
  xlabel('Aproximation error');
  ylabel('Magnitude');
  grid on;
end
