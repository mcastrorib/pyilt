clear all;
clc;
close all;

data = load('C:\Users\jgfil\Google Drive (jgfilgueiras@id.uff.br)\MATLAB\Shell\dados_processados\Grad_MultiTau\grad_BB2.txt');

time = data(:,1); %% em milissegundos
decay = data(:,2);

% [x,y,ss] = LCurve (time,decay,256);    % rodar para otimizar o regularizador

[t2,dt2,err] = ilt_grad(time,decay,3.71, 256)   % rodar com regularizador otimizado

figure;
h = semilogx(t2,dt2);
set(gca, 'FontName', 'Calibri', 'FontSize', 14, 'TickDir', 'out', 'Ticklength', [0.02 0.035]);
xlabel('T_2 (ms)'); set(h(1), 'linewidth', 1.5);

D = 2.2952e-8;    % water diffusion coefficient at 25 oC (cm2/ms)
gamma = 4.258;   % fator magnetogírico do 1H (kHz/G)

aux1 = gamma^2 * D; aux = 3/aux1;

Grad = sqrt(aux./t2);

figure;
h = semilogx(Grad,dt2);
set(gca, 'FontName', 'Calibri', 'FontSize', 14, 'TickDir', 'out', 'Ticklength', [0.02 0.035]);
xlabel('internal gradient (G/cm)'); set(h(1), 'linewidth', 1.5); xlim([10 10000]);