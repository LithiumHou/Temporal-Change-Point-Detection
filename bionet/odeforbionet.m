% re-generate the data using the regression results
global init
global iniy
init = 1;
iniy = record(init+3500,[1,4,7]);
par1 = 1.97; par2 = 1.9982; par3 = 1.9760;   %estimated parameters in 'bioregression.m'
parfinal1=[par1,par2];
parfinal2=[par3,par3];
opts = odeset('RelTol',1e-12,'AbsTol',1e-12);
[t,x]=ode45('reconstruction_factory',[0,250],iniy,[],parfinal1,opts);
R_t = interp1(t,x(:,1),[0:0.25:249.75]','spline');
R1 = R_t;  %Two-stage reconstruction
clear t x R_t
[t,x]=ode45('reconstruction_factory',[0,250],iniy,[],parfinal2,opts);
R_t = interp1(t,x(:,1),[0:0.25:249.75]','spline');
R2 = R_t;  %One-stage reconstruction
clear t x R_t
RD = record(init+3500:init+4499,1);  %Original data
fiterror1 = R1-RD;
fiterror2 = R2-RD;
t = 0:0.25:249.75;
fig = figure;

left_color = [0 0 0];
right_color = [0 0 0];
set(fig,'defaultAxesColorOrder',[left_color; right_color]);
yyaxis left
plot(t,R1,'-ro');
hold on
plot(t,R2,'-ro');
hold on
plot(t,RD,'-*');
ylabel({'Reconstructed \it x'},'FontSize',20.0000000001,'Rotation',90,...
    'Interpreter','latex');
yyaxis right
plot(t,fiterror1,'linewidth',1.5);
hold on
plot(t,fiterror2,'linewidth',1.5);
plot([125,125],[-3,1]);
ylabel({'Fitting error'},'FontSize',20.0000000001,'Rotation',270,...
    'Interpreter','latex');
legend('Two-stage reconstruction','One-stage reconstruction','Original data','Two-stage fitting error','One-stage fitting error','Change point','Interpreter','latex')