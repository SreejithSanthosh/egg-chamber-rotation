%% This code simulatess the delayed migration egg chamber
clear; close all; clc

% Display parameters
fntSz = 24;

% Physical parameters
ecc = 1.2; tau1 = 20; tau2 = 20; tau3 = 20;

%Simulation parameters
nBeta = 40; dBeta = 2*pi/nBeta; betaArr = -pi:dBeta:pi; nBeta = numel(betaArr);
dt = 10^(-2); T = 1; tArr = 0:dt:T; Nt = numel(tArr);

% Making the elipsoidal mesh 
Nspac = 100; thetaArr = linspace(0+0.001,pi-0.001,Nspac);
phiArr = linspace(0+0.001,2*pi-0.001,Nspac);
[thetaMesh,phiMesh] = meshgrid(thetaArr,phiArr);
thetaMesh = thetaMesh(:); phiMesh = phiMesh(:);
xMesh = cos(phiMesh).*sin(thetaMesh); yMesh = sin(phiMesh).*sin(thetaMesh);
zMesh = ecc*cos(thetaMesh); rManifoldBody = [xMesh';yMesh';zMesh'];
triMesh = convhulln(rManifoldBody');

% % Define Ncells and location in material frame  
nCells = 300;  rPosCellsBody = placeCellsUniformly(nCells,ecc); 
nCells = size(rPosCellsBody,2); szCellsPlot = 10; axLim = 2;
[P, K, voronoiboundary, s] = voronoisphere(rPosCellsBody); % Compute vornoi boundaries
phiCell = atan2(rPosCellsBody(2,:),rPosCellsBody(1,:));
thetaCell = atan2(sqrt(rPosCellsBody(2,:).^2+rPosCellsBody(1,:).^2),rPosCellsBody(3,:));

% Define the tangent vector
zeta1 = [cos(thetaCell).*cos(phiCell)./sqrt(cos(thetaCell).^2+(ecc*sin(thetaCell)).^2);...
cos(thetaCell).*sin(phiCell)./sqrt(cos(thetaCell).^2+(ecc*sin(thetaCell)).^2);...
-ecc*sin(thetaCell)./sqrt(cos(thetaCell).^2+(ecc*sin(thetaCell)).^2)];
zeta2 = [-sin(phiCell);cos(phiCell);0*phiCell];

nVecCell = cross(zeta1,zeta2); nVecCellNorm = nVecCell./vecnorm(nVecCell,2,1); % Define the normal vector


%% Simulate the equtions
fricAngMmntCoeff = getfricCoeffAngMomentum(ecc);
omegaStoreBody = nan([3,Nt]); rotMatrStore = zeros([3,3,Nt]); bStoreBody = zeros([3,nCells,Nt]);
cStore = zeros([Nt,nCells,nBeta]);

% Define the intial conditions 
rotMatrStore(:,:,1) = eye(3); bStoreBody(:,:,1) = zeros([3,nCells]);
cStore(1,:,:) = 1/(2*pi);

% Evolve the equations 

for ct = 1:Nt-1
    b_pt = squeeze(bStoreBody(:,:,ct)); %Material frame 
    rotMatr_pt = squeeze(rotMatrStore(:,:,ct));
    c_pt = squeeze(cStore(ct,:,:));
    
    % Angular Momentum Balance for omega 
    totalTorqueBody = cross(rPosCellsBody,b_pt); 
    omegaStoreBody_ct = (mean(totalTorqueBody,2)./fricAngMmntCoeff);
    omegaStoreBody(:,ct) = omegaStoreBody_ct; omegaEulerian = rotMatr_pt*omegaStoreBody_ct;
    o1 = omegaEulerian(1); o2 = omegaEulerian(2); o3 = omegaEulerian(3);
    A = [0,-o3,o2;o3,0,-o1;-o2,o1,0];
    rotMatrStore(:,:,ct+1) = rotMatrStore(:,:,ct) +dt*(A*rotMatr_pt); % Neg sign due to rotMatr(A): A r_b = r_e
    
    % Protrusions update
    term1 = generateProtrusions(zeta1,zeta2,c_pt,betaArr)+b_pt;
    term2 = A*b_pt; % \omega  x p = \Omega p
    bStoreBody(:,:,ct+1) = b_pt+dt*(-tau1*term1+term2);
    
    %Fat2 update
    cStore(ct+1,:,:) =  c_pt + dt*evolveProbCell(zeta1,zeta2,rPosCellsBody,c_pt,betaArr,omegaStoreBody_ct,tau2,tau3);

end 

% Compute the eulerian omega and long-axis orientation
orientLongAxis = nan([2,Nt]); orientOmegaAxis = nan([2,Nt]);
for ct = 1:Nt-1
    rotMatr_ct = squeeze(rotMatrStore(:,:,ct));
    omegaEulerian = rotMatr_ct*omegaStoreBody(:,ct);

    [az,el,~] = cart2sph(omegaEulerian(1),omegaEulerian(2),...
        omegaEulerian(3));
    orientOmegaAxis(1,ct) = az; orientOmegaAxis(2,ct) = pi/2-el;

end 

% %
clc; close all;
figure('color','w','position',[3 200 2.5445e+03 494])
subplot(1,3,1)
plot(tArr,omegaStoreBody(3,:),'b','LineWidth',5,'DisplayName','$\omega_3$'); hold on 
% plot(tArr,vecnorm(omegaStoreBody,2,1),'r','LineWidth',3,'DisplayName','$|\omega|$'); 
hold off;pbaspect([1.2 1 1]);
set(gca,'LineWidth',2,'FontSize',fntSz)
xlabel('$t$','FontSize',fntSz,'Interpreter','latex')
ylabel('$ \omega_3$','FontSize',fntSz,'Interpreter','latex')
% legend('location','southeast','EdgeColor','none','FontSize',fntSz,'Interpreter','latex','color','none')

subplot(1,3,2)
tArrPlot = [10^(-2),0.5,1]*T;
for i = 1:numel(tArrPlot)
    [~,I] = min(abs(tArr-tArrPlot(i)));
    lgd = sprintf('$t = %.1f$',tArrPlot(i));
    plot(betaArr,mean(squeeze(cStore(I,:,:)),1),'Linewidth',5,'DisplayName',lgd); hold on
end 
xticks([-pi/2,0,pi/2]); xticklabels({'-\pi/2 ','0','\pi/2'})
set(gca,'LineWidth',2,'FontSize',fntSz);pbaspect([1.2 1 1]);
xlabel('$ \beta $','FontSize',fntSz,'Interpreter','latex')
ylabel('$ <c_i(\beta,t)>$','FontSize',fntSz,'Interpreter','latex')
legend('location','southeast','EdgeColor','none','FontSize',fntSz,'Interpreter','latex','color','none')

subplot(1,3,3)  % Plot the orientation of the omega axis (%theta, %phi)
cr = 5;  polarplot(rad2deg(orientOmegaAxis(1,:)),rad2deg(orientOmegaAxis(2,:)),'k','LineWidth',0.5); hold on 
polarscatter(rad2deg(orientOmegaAxis(1,1:cr:end)),rad2deg(orientOmegaAxis(2,1:cr:end)),120,tArr(1:cr:end),'filled'); hold on
c = colorbar;  colormap('turbo'); c.FontSize = fntSz; 
set(gca,'color','w','FontSize',fntSz);
rlim([0 180 ]); rticks([90,180]); rticklabels({'90','180'})
title('Omega Axis Orientation','Interpreter','latex','FontSize',fntSz);

%% Visualize the simulations

clc; close all; Nplots = 30;
f= figure('color','k','Units','pixels','OuterPosition',[23 241 2550 874]);

for ct = 2:round((Nt-1)/Nplots):Nt-1

    b_ct = squeeze(bStoreBody(:,:,ct)); %Material frame 
    rotMatr_ct = squeeze(rotMatrStore(:,:,ct));
    omega_ct = squeeze(omegaStoreBody(:,ct));
    c_ct = squeeze(cStore(ct,:,:));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    %%  Plot the egg chamber  
    subplot(1,4,[1,2])
    rManifoldEulerian_ct = rotMatr_ct*rManifoldBody;
    rPosCellsEulerian_ct = rotMatr_ct*rPosCellsBody;

    p = trisurf(triMesh,-rManifoldEulerian_ct(3,:),rManifoldEulerian_ct(2,:),rManifoldEulerian_ct(1,:),'FaceColor',[0.5 0.5 0.5],'EdgeColor','none');
    p.SpecularStrength = 0.2; p.AmbientStrength = 0.3;

    for k = 1:nCells
        hold on
        X = voronoiboundary{k}; X(1,:) = X(1,:); X(2,:) = X(2,:);
        X(3,:) = ecc*X(3,:); X = rotMatr_ct*X;
        fill3(-X(3,:),X(2,:),X(1,:),'g','FaceAlpha',0,'EdgeColor','k','LineWidth',1);hold off
    end
    
    % plot the protrusions 
    bEulerian = rotMatr_ct*b_ct;
    hold on; quiver3(-rPosCellsEulerian_ct(3,:),rPosCellsEulerian_ct(2,:),...
        rPosCellsEulerian_ct(1,:),-bEulerian(3,:),bEulerian(2,:),...
        bEulerian(1,:),'Color',[0.466 0.674 0.188],'LineWidth',1);

    % Plot the long and omega axis 
    
    p1 = plot3([0,0],[0,0],[-1.3,1.3],'r','LineWidth',2,'DisplayName','AP axis'); hold off
    legend(p1,'TextColor','w','FontSize',fntSz,'color','none','Interpreter','latex','EdgeColor','none')
    set(gca,'color','k','FontSize',fntSz);
    rotate(p1,[0 1 0],90); 
    
    azLim = 1.1;
    xlim([-axLim axLim]); ylim([-axLim axLim]); zlim([-axLim axLim]);
    set(gca,'color','k','FontSize',fntSz); camva(5); camlight;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    subplot(1,4,3) % The polarity distribution of the cells 
    plot(betaArr,mean(c_ct,1),'LineWidth',2,'Color','w')
    xlabel('$ t $','Interpreter','latex');
    ylabel('$ <c_i(\beta,t)>$','Interpreter','latex');
    set(gca,'color','k','FontSize',fntSz,'Xcolor','w','Ycolor','w');
    xlim([-pi,pi]); ylim([0, 1.5]);
    title('Fat2 distribution','Interpreter','latex','FontSize',fntSz,'Color','w'); 
    pbaspect([1 1 1]);
    %%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(1,4,4) % Omega visualization 
    omegaModMaterial = vecnorm(omegaStoreBody(:,1:ct),2,1);
    plot(tArr(1:ct),omegaStoreBody(3,1:ct),'b','LineWidth',2); hold off
    set(gca,'color','k','FontSize',fntSz,'Xcolor','w','Ycolor','w');
    xlim([0,max(tArr)]); ylim([0,max(vecnorm(omegaStoreBody,2,1))]);
    xlabel('$ t $','Interpreter','latex');
    ylabel('$ \omega_3 $','Interpreter','latex');
    pbaspect([1 1 1]);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Include a description of the parameters 
    annstr0 = 'Parameters';
    annstr1 = sprintf('$ \\tau_1 = %.1f $, $ \\tau_2 = %.1f $, $ \\tau_3 = %.1f $',tau1,tau2,tau3);
    annpos = [0.05 0.05 0.1 0.1];
    ha = annotation('textbox',annpos,'string',{annstr0,annstr1});
    ha.Color = 'w'; ha.Interpreter = 'latex'; ha.FontSize = fntSz;

    sgtitle(sprintf('$$ t = %.1f $$',tArr(ct)),'Interpreter','latex','FontSize',30,'Color','w'); pause(0.1)
end


