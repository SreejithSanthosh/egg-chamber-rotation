%% This code simulatess the stage 1 egg chamber
clear; close all; clc;

% Display parameters
fntSz = 24; camView = [-21,30]; camView2 = [-92,-0.04]; 

% Physical parameters
tau1 = 20; tau2 = 20; tau3 = 20; thetaS = pi/6;
kStalk = 20; mu = 1; 

%Simulation parameters
nBeta = 40; dBeta = 2*pi/nBeta; betaArr = -pi:dBeta:pi; nBeta = numel(betaArr);
dt = 10^(-2); T = 10; tArr = 0:dt:T; Nt = numel(tArr);

% Making the mesh 
clc; close all;
Nspac = 100; thetaArr = linspace(0+0.001,pi-0.001,Nspac);
phiArr = linspace(0+0.001,2*pi-0.001,Nspac);
[thetaMesh,phiMesh] = meshgrid(thetaArr,phiArr);
thetaMesh = thetaMesh(:); phiMesh = phiMesh(:);
xMesh = cos(phiMesh).*sin(thetaMesh); yMesh = sin(phiMesh).*sin(thetaMesh);
zMesh = cos(thetaMesh); rManifoldBody = [xMesh';yMesh';zMesh'];
logic = squeeze(rManifoldBody(3,:))<cos(thetaS); rManifoldBody = rManifoldBody(:,logic);
triMesh = convhulln(rManifoldBody');

% % Define Ncells and location in material frame  
nCells = 100;  rPosCellsBody = placeCellsUniformly(nCells,thetaS);
nCells = size(rPosCellsBody,2); szCellsPlot = 10; axLim = 2;
[P, K, voronoiboundary, s] = voronoisphere(rPosCellsBody); % Compute vornoi boundaries

%%  Visualize the egg chamber 
trisurf(triMesh,-rManifoldBody(3,:),rManifoldBody(2,:),rManifoldBody(1,:),'FaceColor',[0.5 0.5 0.5],'EdgeColor','none'); hold on 
axis equal off; hold off; set(gcf,'color','w')
% Plot the voronoi tesselations  
for k = 1:nCells
    hold on
    X = voronoiboundary{k};
    logic = squeeze(X(3,:))<cos(thetaS);
    % fill3(X(1,logic),X(2,logic),X(3,logic),'g','FaceAlpha',0,'EdgeColor','k');hold off
    fill3(-X(3,logic),X(2,logic),X(1,logic),'g','FaceAlpha',0,'EdgeColor','k');hold off
end
%Plot the top 

Nstalk = 1000; rStalk = sin(thetaS);
theta = linspace(0, 2*pi, Nstalk+1); % +1 to close the circle
theta(end) = [];        % Remove duplicate endpoint

% Circle edge points
x_edge = rStalk * cos(theta); y_edge = rStalk * sin(theta);
z_edge = zeros(1, Nstalk)+cos(thetaS);

% Center point
xStalk = [0, x_edge]; yStalk = [0, y_edge];
zStalk = [cos(thetaS), z_edge];

% Triangle connectivity: center is point 1, edges are 2:N+1
TStalk = [(ones(Nstalk,1)), (2:Nstalk+1)', [3:Nstalk+1,2]'];

% hold on; trisurf(TStalk, xStalk, yStalk, zStalk, 'FaceColor', 'cyan', 'EdgeColor', 'none'); hold off
colorOrange = [255 180 71]*(1/255);
hold on; trisurf(TStalk, -zStalk, yStalk, xStalk, 'FaceColor',[0.5 0.5 0.5], 'EdgeColor', 'none','FaceAlpha',1); hold off

% Plot a cylinder 
r = sin(thetaS); h = 0.5; [xCyl,yCyl,zCyl] = cylinder(r,100); zCyl = zCyl*h;
[FCyl,VCyl] = surf2patch(xCyl,yCyl,zCyl,'triangles');
hold on; trisurf(FCyl,-VCyl(:,3)-cos(thetaS),VCyl(:,2),VCyl(:,1), 'FaceColor',colorOrange, 'EdgeColor', 'none'); hold on
trisurf(TStalk, -zStalk, yStalk, xStalk, 'FaceColor',colorOrange, 'EdgeColor', 'none','FaceAlpha',1); hold on % Face 1
trisurf(TStalk, -zStalk-h, yStalk, xStalk, 'FaceColor',colorOrange, 'EdgeColor', 'none','FaceAlpha',1); hold off % Face 2
view(camView);camlight; camlight(camView2(1),camView2(2))

phiCell = atan2(rPosCellsBody(2,:),rPosCellsBody(1,:));
thetaCell = atan2(sqrt(rPosCellsBody(2,:).^2+rPosCellsBody(1,:).^2),rPosCellsBody(3,:));

% Define the tangent vector
zeta1 = [cos(thetaCell).*cos(phiCell);cos(thetaCell).*sin(phiCell);-sin(thetaCell)];
zeta2 = [-sin(phiCell);cos(phiCell);0*phiCell];
nVecCell = cross(zeta1,zeta2); nVecCellNorm = nVecCell./vecnorm(nVecCell,2,1); % Define the normal vector


%% Simulate the equtions
omegaStoreBody = nan([3,Nt]); rotMatrStore = zeros([3,3,Nt]); bStoreBody = zeros([3,nCells,Nt]);
cStore = zeros([Nt,nCells,nBeta]);

% Define the intial conditions 
rotMatrStore(:,:,1) = eye(3); bStoreBody(:,:,1) = zeros([3,nCells]);
cStore(1,:,:) = 1/(2*pi);

% Evolve the equations 
progressbar
for ct = 1:Nt-1
    b_pt = squeeze(bStoreBody(:,:,ct)); %Material frame 
    rotMatr_pt = squeeze(rotMatrStore(:,:,ct));
    R33 = rotMatr_pt(3,3); R32 = rotMatr_pt(3,2); R31 = rotMatr_pt(3,1);
    c_pt = squeeze(cStore(ct,:,:));
    
    % Angular Momentum Balance for omega 
    totalTorqueActiveBody = cross(rPosCellsBody,b_pt); 
    totalTorqueActiveBody = mean(totalTorqueActiveBody,2);
    
    tauElStalk = [R32*(-4+3*R33+(-4+5*R33)*cos(2*thetaS));...
        -R31*(-4+3*R33+(-4+5*R33)*cos(2*thetaS));...
        0]; tauElStalk =(kStalk*pi/8)*(sin(thetaS)^2)*tauElStalk;
    tauTotal = tauElStalk+totalTorqueActiveBody;

    % The angular momentum balance 
    fricAngMmntCoeff = getfricCoeffAngMomentum(thetaS,mu);
    omegaStoreBody_ct = nan(3,1);
    omegaStoreBody_ct(1) = (1/fricAngMmntCoeff(1))*tauTotal(1);
    omegaStoreBody_ct(2) = (1/fricAngMmntCoeff(2))*tauTotal(2);
    omegaStoreBody_ct(3) = (1/fricAngMmntCoeff(3))*tauTotal(3);
    

    omegaStoreBody(:,ct) = omegaStoreBody_ct; omegaEulerian = rotMatr_pt*omegaStoreBody_ct;
    o1 = omegaEulerian(1); o2 = omegaEulerian(2); o3 = omegaEulerian(3);
    A = [0,-o3,o2;o3,0,-o1;-o2,o1,0];
    rotMatrStore(:,:,ct+1) = rotMatrStore(:,:,ct) +dt*(A*rotMatr_pt); % A r_b = r_e
    
    % Protrusions update
    term1 = generateProtrusions(zeta1,zeta2,c_pt,betaArr)+b_pt;
    term2 = A*b_pt; % \omega  x p = \Omega p
    bStoreBody(:,:,ct+1) = b_pt+dt*(-tau1*term1+term2);
    
    %Fat2 update
    cStore(ct+1,:,:) =  c_pt + dt*evolveProbCell(zeta1,zeta2,rPosCellsBody,c_pt,betaArr,omegaStoreBody_ct,tau2,tau3);
    progressbar(ct/(Nt-1))
end 

% Compute the eulerian omega and long-axis orientation
orientLongAxis = nan([2,Nt]); orientOmegaAxis = nan([2,Nt]);
for ct = 1:Nt-1
    rotMatr_ct = squeeze(rotMatrStore(:,:,ct));
    omegaEulerian = rotMatr_ct*omegaStoreBody(:,ct);
    longAxisEulerian = rotMatr_ct*[0;0;1];

    [az,el,~] = cart2sph(longAxisEulerian(1),longAxisEulerian(2),...
        longAxisEulerian(3));
    orientLongAxis(1,ct) = az;orientLongAxis(2,ct) = pi/2-el;

    [az,el,~] = cart2sph(omegaEulerian(1),omegaEulerian(2),...
        omegaEulerian(3));
    orientOmegaAxis(1,ct) = az; orientOmegaAxis(2,ct) = pi/2-el;

end 
 
clc; close all; figure('color','w','position',[4 200 2.5435e+03 416.5000])

subplot(1,4,1)
plot(tArr,vecnorm(omegaStoreBody,2,1),'b','LineWidth',5,'DisplayName','$|\omega|$'); hold on 
plot(tArr,abs(omegaStoreBody(3,:)),'r','LineWidth',5,'DisplayName','$|\omega_3|$'); hold off
set(gca,'LineWidth',2,'FontSize',fntSz)
xlabel('$t$','FontSize',fntSz,'Interpreter','latex')
ylabel('$ |\omega|,|\omega_3|$','FontSize',fntSz,'Interpreter','latex')
yy = legend; yy.EdgeColor = 'none'; yy.Interpreter = 'latex';
pbaspect([1 1 1]);

subplot(1,4,2)
tSelect2ShowArr = 0.00:0.01:0.98; tSelect2ShowArr = T*tSelect2ShowArr;
idXArry = [];
for i = 1:numel(tSelect2ShowArr)
    [~,idx2Show_i] = min(abs(tArr-tSelect2ShowArr(i)));
    idXArry = [idXArry,idx2Show_i];
end 
phiOmega = squeeze(orientOmegaAxis(1,:)); thetaOmega = squeeze(orientOmegaAxis(2,:));
polarplot(phiOmega(idXArry),rad2deg(thetaOmega(idXArry)),'k','LineWidth',1.5); hold on 
polarscatter(phiOmega(idXArry),rad2deg(thetaOmega(idXArry)),120,tSelect2ShowArr,'filled'); hold on
c = colorbar; colormap('turbo'); c.FontSize = fntSz;
title('Omega Axis','FontSize',fntSz,'Interpreter','latex')
rlim([0 180 ]); set(gca,'FontSize',fntSz,'LineWidth',1)
rticks([90,180]); rticklabels({'90','180'})

subplot(1,4,3)
phiOmega = squeeze(orientLongAxis(1,:)); thetaOmega = squeeze(orientLongAxis(2,:));
polarplot(phiOmega(idXArry),rad2deg(thetaOmega(idXArry)),'k','LineWidth',1.5); hold on 
polarscatter(phiOmega(idXArry),rad2deg(thetaOmega(idXArry)),120,tSelect2ShowArr,'filled'); hold on
c = colorbar; colormap('turbo'); c.FontSize = fntSz;
title('Long Axis','FontSize',fntSz,'Interpreter','latex')
rlim([0 180 ]); set(gca,'FontSize',fntSz,'LineWidth',1)
rticks([90,180]); rticklabels({'90','180'})


subplot(1,4,4)
tArrPlot = [10^(-3),0.5*10^(-1),0.1,1]; tArrPlot = tArrPlot*T;
for i = 1:numel(tArrPlot)
    [~,I] = min(abs(tArr-tArrPlot(i)));
    lgd = sprintf('$t = %.1f$',tArrPlot(i));
    plot(betaArr,mean(squeeze(cStore(I,:,:)),1),'Linewidth',5,'DisplayName',lgd); hold on
end 
hold off
xticks([-pi/2,0,pi/2]); xticklabels({'-\pi/2 ','0','\pi/2'})
set(gca,'LineWidth',2,'FontSize',fntSz)
xlabel('$ \beta $','FontSize',fntSz,'Interpreter','latex')
ylabel('$ <c_i(\beta,t)>$','FontSize',fntSz,'Interpreter','latex')
yy = legend; yy.EdgeColor = 'none';yy.Interpreter = 'latex';
pbaspect([1 1 1]);

%% Visualize the simulations

clc; close all; Nplots = 40; 

f= figure('color','k','Units','normalized','OuterPosition',[0.0031 0.0056 0.9938 1]);

for ct = 2:round((Nt-1)/Nplots):Nt-1

    b_ct = squeeze(bStoreBody(:,:,ct)); %Material frame 
    rotMatr_ct = squeeze(rotMatrStore(:,:,ct));
    omega_ct = squeeze(omegaStoreBody(:,ct));
    c_ct = squeeze(cStore(ct,:,:));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
    %%  Plot the egg chamber  
    subplot(2,4,[1,2,5,6])
    rManifoldEulerian_ct = rotMatr_ct*rManifoldBody;
    rPosCellsEulerian_ct = rotMatr_ct*rPosCellsBody;

    p = trisurf(triMesh,-rManifoldEulerian_ct(3,:),rManifoldEulerian_ct(2,:),rManifoldEulerian_ct(1,:),'FaceColor',[0.5 0.5 0.5],'EdgeColor','none');
    p.SpecularStrength = 0.2; p.AmbientStrength = 0.3;

    for k = 1:nCells
        hold on
        X = voronoiboundary{k}; logic = squeeze(X(3,:))<cos(thetaS);
        X = X(:,logic); X = rotMatr_ct*X;
        fill3(-X(3,:),X(2,:),X(1,:),'g','FaceAlpha',0,'EdgeColor','k','LineWidth',1);hold off
    end
    
    % plot the protrusions 
    bEulerian = rotMatr_ct*b_ct;
    hold on; quiver3(-rPosCellsEulerian_ct(3,:),rPosCellsEulerian_ct(2,:),...
        rPosCellsEulerian_ct(1,:),-bEulerian(3,:),bEulerian(2,:),...
        bEulerian(1,:),'Color',[0.466 0.674 0.188],'LineWidth',1);


    % Plot the stalk 
    rStalkCircleBody = [xStalk;yStalk;zStalk];
    rStalkCircleEulerian = rotMatr_ct*rStalkCircleBody;

    hold on;trisurf(TStalk, -rStalkCircleEulerian(3,:), rStalkCircleEulerian(2,:),...
        rStalkCircleEulerian(1,:), 'FaceColor',[0.5 0.5 0.5], 'EdgeColor', 'none'); hold off
     camlight; axis equal off;

    
    % Plot the long and omega axis 
    nVecLong = rotMatr_ct*[0;0;1]; scaleRotAxis = 1.4;
    nVecOmega = rotMatr_ct*omega_ct;nVecOmega = nVecOmega./norm(nVecOmega);

    x1Long =  scaleRotAxis*nVecLong(1); x2Long =  -scaleRotAxis*nVecLong(1);
    y1Long =  scaleRotAxis*nVecLong(2); y2Long =  -scaleRotAxis*nVecLong(2);
    z1Long =  scaleRotAxis*nVecLong(3); z2Long =  -scaleRotAxis*nVecLong(3);

    x1Omega =  scaleRotAxis*nVecOmega(1); x2Omega =  -scaleRotAxis*nVecOmega(1);
    y1Omega =  scaleRotAxis*nVecOmega(2); y2Omega =  -scaleRotAxis*nVecOmega(2);
    z1Omega =  scaleRotAxis*nVecOmega(3); z2Omega =  -scaleRotAxis*nVecOmega(3);
    hold on; p1 = plot3([x1Long,x2Long],[y1Long,y2Long],[z1Long,z2Long],'r','LineWidth',2,'DisplayName','AP axis'); hold on
    p2 = plot3([x1Omega,x2Omega],[y1Omega,y2Omega],[z1Omega,z2Omega],'w','LineWidth',2,'DisplayName','Rotational axis'); hold on
    p3 = plot3([0,0],[0,0],[-scaleRotAxis,scaleRotAxis],'cyan','LineWidth',2,'DisplayName','Stalk axis'); hold off
    rotate(p1,[0 1 0],90); rotate(p2,[0 1 0],90); rotate(p3,[0 1 0],90); 

    % Plot a cylinder 
    r = sin(thetaS); h = 0.2; [xCyl,yCyl,zCyl] = cylinder(r,100); zCyl = zCyl*h;
    [FCyl,VCyl] = surf2patch(xCyl,yCyl,zCyl,'triangles');
    hold on; p4 = trisurf(FCyl,-VCyl(:,3)-cos(thetaS),VCyl(:,2),VCyl(:,1), 'FaceColor',colorOrange, 'EdgeColor', 'none','DisplayName','Stalk'); hold on
    trisurf(TStalk, -zStalk, yStalk, xStalk, 'FaceColor',colorOrange, 'EdgeColor', 'none','FaceAlpha',1); hold on % Face 1
    trisurf(TStalk, -zStalk-h, yStalk, xStalk, 'FaceColor',colorOrange, 'EdgeColor', 'none','FaceAlpha',1); hold off % Face 2
    view(camView);camlight; camlight(camView2(1),camView2(2))
    
    % The properties and axis specifications 
    legend([p1,p2,p3,p4],'TextColor','w','FontSize',fntSz,'color','none','Interpreter','latex','EdgeColor','none')
    set(gca,'color','k','FontSize',fntSz); xlim([-1.5, 1.5]);ylim([-1.5, 1.5]);zlim([-1.5, 1.5]);camva(5)
    


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    cr = 50;

    subplot(2,4,3) % Plot the orientation of the long axis
    polarplot(rad2deg(orientLongAxis(1,1:cr:ct)),rad2deg(orientLongAxis(2,1:cr:ct)),'w','LineWidth',1); hold on 
    polarscatter(rad2deg(orientLongAxis(1,1:cr:ct)),rad2deg(orientLongAxis(2,1:cr:ct)),120,tArr(1:cr:ct),'filled'); hold on
    c = colorbar;  colormap('turbo'); c.FontSize = fntSz; c.Color = 'w'; clim([0 T]);
    set(gca,'color','k','FontSize',fntSz,'Rcolor','w','Thetacolor','w');
    rlim([0 180 ]);rticks([90,180]); rticklabels({'90','180'})
    title('AP Axis Orientation','Interpreter','latex','FontSize',fntSz,'Color','w'); 

    subplot(2,4,4) % Plot the orientation of the omega axis
    polarplot(rad2deg(orientOmegaAxis(1,1:cr:ct)),rad2deg(orientOmegaAxis(2,1:cr:ct)),'w','LineWidth',1); hold on 
    polarscatter(rad2deg(orientOmegaAxis(1,1:cr:ct)),rad2deg(orientOmegaAxis(2,1:cr:ct)),120,tArr(1:cr:ct),'filled'); hold on
    c = colorbar;  colormap('turbo'); c.FontSize = fntSz; c.Color = 'w';clim([0 T]);
    set(gca,'color','k','FontSize',fntSz,'Rcolor','w','Thetacolor','w');
    rlim([0 180 ]); rticks([90,180]); rticklabels({'90','180'})
    title('Omega Axis Orientation','Interpreter','latex','FontSize',fntSz,'Color','w'); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    subplot(2,4,7) % The polarity distribution of the cells 
    plot(betaArr,mean(c_ct,1),'LineWidth',2,'Color','w')
    xlabel('$ \beta $','Interpreter','latex');
    ylabel('$ <c_i(\beta,t)>$','Interpreter','latex');
    xticks([-pi/2,0,pi/2]); xticklabels({'-\pi/2 ','0','\pi/2'})

    set(gca,'color','k','FontSize',fntSz,'Xcolor','w','Ycolor','w');
    xlim([-pi,pi]); ylim([0, 1.5]);pbaspect([1 1 1]);
    title('Fat2 distribution','Interpreter','latex','FontSize',fntSz,'Color','w'); 

    %%%%%%%%%%%%%%%%%%%%%%%%%
    subplot(2,4,8) % Omega visualization 
    omegaModMaterial = vecnorm(omegaStoreBody(:,1:ct),2,1);
    p1 = plot(tArr(1:ct),omegaModMaterial(1:ct),'r','LineWidth',2,'DisplayName','$ |\omega| $'); hold on 
    p2 = plot(tArr(1:ct),abs(omegaStoreBody(3,1:ct)),'b','LineWidth',2,'DisplayName','$ |\omega_3| $'); hold off
    legend([p1,p2],'TextColor','w','FontSize',fntSz,'color','none','Interpreter','latex','EdgeColor','none','Location','southeast')
    set(gca,'color','k','FontSize',fntSz,'Xcolor','w','Ycolor','w');
    xlim([0,max(tArr)]); ylim([0,max(vecnorm(omegaStoreBody,2,1))]);
    title('$ \omega$ profile','Interpreter','latex','FontSize',fntSz,'Color','w'); 
    xlabel('$ t $','Interpreter','latex'); pbaspect([1 1 1]);
    ylabel('$ |\omega |$','Interpreter','latex');

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Include a description of the parameters 
    annstr0 = 'Parameters';
    annstr1 = sprintf('$ \\tau_1 = %.1f $, $ \\tau_2 = %.1f $, $ \\tau_3 = %.1f $',tau1,tau2,tau3);
    annstr2 = sprintf('$ \\theta_s = %.1f $(deg), $ K = %.1f $, $ \\mu = %.1f$',rad2deg(thetaS),kStalk,mu);
    annpos = [0.05 0.05 0.1 0.1];
    ha = annotation('textbox',annpos,'string',{annstr0,annstr1,annstr2});
    ha.Color = 'w'; ha.Interpreter = 'latex'; ha.FontSize = fntSz;

    sgtitle(sprintf('$$ t = %.1f $$',tArr(ct)),'Interpreter','latex','FontSize',30,'Color','w'); 
    pause(0.1)
end



