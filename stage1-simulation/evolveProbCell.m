function dprobpt_dt = evolveProbCell(zeta1,zeta2,rPosCellsBody,prob_pt,betaArr,omegaStoreBody_ct,tau2,tau3)
% This function simulates the evolution of the c(\beta,t) equation

dprobpt_dt = 0*prob_pt;
nCells = size(prob_pt,1); nBeta= size(prob_pt,2);

% Find the velocity of each cell in body frame
omegaMatr = repmat(omegaStoreBody_ct,1,nCells);
vCell = cross(omegaMatr,rPosCellsBody); 

for i = 1:nCells
    betaVec = cos(betaArr).*zeta1(:,i)+sin(betaArr).*zeta2(:,i);
    vCellVec = repmat(vCell(:,i),1,nBeta); term = dot(vCellVec,betaVec);
    dprobpt_dt(i,:) = -tau2*(tau3*term+ prob_pt(i,:)).*prob_pt(i,:);
end 

end 