function protrTest = generateProtrusions(zeta1,zeta2,prob_pt,betaArr)
% Generates the inital protrusion distribution given a fat2 distribution

nCells = size(zeta1,2); 
gRandomDirection = zeros([1,nCells]);

for i = 1:nCells
    prob_pti = prob_pt(i,:);
    gRandomDirection(i) = randsample(betaArr,1,true,prob_pti);
end 

gRandomDirection = [gRandomDirection;gRandomDirection;gRandomDirection];
protrTest = cos(gRandomDirection).*zeta1+sin(gRandomDirection).*zeta2;

end 