function fricAngMmntCoeff = getfricCoeffAngMomentum(thetaS,mu)

% drag with the basement membrane 
fricBaseXY = (pi/12)*(16+15*cos(thetaS)+cos(3*thetaS));
fricBaseZ = (pi/12)*(32*cos(thetaS/2)^4)*(2-cos(thetaS)); 

% drag with the stalk 
fricStalkXY = mu*pi*(sin(thetaS)*cos(thetaS))^2;
fricStalkZ = mu*(pi/2)*sin(thetaS)^4; 

fricAngMmntCoeff(1) = fricBaseXY+fricStalkXY;
fricAngMmntCoeff(2) = fricAngMmntCoeff(1);
fricAngMmntCoeff(3) = fricBaseZ+fricStalkZ;

end 