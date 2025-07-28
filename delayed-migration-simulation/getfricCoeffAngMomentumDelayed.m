function fricAngMmntCoeff = getfricCoeffAngMomentum(ecc)
% This function outputs the friction coefficient for the motion of the egg
% chamber - given different egg geometry 
% For ecc > 1 - motion is restricted to the long axis 
% For ecc = 1 - there is no restriction on the rotation axis 
 
fricAngMmntCoeff = nan(3,1);
if ecc > 1
    aZTang = 8*pi/3+26*pi*(ecc-1)/15;

    fricAngMmntCoeff(1) = Inf;
    fricAngMmntCoeff(3) = aZTang;
    fricAngMmntCoeff(2) = Inf;

elseif ecc == 1
    aZTang = 8*pi/3;

    fricAngMmntCoeff(1) = aZTang;
    fricAngMmntCoeff(3) = aZTang;
    fricAngMmntCoeff(2) = aZTang;
end 


end 