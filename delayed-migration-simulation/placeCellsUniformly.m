function rPosCells = placeCellsUniformly(N_tot,ecc)
% Defines the location of each cell centroid on the egg chaber surface
% given the total number of cells 

R = 1; xArr_for = []; yArr_for = []; zArr_for = []; N_count = 0;
areaElem = 4*pi*R^2/N_tot; dElem = sqrt(areaElem);
M_nu = round(pi/dElem);
d_nu = pi/M_nu; d_phi = areaElem/d_nu;

for m = 0:M_nu-1
    nu_m = pi*(0.5+m)/M_nu;
    M_phi = round(2*pi*sin(nu_m)/d_phi);
    for n=0:M_phi-1
        phi_n = 2*pi*n/M_phi;
        xArr_for = [xArr_for;R*sin(nu_m)*cos(phi_n)];
        yArr_for = [yArr_for;R*sin(nu_m)*sin(phi_n)];
        zArr_for = [zArr_for;R*ecc*cos(nu_m)];
        N_count = N_count+1;
    end
end

rPosCells = [xArr_for,yArr_for,zArr_for]';


end 