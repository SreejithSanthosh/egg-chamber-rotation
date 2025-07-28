function polarOP = calculatepolarorder(vx,vy,thresh)

vMag = sqrt(vx.^2+vy.^2);
vMagSum = 0; polarX = 0; polarY = 0;
for i = 1:numel(vMag)
    vMag_i = vMag(i);

    if vMag_i>prctile(vMag,thresh)
        vMagSum = vMagSum+vMag_i; 
        polarX = polarX+vx(i); polarY = polarY+vy(i);
     
    end 
end 

 polarOP = sqrt(polarX^2+polarY^2)/vMagSum;
end 