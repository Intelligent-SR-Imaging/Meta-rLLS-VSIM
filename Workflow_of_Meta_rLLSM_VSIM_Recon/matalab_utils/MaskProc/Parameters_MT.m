function pParams = Parameters_MT()
pParams.saveRmFbgFlag = true;
pParams.gsWindow = 15; % pxls
pParams.scaleFactor = 5; % width scale of mask_b
pParams.sigama = 1; % weight scalar of mask_a
pParams.deltaDispX = 1; % pxls
pParams.deltaDispA = 0.1; % degrees
pParams.modAmp = 0.3; % modamp of total mask, [0-0.5]
pParams.nAverage = 21; % should be odd
pParams.fixCoverslipAngle = -2; % tilts of coverslip along x-axis
pParams.maxIters = 3;
pParams.showFlag = 0;
pParams.rotAngle = 30.8;
pParams.background = 100;
end