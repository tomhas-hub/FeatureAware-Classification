function v3BlockGMMsAndTarFeatGMMGenAtTarFeatTS()
    clear all;
    clc;

    %计算程序运行时间.
    tic;
    
    
    %---------%only information here needs to be provided by user.----------%
    %dataSource = 'D:\research\data\3DFlowAroundAConfinedSquareCylinder\SquareCylinder\';  
    dataSource = 'D:\Graduate_Design\Data\3DFlow\';
    %saveDataPath = 'D:\research\codes\DDExtractionAndTracking\SquareCylinder\';
    saveDataPath = 'D:\FeatureAware-Classification\build_GMM(Teacher-Version)\build_GMM(Teacher-Version)\NewSquareCylinder\';
    charDataType = '*char*1';
    singleDataType = 'single';
    xBlockSize = 4;
    yBlockSize = 4;
    zBlockSize = 4;
    numGMMComponents = 3;   %papameter obtained from the paper.
    regularizationValue = 0.01; %
    %the region is selected at time step flow_t0208.am.
    %Note: selectedRegion = [xLowBound, xHighBound; yLowBound, yHighBound; zLowBound, zHighBound] (all begins at 1);
	%selectedRegion = [77, 83; 22, 25; 3, 43]; %特征范围
    selectedRegion = [75, 87; 18, 23; 3, 43]; %特征范围v8
    %selectedRegion = [80, 86; 17, 21; 3, 43]; %特征范围
    fileStartVal = 8;
    fileIncrement = 40;
    %tarFeatTimeStep = 5;  %the time step where the target feature is selected (note: time step begins at 0).
    tarFeatTimeStep = 10;
    %tarFeatTimeStep = 23; %flowt_0928
    %---------%only information here needs to be provided by user.----------%
    
    
    
    %------------(A) here is the code to generate velMag + blockGMMs at tarFeat time step.------------%
    %1. read the .am file at tarFeat time step.
    fileName = sprintf('flow_t%.4d', fileStartVal + tarFeatTimeStep * fileIncrement);
    
    [data, xDim, yDim, zDim, numVecComponents, xmin, xmax, ymin, ymax, zmin, zmax, isUniform] =...
        AmiraMeshReader(strcat(dataSource, fileName, '.am'), charDataType, singleDataType);
    if(isempty(data) == 1)    %if data == [].
        fprintf('Fail to read %s.\n', strcat(dataSource, fileName));
        return;
    end
    
    %once successfully read the .am data at tarFeat time step, printf all output parameters.
    fprintf("Read %s successfully.\n", strcat(dataSource, fileName));
    fprintf("Grid Dimensions: %d %d %d.\n", xDim, yDim, zDim);
    fprintf("BoundingBox in x-Direction: [%f ... %f].\n", xmin, xmax);
    fprintf("BoundingBox in y-Direction: [%f ... %f].\n", ymin, ymax);
    fprintf("BoundingBox in z-Direction: [%f ... %f].\n", zmin, zmax);
    fprintf('Uniform Grid: %s.\n', string(isUniform));
    fprintf('Number of Vector Components: %d.\n', numVecComponents);
    
    
    %{
    %********test: output the binary data.********
    %Note: Data runs x-fastest, i.e., the loop over the x-axis is the
    %innermost.
    for k = 1:zDim
        for j = 1:yDim
            for i = 1:xDim
                for c = 1:numComponents
                    fprintf('%f, ', data(((i-1) + (j-1) * xDim + (k-1) * xDim * yDim) * numComponents + c));
                end
                fprintf('\n');
            end
        end
    end
    %********test: output the binary data.********
    %}
    
    
    %2. once successfully read the .am data at tarFeat time step, process it. 
    %(2.1)compute the velocity magnitude at tarFeat time step.
    velMag = zeros(xDim, yDim, zDim); %将三维体数据存储在这个三维数组中
    %velocityMag = zeros(xDim * yDim * zDim, 1);
    for k = 1:zDim
        for j = 1:yDim
            for i = 1:xDim
                %for each voxel v(i, j, k):
                velMag_ijk = 0;
                for c = 1:numVecComponents
                    velMag_ijk = velMag_ijk + (data(((i-1) + (j-1) * xDim + (k-1) * xDim * yDim) * numVecComponents + c))^2;
                end %end c.
                velMag_ijk = sqrt(velMag_ijk);
                
                %assign velMag_ijk to velMag(i, j , k).
                velMag(i, j, k) = velMag_ijk;
                %velocityMag((i-1) + (j-1) * xDim + (k-1) * xDim * yDim + 1, 1) = velMag_ijk;
            end %end i.
        end %end j.
    end %end k.
    
    
    %(2.2)find velMag/velocityMag's min and max values at tarFeat time step.
    minVal_velMag = min(min(min(velMag)));
    maxVal_velMag = max(max(max(velMag)));
    fprintf('minVal_velMag: %f, maxVal_velMag: %f.\n', minVal_velMag, maxVal_velMag);
    %minVal_velocityMag = min(min(min(velocityMag)));
    %maxVal_velocityMag = max(max(max(velocityMag)));
    %fprintf('minVal_velocityMag: %f, maxVal_velocityMag: %f.\n', minVal_velocityMag, maxVal_velocityMag);
    
    
    %(2.3)given velocityMag at tarFeat time step, for each 4 * 4 * 4 block, compute its GMM at tarFeat time step.
    xBlockDim = xDim / xBlockSize;  %192/4=48.
    yBlockDim = yDim / yBlockSize;  %64/4=16.
    zBlockDim = zDim / zBlockSize;  %48/4=12.
    fprintf('xBlockDim: %d, yBlockDim: %d, zBlockDim: %d.\n', xBlockDim, yBlockDim, zBlockDim);
    
    BlockGMMmus = zeros(xBlockDim, yBlockDim, zBlockDim, numGMMComponents); %均值
    BlockGMMsigmas = zeros(xBlockDim, yBlockDim, zBlockDim, numGMMComponents); %方差
    BlockGMMcompProps = zeros(xBlockDim, yBlockDim, zBlockDim, numGMMComponents); %权重
    
    for k = 1:zBlockDim
        for j = 1:yBlockDim
            for i = 1:xBlockDim
                %for each block b(i, j, k):
                %(i)obtain this block's neighborhood data X.
                X = zeros(xBlockSize * yBlockSize * zBlockSize, 1); %X = zeros(64, 1);
                row = 1;
                for kk = 1:zBlockSize
                    for jj = 1:yBlockSize
                        for ii = 1:xBlockSize
                            %obtain each voxel id.
                            xVoxelId = (ii-1) + (i-1) * xBlockSize + 1; %range: [1-192].
                            yVoxelId = (jj-1) + (j-1) * yBlockSize + 1; %range: [1-64].
                            zVoxelId = (kk-1) + (k-1) * zBlockSize + 1; %range: [1-48].
                         
                            X(row, 1) = velMag(xVoxelId, yVoxelId, zVoxelId);
                            %X(row, 1) = velocityMag((xVoxelId-1) + (yVoxelId-1) * xDim + (zVoxelId-1) * xDim * yDim + 1, 1);
                            row = row + 1;
                        end
                    end
                end
                
                
                %(ii)given this block's neighborhood data X, obtain its
                %GMM.
                GMM = fitgmdist(X, numGMMComponents, 'RegularizationValue', regularizationValue);
                %turn the warnings off.
                warning('off');
                
                
                %(iii)save this block's GMM parameters into GMMmu(i, j, k, :), GMMsigma(i, j, k, :), GMMcompProp(i, j, k, :).
                for c = 1:numGMMComponents
                    BlockGMMmus(i, j, k, c) = GMM.mu(c, 1);
                    BlockGMMsigmas(i, j, k, c) = GMM.Sigma(:, :, c); %协方差矩阵
                    BlockGMMcompProps(i, j, k, c) = GMM.ComponentProportion(1, c);
                end
                                          
            end %end i.
        end %end j.
    end %end k.
    
    
    %3. once obtain velMag, BlockGMMmus, BlockGMMsigmas, BlockGMMcompProps at tarFeat time step, save them.
    %(3.1)save velMag as .raw file at tarFeat time step.
    fid = fopen(strcat(saveDataPath, fileName, '_velMag.raw'), 'w');
    numVoxels = fwrite(fid, velMag, singleDataType);
    if(numVoxels ~= (xDim * yDim * zDim))
        fprintf('Somthing wrong when saving velMag.raw.\n');
        fclose(fid);
        return;
    end
    fclose(fid);
    fprintf('velMag.raw has been saved.\n');
    
    %fid = fopen(strcat(saveDataPath, fileName, '_velocityMag.raw'), 'w');
    %numOfVoxels = fwrite(fid, velocityMag, singleDataType);
    %fclose(fid);
    %fprintf('velocityMag.raw has been saved.\n');
    
  
    %(3.2)save BlockGMMmus at tarFeat time step.
    fid = fopen(strcat(saveDataPath, fileName, '_BlockGMMmus.raw'), 'w');
    numVoxels = fwrite(fid, BlockGMMmus, singleDataType);
    if(numVoxels ~= (xBlockDim * yBlockDim * zBlockDim * numGMMComponents))
        fprintf('Somthing wrong when saving BlockGMMmus.raw.\n');
        fclose(fid);
        return;
    end
    fclose(fid);
    fprintf('BlockGMMmus.raw has been saved.\n');
    
    
    %(3.3)save BlockGMMsigmas at tarFeat time step.
    fid = fopen(strcat(saveDataPath, fileName, '_BlockGMMsigmas.raw'), 'w');
    numVoxels = fwrite(fid, BlockGMMsigmas, singleDataType);
    if(numVoxels ~= (xBlockDim * yBlockDim * zBlockDim * numGMMComponents))
        fprintf('Somthing wrong when saving BlockGMMsigmas.raw.\n');
        fclose(fid);
        return;
    end
    fclose(fid);
    fprintf('BlockGMMsigmas.raw has been saved.\n');
    
    
    %(3.4)save BlockGMMcompProps at tarFeat time step.
    fid = fopen(strcat(saveDataPath, fileName, '_BlockGMMcompProps.raw'), 'w');
    numVoxels = fwrite(fid, BlockGMMcompProps, singleDataType);
    if(numVoxels ~= (xBlockDim * yBlockDim * zBlockDim * numGMMComponents))
        fprintf('Somthing wrong when saving BlockGMMcompProps.raw.\n');
        fclose(fid);
        return;
    end
    fclose(fid);
    fprintf('BlockGMMcompProps.raw has been saved.\n\n');
    %------------(A) here is the code to generate velMag + blockGMMs at tarFeat time step.------------%
    
    
    %------------(B) here is the code to generate tarFeatGMM at tarFeat time step.------------%
    %4. read the .am file at selected time step to obtain the tarFeatGMM.
    fileName = sprintf('flow_t%.4d', fileStartVal + tarFeatTimeStep * fileIncrement);
    
    [data, xDim, yDim, zDim, numVecComponents, xmin, xmax, ymin, ymax, zmin, zmax, isUniform] =...
        AmiraMeshReader(strcat(dataSource, fileName, '.am'), charDataType, singleDataType);
    if(isempty(data) == 1)    %if data == [].
        fprintf('Fail to read %s.\n', strcat(dataSource, fileName));
        return;
    end
    
    
    %5. once successfully read the am. data at tarFeat time step, compute its velMag_tarFeatTimeStep at this time step.
    velMag_tarFeatTimeStep = zeros(xDim, yDim, zDim);
    %velocityMag = zeros(xDim * yDim * zDim, 1);
    for k = 1:zDim
        for j = 1:yDim
            for i = 1:xDim
                %for each voxel v(i, j, k):
                velMag_ijk = 0;
                for c = 1:numVecComponents
                    velMag_ijk = velMag_ijk + (data(((i-1) + (j-1) * xDim + (k-1) * xDim * yDim) * numVecComponents + c))^2;
                end %end c.
                velMag_ijk = sqrt(velMag_ijk);
                
                %assign velMag_ijk to velMag(i, j , k).
                velMag_tarFeatTimeStep(i, j, k) = velMag_ijk;
                %velocityMag((i-1) + (j-1) * xDim + (k-1) * xDim * yDim + 1, 1) = velMag_ijk;
            end %end i.
        end %end j.
    end %end k.
    
    
    %find min. and max. values of velMag_tarFeatTimeStep at tarFeat time
    %step.
    minVal_velMag_tarFeatTimeStep = min(min(min(velMag_tarFeatTimeStep)));
    maxVal_velMag_tarFeatTimeStep = max(max(max(velMag_tarFeatTimeStep)));
    fprintf('minVal_velMag_tarFeatTimeStep: %f, maxVal_velMag_tarFeatTimeStep: %f.\n', minVal_velMag_tarFeatTimeStep, maxVal_velMag_tarFeatTimeStep);
    
    
    
    %6. given the velMag_tarFeatTimeStep + selected region at tarFeat time step, compute its target feature GMM.
    %(6.1)store the selected region's velocity magnitude 
    %into X_selectedRegion.
    X_selectedRegion = zeros((selectedRegion(1, 2)-selectedRegion(1, 1)+1) * (selectedRegion(2, 2)-selectedRegion(2,1)+1) * (selectedRegion(3,2)-selectedRegion(3,1)+1), 1);
    row_selectedRegion = 1;
    for k = selectedRegion(3, 1):selectedRegion(3, 2)
        for j = selectedRegion(2, 1):selectedRegion(2, 2)
            for i = selectedRegion(1, 1):selectedRegion(1, 2)
                %for each voxel (within the selected region):
                X_selectedRegion(row_selectedRegion, 1) = velMag_tarFeatTimeStep(i, j, k);
                row_selectedRegion = row_selectedRegion + 1;
            end
        end
    end
    
    
    %(6.2)given the selected region's X_selectedRegion, compute its tarFeatGMM.
    tarFeatGMM = fitgmdist(X_selectedRegion, numGMMComponents, 'RegularizationValue', regularizationValue);
    
    
    %(6.3)save the tarFeatGMM into tarFeatGMMmu, tarFeatGMMsigma and tarFeatGMMcompProp.
    %tfGMM = zeros(numGMMComponents, numGMMComponents);
    tarFeatGMMmu = zeros(numGMMComponents, 1);
    tarFeatGMMsigma = zeros(numGMMComponents, 1);
    tarFeatGMMcompProp = zeros(numGMMComponents, 1);
    for c = 1:numGMMComponents
        tarFeatGMMmu(c, 1) = tarFeatGMM.mu(c, 1);
        tarFeatGMMsigma(c, 1) = tarFeatGMM.Sigma(:, :, c);
        tarFeatGMMcompProp(c, 1) = tarFeatGMM.ComponentProportion(1, c);
    end
    %Test: print tarFeatMM.
    tarFeatGMMmu
    tarFeatGMMsigma
    tarFeatGMMcompProp
    %Test: print tarFeatMM.
    
    
    %(6.4)save tarFeatGMMmu, tarFeatGMMsigma and tarFeatGMMcompProp as .raw files.
    %(6.4.1)save tarFeatGMMmu as .raw.
    fid = fopen(strcat(saveDataPath, fileName, '_tarFeatGMMmu.raw'), 'w');
    numVoxels = fwrite(fid, tarFeatGMMmu, singleDataType);
    if(numVoxels ~= numGMMComponents)
        fprintf('Somthing wrong when saving tarFeatGMMmu.raw.\n');
        fclose(fid);
        return;
    end
    fclose(fid);
    fprintf('tarFeatGMMmu.raw has been saved.\n');
    
    %(6.4.2)save tarFeatGMMsigma as .raw.
    fid = fopen(strcat(saveDataPath, fileName, '_tarFeatGMMsigma.raw'), 'w');
    numVoxels = fwrite(fid, tarFeatGMMsigma, singleDataType);
    if(numVoxels ~= numGMMComponents)
        fprintf('Somthing wrong when saving tarFeatGMMsigma.raw.\n');
        fclose(fid);
        return;
    end
    fclose(fid);
    fprintf('tarFeatGMMsigma.raw has been saved.\n');
    
    %(6.4.3)save tarFeatGMMcompProp as .raw.
    fid = fopen(strcat(saveDataPath, fileName, '_tarFeatGMMcompProp.raw'), 'w');
    numVoxels = fwrite(fid, tarFeatGMMcompProp, singleDataType);
    if(numVoxels ~= numGMMComponents)
        fprintf('Somthing wrong when saving tarFeatGMMcompProp.raw.\n');
        fclose(fid);
        return;
    end
    fclose(fid);
    fprintf('tarFeatGMMcompProp.raw has been saved.\n');
    %------------(B) here is the code to generate tarFeatGMM at tarFeat time step.------------%
    
    
    %计算程序运行时间.
    executionTime = toc;
    fprintf('main.m execution time = %fs.\n', executionTime);
end


function [data, xDim, yDim, zDim, numVecComponents, xmin, xmax, ymin, ymax, zmin, zmax, isUniform] = ...
    AmiraMeshReader(fileName, charDataType, singleDataType)
    %initialize output parameters.
    data = [];
    xDim = -1;
    yDim = -1;
    zDim = -1;
    numVecComponents = -1;
    xmin = 1;
    xmax = -1;
    ymin = 1;
    ymax = -1;
    zmin = 1;
    zmax = -1;
    isUniform = false; %the default grid type is not uniform.


    %1. open a .am file.
    fid = fopen(fileName, 'r');
    if(fid == -1) %fail to open .am file.
        fprintf('Could not find %s.\n', fileName);
        return;
    end   
    
    
    %2. we read the first 2k bytes into memory to parse the header.
    %The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
    buffer = fread(fid, [1 2047], charDataType);
   
    
    %(2.1)check if it is an AmiraMesh file.
    pos = strfind(buffer, '# AmiraMesh BINARY-LITTLE-ENDIAN 2.1');
    if(isempty(pos) == 1) %if not AmiraMesh file.
        fprintf("Not a proper AmiraMesh file.\n");
        fclose(fid);
        return;
    end
    
   
    %(2.2)obtain the grid dimensions.
    newStr = extractAfter(buffer, 'define Lattice');
    dims = sscanf(newStr, '%d %d %d');
    xDim = dims(1);
    yDim = dims(2);
    zDim = dims(3);
    %fprintf("Grid Dimensions: %d %d %d.\n", xDim, yDim, zDim);

    
    %(2.3)obtain the BoundingBox.
    newStr = extractAfter(buffer, 'BoundingBox');
    bBox = sscanf(newStr, '%f %f %f %f %f %f');
    xmin = bBox(1);
    xmax = bBox(2);
    ymin = bBox(3);
    ymax = bBox(4);
    zmin = bBox(5);
    zmax = bBox(6);
    %fprintf("BoundingBox in x-Direction: [%f ... %f].\n", xmin, xmax);
    %fprintf("BoundingBox in y-Direction: [%f ... %f].\n", ymin, ymax);
    %fprintf("BoundingBox in z-Direction: [%f ... %f].\n", zmin, zmax);
    
   
    %(2.4)check if it is a uniform grid.
    pos = strfind(buffer, 'CoordType "uniform"');
    isUniform = true;
    if(isempty(pos) == 1) %if not a uniform grid.
        isUniform = false;
        fprintf('Uniform Grid: %s.\n', string(isUniform));
        fclose(fid);
        return;
    end
    %fprintf('Uniform Grid: %s.\n', string(isUniform));
    
    
    %(2.5)obtain numVecComponents.
    newStr = extractAfter(buffer, 'Lattice { float[');
    numVecComponents = sscanf(newStr, '%d');
    %fprintf('Number of Components: %d.\n', numComponents);
    
    
    %(2.6)position to the data section, and read data from there.
    pos = strfind(buffer, '# Data section follows');
    %set the file pointer to the beginning of "# Data section follows".
    fseek(fid, pos, 'bof');
    %consume this line, which is "# Data section follows".
    oneLine = fgets(fid);
    %consume the next line, which is "@1".
    oneLine = fgets(fid);
    
    
    %3. read the binary data, and close the .am file.
    [data, numToRead] = fread(fid, xDim * yDim * zDim * numVecComponents, singleDataType);
    if(numToRead ~= (xDim * yDim * zDim * numVecComponents))
        printf('Something wrong while reading the binary data.\n');
        fclose(fid);
        data = NaN;
        return;
    end
    fclose(fid);
    
    
    %{
    %********test: output the binary data.********
    %Note: Data runs x-fastest, i.e., the loop over the x-axis is the
    %innermost.
    for k = 1:zDim
        for j = 1:yDim
            for i = 1:xDim
                for c = 1:numComponents
                    fprintf('%f, ', data(((i-1) + (j-1) * xDim + (k-1) * xDim * yDim) * numComponents + c));
                end
                fprintf('\n');
            end
        end
    end
    %********test: output the binary data.********
    %}
end