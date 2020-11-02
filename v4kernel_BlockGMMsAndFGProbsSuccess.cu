#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <math.h>
#include <time.h>
#include "image_components.h" //come from https://people.sc.fsu.edu/~jburkardt/c_src/image_components/image_components.html
#include "image_components.c"	//come from https://people.sc.fsu.edu/~jburkardt/c_src/image_components/image_components.html


using namespace std;


//define constants.
#define		PIE						3.1415926 
#define		NUMGMMCOMPONENTS		3
#define		SDPARAMETER				2.5/2
#define		BETA					0.2		//BETA =learning rate in the paper.
#define		INITIALSD				10000
#define		INITIALWEIGHT			0.001
#define		NUMFILES				102
#define		FILESTARTVAL			8
#define		FILEINCREMENT			40
#define		THRESHOLD_WEIGHT		0.2	//did not mention in this paper.
//#define		TARFEATTIMESTEP			5	//the time step where the target feature is selected (note: time step begins at 0).
#define		TARFEATTIMESTEP			10 // flow_t0408
#define		GAMMA					0.7	//according to the paper.
#define		THRESHOLD_POSS			0.8	//according to the paper.



//add on 2020/9/18.
struct perBlockGMM
{
	//GMM数据结构
	float mu[NUMGMMCOMPONENTS];
	float sigma[NUMGMMCOMPONENTS];
	float compProp[NUMGMMCOMPONENTS];
};
//add on 2020/9/18.



void findMinAndMaxVals(dim3 numBlocks, float *minVal, float *maxVal, float *data)
{
	//找出一维数组中的最大最小值
	//*注意
	for (int k = 0; k < numBlocks.z; k++)
	for (int j = 0; j < numBlocks.y; j++)
	for (int i = 0; i < numBlocks.x; i++)
	{
		//for each block b(i, j, k):
		float bVal = data[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y];

		if (bVal < *minVal)
		{
			*minVal = bVal;
		}

		if (bVal > *maxVal)
		{
			*maxVal = bVal;
		}

	}

}


void saveToRawFile(char *rawFileName, void *data, char dataType, int size)
{
	//将data写入raw文件中
	//Note: 
	//(i)fopen第一个参数必须加(char *), 否则错误; 
	//(ii)fopen第二个参数必须为"wb"而非"w", 否则错误).
	FILE *fp = fopen(rawFileName, "wb");

	//fail to open .raw file.
	if (fp == NULL)
	{
		printf("Fail to open %s file.\n", rawFileName);
		exit(1);	//terminate with error.
	}

	//successfully open .raw file.
	size_t numOfVoxels;
	switch (dataType)
	{
	case 'u':	//unsigned char.
		numOfVoxels = fwrite(data, sizeof(unsigned char), size, fp);
		break;
	case 'f':	//float.
		numOfVoxels = fwrite(data, sizeof(float), size, fp);
		break;
	}

	if (numOfVoxels != size)
	{
		printf("Writing error!\n");
		exit(1);	//terminate with error.
	}
	fclose(fp);
	printf("%s has been saved.\n", rawFileName);
}



void* readVolumeData(char *rawFileName, size_t bytesPerVoxel, size_t numOfVoxels)
{
	//从raw文件中读取数据到data中
	FILE *fp = fopen(rawFileName, "rb");

	//fail to open .raw file
	if (!fp)
	{
		cout << "Fail to open " << rawFileName << endl;
		exit(1);	//terminate with error.
	}

	//successfully open .raw file
	void *data = malloc(bytesPerVoxel * numOfVoxels);
	//fread：按列读取
	size_t totalNumOfReadElements = fread(data, bytesPerVoxel, numOfVoxels, fp);
	if (totalNumOfReadElements != numOfVoxels)
	{
		cout << "Reading error!" << endl;
		cout << "Total number of elements successfully read is: " << totalNumOfReadElements << endl;
		exit(3);
	}
	fclose(fp);

	printf("Read %s successfully.\n", rawFileName);
	return data;
}



/** Find a string in the given buffer and return a pointer
to the contents directly behind the SearchString.
If not found, return the buffer. A subsequent sscanf()
will fail then, but at least we return a decent pointer.
*/
const char* FindAndJump(const char* buffer, const char* SearchString)
{
	const char* FoundLoc = strstr(buffer, SearchString);
	if (FoundLoc) return FoundLoc + strlen(SearchString);
	return buffer;
}



float* readAmiraMeshData(char *fileName, int xDim, int yDim, int zDim, int numVecComponents)
{
	//从一个am文件读取数据，并返回
	//1. open a .am file.
	FILE *fp = fopen(fileName, "rb");
	if (!fp)
	{
		printf("Could not find %s.\n", fileName);
		exit(1);
	}
	printf("Reading %s successfully.\n", fileName);


	//2. we read the first 2k bytes into memory to parse the header.
	//The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
	char buffer[2048];
	fread(buffer, sizeof(char), 2047, fp);
	buffer[2047] = '\0'; //The following string routines prefer null-terminated strings.


	//3. find the beginning of the data section.
	int idxStartData = strstr(buffer, "# Data section follows") - buffer;
	//printf("idxStarData: %d.\n", idxStartData);


	//set the file pointer to the beginning of "# Data section follows".
	fseek(fp, idxStartData, SEEK_SET);
	//consume this line, which is "# Data section follows".
	fgets(buffer, 2047, fp);
	//consume the next line, which is "@1".
	fgets(buffer, 2047, fp);


	//4. read the .am data.
	float *data = (float *)malloc(sizeof(float)* xDim * yDim * zDim * numVecComponents);
	size_t numElements = fread((void*)data, sizeof(float), xDim * yDim * zDim * numVecComponents, fp);
	if (numElements != (xDim * yDim * zDim * numVecComponents))
	{
		printf("Something wrong when reading the .am data section.\n");
		free(data);
		fclose(fp);
		exit(1);
	}


	/*
	////////Test: Print all data values.////////
	//Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
	printf("Printing all values in the same order in which they are in memory:\n");
	for (int k = 0; k < zDim; k++)
	for (int j = 0; j < yDim; j++)
	for (int i = 0; i < xDim; i++)
	{
	//Note: Random access to the value (of the first component) of the grid point (i,j,k):
	// pData[((k * yDim + j) * xDim + i) * NumComponents]
	//assert(pData[((k * yDim + j) * xDim + i) * NumComponents] == pData[Idx * NumComponents]);

	for (int c = 0; c < numVecComponents; c++)
	{
	printf("%f, ", data[(i + j * xDim + k * xDim * yDim) * numVecComponents + c]);
	}
	printf("\n");
	}
	////////Test: Print all data values.////////
	*/


	//5. once read data successfully, return it.
	fclose(fp);
	return data;
}



void parseAmiraMeshHeader(char *fileName, int *xDim, int *yDim, int *zDim, int *numVecComponents,
	float *xmin, float *xmax, float *ymin, float *ymax, float *zmin, float *zmax, bool *isUniform)
{
	//获取am文件的头数据
	//1. open a .am file.
	FILE *fp = fopen(fileName, "rb");
	if (!fp)
	{
		printf("Could not find %s.\n", fileName);
		exit(1);
	}
	printf("Reading %s successfully.\n", fileName);


	//2. we read the first 2k bytes into memory to parse the header.
	//The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
	char buffer[2048];
	fread(buffer, sizeof(char), 2047, fp);
	buffer[2047] = '\0'; //The following string routines prefer null-terminated strings.


	//(2.1)check if it is an AmiraMesh file.
	if (!strstr(buffer, "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1"))
	{
		printf("Not a proper AmiraMesh file.\n");
		fclose(fp);
		exit(1);
	}


	//(2.2)obtain the grid dimensions.
	sscanf(FindAndJump(buffer, "define Lattice"), "%d %d %d", xDim, yDim, zDim);
	//printf("Grid Dimensions: %d %d %d.\n", *xDim, *yDim, *zDim);


	//(2.3)obtain the BoundingBox.
	sscanf(FindAndJump(buffer, "BoundingBox"), "%f %f %f %f %f %f", xmin, xmax, ymin, ymax, zmin, zmax);
	//printf("BoundingBox in x-Direction: [%g ... %g]\n", *xmin, *xmax);
	//printf("BoundingBox in y-Direction: [%g ... %g]\n", *ymin, *ymax);
	//printf("BoundingBox in z-Direction: [%g ... %g]\n", *zmin, *zmax);


	//(2.4)check if it is a uniform grid.
	*isUniform = (strstr(buffer, "CoordType \"uniform\"") != NULL);
	//printf("Uniform Grid: %s\n", (*isUniform) ? "true" : "false");



	//(2.5)obtain numVecComponents.
	sscanf(FindAndJump(buffer, "Lattice { float["), "%d", numVecComponents);
	//printf("Number of Vector Components: %d.\n", *numVecComponents);


	//sanity check.
	if ((*xDim) <= 0 || (*yDim) <= 0 || (*zDim) <= 0
		|| (*xmin) > (*xmax) || (*ymin) > (*ymax) || (*zmin) > (*zmax)
		|| !(*isUniform) || (*numVecComponents) <= 0)
	{
		printf("Something went wrong when reading the .am file.\n");
		fclose(fp);
		exit(1);
	}


	/*
	//Test.
	//Find the beginning of the data section.
	int idxStartData = strstr(buffer, "# Data section follows") - buffer;
	printf("(comparison)idxStarData: %d.\n", idxStartData);
	//Test.
	*/


	fclose(fp);
}



__device__ void setBestMatchFlags(float *matchVals, int *bestMatchFlags)
{
	//Note:
	//标注分量是否为最佳匹配
	//与分量的均值越近，越匹配
	float minVal = 1000000;
	//float maxVal = -1;
	int minValIndex;

	//1. obtain the minimum value index within matchVals.
	for (int c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		if ((matchVals[c] != -1) && (matchVals[c] > minVal))
		{
			minValIndex = c;
			minVal = matchVals[c];
		}
	}


	//2. given the minimum value index, set bestMatchFlags.
	for (int c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		if (c == minValIndex)
			bestMatchFlags[c] = 1;
		else
			bestMatchFlags[c] = 0;
	}
}

//计算给定高斯参数和数据点对应的概率
__device__ float computeProbForDataPoint(float comProp,float mu, float sigma, float data)
{
	float prob = exp(-pow((data - mu) / sigma, 2) / 2.0f) / sqrtf(2 * PIE*sigma);
	return prob*comProp;
}

//return the minimum value index in 'distance' array.
__device__ int findLeastProbableGMMcomponentIndex(float *distance)
{
	//找出距离最远的高斯分量
	int minValIndex;
	float minVal = 100000;


	//find the index with maximum value.
	for (int c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		if (distance[c] < minVal)
		{
			minValIndex = c;
			minVal = distance[c];
		}
	}


	return minValIndex;
}

__global__ void printFirstBlockParams(float *dev_blockGMMmus_currentTimeStep, float *dev_blockGMMsigmas_currentTimeStep, float *dev_blockGMMcompProps_currentTimeStep, int i, int j, int k, dim3 numBlocks)
{
	// i,j,k 分块的坐标
	printf("Parameters of The [%d %d %d] GMM:\n",i,j,k);
	int c;
	printf("	GMMmus:");
	for (c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		printf("%lf ", dev_blockGMMmus_currentTimeStep[i + j*numBlocks.x + k*numBlocks.x*numBlocks.y+c*numBlocks.x*numBlocks.y*numBlocks.z]);
	}
	printf("\n");

	printf("	GMMsigmas:");
	for (c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		printf("%lf ", dev_blockGMMsigmas_currentTimeStep[i + j*numBlocks.x + k*numBlocks.x*numBlocks.y + c*numBlocks.x*numBlocks.y*numBlocks.z]);
	}
	printf("\n");

	printf("	GMMcomProps:");
	for (c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		printf("%lf ", dev_blockGMMcompProps_currentTimeStep[i + j*numBlocks.x + k*numBlocks.x*numBlocks.y + c*numBlocks.x*numBlocks.y*numBlocks.z]);
	}
	printf("\n");
}

//启动blockDim个threads: (48, 16, 12).
__global__ void computePerBlockGMMAndProbForeground_currentTimeStep(int xDim, int yDim, int zDim, dim3 numBlocks, dim3 blockSize,
	float *dev_blockGMMmus_lastTimeStep, float *dev_blockGMMsigmas_lastTimeStep, float *dev_blockGMMcompProps_lastTimeStep,
	float *dev_velMagVol_currentTimeStep,
	float *dev_blockGMMmus_currentTimeStep, float *dev_blockGMMsigmas_currentTimeStep, float *dev_blockGMMcompProps_currentTimeStep,
	float *dev_blockProbFG_currentTimeStep)
{
	//更新GMM参数，并计算出前景概率
	//for each block b(i, j, k):
	int i = threadIdx.x + blockIdx.x * blockDim.x;	//x range: [0 - 47].
	int j = threadIdx.y + blockIdx.y * blockDim.y;	//y rage: [0 - 15].
	int k = threadIdx.z + blockIdx.z * blockDim.z;	//z range: [0 - 11].


	//add on 2020/9/21.
	//1. declare variables to compute this block's probability.
	float numNewDataPoints = 0;
	float numTotalDataPoints = blockSize.x * blockSize.y * blockSize.z;
	//printf("numTotalDataPoints = %f.\n", numTotalDataPoints);
	//add on 2020/9/21.


	//add on 2020/9/21.
	//2. declare/initialize this block's GMM_lastTimeStep = dev_blockGMMxxs_lastTimeStep.
	perBlockGMM GMM_lastTimeStep;
	for (int c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		GMM_lastTimeStep.mu[c] = dev_blockGMMmus_lastTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c * numBlocks.x * numBlocks.y * numBlocks.z];
		GMM_lastTimeStep.sigma[c] = dev_blockGMMsigmas_lastTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c * numBlocks.x * numBlocks.y * numBlocks.z];
		GMM_lastTimeStep.compProp[c] = dev_blockGMMcompProps_lastTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c * numBlocks.x * numBlocks.y * numBlocks.z];
	}
	//add on 2020/9/21.


	//3. declare this block's GMM_beforeVoxel and GMM_afterVoxel,
	//and initialize the GMM_beforeVoxel = GMM_lastTimeStep.
	perBlockGMM GMM_beforeVoxel, GMM_afterVoxel;
	for (int c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		GMM_beforeVoxel.mu[c] = GMM_lastTimeStep.mu[c];
		GMM_beforeVoxel.sigma[c] = GMM_lastTimeStep.sigma[c];
		GMM_beforeVoxel.compProp[c] = GMM_lastTimeStep.compProp[c];
	}


	//loop each voxel (within this block), and use it to gradually obtain this block's 
	//(i)GMM parameters;
	//(ii)foreground probability.
	bool newlyCreated[3] = { false, false, false };
	for (int kk = 0; kk < blockSize.z; kk++)
	for (int jj = 0; jj < blockSize.y; jj++)
	for (int ii = 0; ii < blockSize.x; ii++)
	/*for (int kk = blockSize.z-1; kk >=0; kk--)
	for (int jj = blockSize.y-1; jj >=0; jj--)
	for (int ii = blockSize.x-1; ii >=0; ii--)*/
	{
		//for each voxel (within this block):
		//4. clear this block's GMM_afterVoxel to be 0, for later assignment.
		for (int c = 0; c < NUMGMMCOMPONENTS; c++)
		{
			GMM_afterVoxel.mu[c] = 0;
			GMM_afterVoxel.sigma[c] = 0;
			GMM_afterVoxel.compProp[c] = 0;
		}


		//5. obtain this voxel global id and its value.
		dim3 vid;
		vid.x = ii + i * blockSize.x;	//[0 - 191].
		vid.y = jj + j * blockSize.y;	//[0 - 63].
		vid.z = kk + k * blockSize.z;	//[0 - 47].
		float vVal = dev_velMagVol_currentTimeStep[vid.x + vid.y * xDim + vid.z * xDim * yDim];



		//here comes the incremental update scheme.
		//6. use this voxel value to change this block's GMM_beforeVoxel to obtain GMM_afterVoxel.	
		//(6.1)check this voxel value with 3 GMM components, respectively to obtain matchVals. 
		bool match = false;
		bool matchForProb1 = false;
		bool matchForProb3 = false;
		float matchVals[NUMGMMCOMPONENTS] = { -1, -1, -1 };
		for (int c = 0; c < NUMGMMCOMPONENTS; c++)
		{
			//matchVals[c] = -1;
			float lowBound = GMM_beforeVoxel.mu[c] - SDPARAMETER * GMM_beforeVoxel.sigma[c];
			float highBound = GMM_beforeVoxel.mu[c] + SDPARAMETER * GMM_beforeVoxel.sigma[c];

			//if the voxel value lies within the 2.5 SD of the component[c].
			if ((vVal >= lowBound) && (vVal <= highBound))
			{
				matchVals[c] = abs(vVal - GMM_beforeVoxel.mu[c]);
				//matchVals[c] = GMM_beforeVoxel.compProp[c];
				//如果该点数据匹配到一个新创建的高斯分量——标准差在INITIALSD附近，
				//即视为一个新数据点
				if (newlyCreated[c] == true)
				{
					matchForProb3 = true;
				}
				//如果该点数据匹配到至少一个权重大于阈值T的高斯分量，则不会视为一个新数据点
				if (GMM_beforeVoxel.compProp[c] > THRESHOLD_POSS)
				{
					matchForProb1 = true;
				}
				match = true;
			}
		}



		//(6.2)branch 1: if the voxel value matches one or more GMM_beforeVoxel components.
		if (match == true)
		{
			//(6.2.1)given matchVals, set bestMatchFlags (note:
			//the bestMatchFlags result includes one '1' and two '0').
			int bestMatchFlags[NUMGMMCOMPONENTS] = { 0, 0, 0 };
			setBestMatchFlags(matchVals, bestMatchFlags);
			//printf("bestMatchFlags: %d, %d, %d.\n", bestMatchFlags[0], bestMatchFlags[1], bestMatchFlags[2]);


			//(6.2.2)given bestMatchFlags, compute GMM_afterVoxel based on GMM_beforeVoxel using Equation (2)-(4).
			//(i)compute GMM_afterVoxel's weights using Equation (2).
			float sumWeights = 0.0f;
			for (int c = 0; c < NUMGMMCOMPONENTS; c++)
			{
				GMM_afterVoxel.compProp[c] = (1 - BETA) * GMM_beforeVoxel.compProp[c] + BETA * bestMatchFlags[c];
				sumWeights += GMM_afterVoxel.compProp[c];
				//printf("%lf ", GMM_afterVoxel.compProp[c]);
			}
			//printf("\nSumWeigts:%lf\n", sumWeights);
			//(ii)normalize GMM_afterVoxel weights.
			for (int c = 0; c < NUMGMMCOMPONENTS; c++)
			{
				GMM_afterVoxel.compProp[c] /= sumWeights;
			}

			//(iii)compute GMM_afterVoxel's means + standard deviations using Equation (3)-(4).
			for (int c = 0; c < NUMGMMCOMPONENTS; c++)
			{
				//if the GMM component is unmacthed, its means + standard deviations are unchanged.
				if (bestMatchFlags[c] == 0)
				{
					GMM_afterVoxel.mu[c] = GMM_beforeVoxel.mu[c];
					GMM_afterVoxel.sigma[c] = GMM_beforeVoxel.sigma[c];
				}
				//if the GMM component is matched, the mean + standard deviation are changed.
				else if (bestMatchFlags[c] == 1)
				{
					GMM_afterVoxel.mu[c] = (1 - BETA) * GMM_beforeVoxel.mu[c] + BETA * vVal;
					GMM_afterVoxel.sigma[c] = sqrtf((1 - BETA) * powf(GMM_beforeVoxel.sigma[c], 2) + BETA * powf((GMM_afterVoxel.mu[c] - vVal), 2));
				}
			}

		}//end branch 1.


		//(6.3)branch 2: if the voxel value does not match any GMM_beforeVoxel components.
		if (match == false)
		{
			//(6.3.1)find out the least probably GMM component index.
			float distance[NUMGMMCOMPONENTS];
			for (int c = 0; c < NUMGMMCOMPONENTS; c++)
			{
				//distance[c] = abs(vVal - GMM_beforeVoxel.mu[c]);
				distance[c] = GMM_beforeVoxel.compProp[c];
			}

			int leastProbGMMcomponentIndex = findLeastProbableGMMcomponentIndex(distance);
			//printf("matchVals: %f, %f, %f; leaseProbGMMcomponentIndex: %d.\n", 
			//	matchVals[0], matchVals[1], matchVals[2], leastProbGMMcomponentIndex);

			//将该高斯分量标注为新创建的
			newlyCreated[leastProbGMMcomponentIndex] = true;

			//(6.3.2)given the least probable GMM component index,
			//generate the GMM_afterVoxel's weights, mus and sigmas.
			float sumWeights = 0.0f;
			for (int c = 0; c < NUMGMMCOMPONENTS; c++)
			{
				//if the GMM component is the least probable component, 
				//its mean = vVal; sd = a high sd; weight = a low weight.
				if (c == leastProbGMMcomponentIndex)
				{
					GMM_afterVoxel.mu[c] = vVal;
					GMM_afterVoxel.sigma[c] = INITIALSD;
					GMM_afterVoxel.compProp[c] = INITIALWEIGHT;
				}
				//if the GMM component is not the least probable component,
				//its mean, sd and weight remain unchanged.
				else
				{
					GMM_afterVoxel.mu[c] = GMM_beforeVoxel.mu[c];
					GMM_afterVoxel.sigma[c] = GMM_beforeVoxel.sigma[c];
					GMM_afterVoxel.compProp[c] = GMM_beforeVoxel.compProp[c];
				}

				sumWeights += GMM_afterVoxel.compProp[c];
			}

			//printf("SumWeights:%lf", sumWeights);
			//(6.3.3)normalize the GMM_afterVoxel's weights.
			for (int c = 0; c < NUMGMMCOMPONENTS; c++)
			{
				GMM_afterVoxel.compProp[c] /= sumWeights;
			}

		}//end branch 2.



		//add on 2020/9/21.
		//7. compute the numNewDataPoints that sarisfies either clause (1) or (2), according to the paper, 
		//so as to obtain this block's foreground probability. 
		//(7.1)branch 1 (matchForProb1 == false): if the voxel value does not match any GMM_lastTimeStep components 
		//(may != clause 2 in the paper and thus may need recode).
		//条件1：不匹配任何权重大于阈值T的高斯分量
		/*bool matchForProb1 = false;
		int matchValsForProb[NUMGMMCOMPONENTS] = { -1, -1, -1 };
		for (int c = 0; c < NUMGMMCOMPONENTS; c++)
		{
			float lowBound = GMM_lastTimeStep.mu[c] - SDPARAMETER * GMM_lastTimeStep.sigma[c];
			float highBound = GMM_lastTimeStep.mu[c] + SDPARAMETER * GMM_lastTimeStep.sigma[c];

			if ((vVal >= lowBound) && (vVal <= highBound))
			{
				matchValsForProb[c] = abs(vVal - GMM_lastTimeStep.mu[c]);

				matchForProb1 = true;
			}
		}*/


		//(7.2)branch 2 (matchForProb2 == false): if the voxel value does not match any GMM_lastTimeStep components 
		//with corresponding weight > THRESHOLD_WEIGHT (= clause 1 in the paper).
		//不匹配任何权重大于阈值T的现有高斯函数
		/*bool matchForProb2 = false;
		for (int c = 0; c < NUMGMMCOMPONENTS; c++)
		{
			//(i)if the GMM_beforeVoxel.compProp[c] > threshold, 
			//we judge the matchToComputeProb.
			if (GMM_lastTimeStep.compProp[c] > THRESHOLD_WEIGHT)
			{
				if (matchValsForProb[c] != -1)
				{
					matchForProb2 = true;
					//break;
				}
			}

			//(ii)if the GMM_beforeVoxel.compProp[c] <= threshold, 
			//we do not care/judge the mathToComputeProb.
		}*/


		//if ((matchForProb1 == false) || (matchForProb2 == false) || (matchForProb3 == true))
		if ((matchForProb1 == false) || (matchForProb3 == true))
		{
			numNewDataPoints++;
		}
		//add on 2020/9/21.



		//8. so far, we obtain the GMM_afterVoxel parameters, 
		//which is used for next voxel loop.
		for (int c = 0; c < NUMGMMCOMPONENTS; c++)
		{
			GMM_beforeVoxel.mu[c] = GMM_afterVoxel.mu[c];
			GMM_beforeVoxel.sigma[c] = GMM_afterVoxel.sigma[c];
			GMM_beforeVoxel.compProp[c] = GMM_afterVoxel.compProp[c];
		}
		//here comes the incremental update scheme.

	}//end voxel loop.


	//9. for this block b(i, j, k): 
	//(9.1)store its GMM_afterVoxel into dev_blockGMMxxs_currentTimeStep(i, j, k).
	//疑点
	for (int c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		dev_blockGMMmus_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c * numBlocks.x * numBlocks.y * numBlocks.z] = GMM_afterVoxel.mu[c];
		dev_blockGMMsigmas_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c * numBlocks.x * numBlocks.y * numBlocks.z] = GMM_afterVoxel.sigma[c];
		dev_blockGMMcompProps_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c * numBlocks.x * numBlocks.y * numBlocks.z] = GMM_afterVoxel.compProp[c];
	}


	//add on 2020/9/21.
	//(9.2)store its foreground probability to dev_blockProbFG_currentTimeStep(i, j, k)
	//(Note: this blockProbFG value should be [0, 1]; otherwise error happens).
	float blockProbFG = numNewDataPoints / numTotalDataPoints;
	if ((blockProbFG < 0.0f) || (blockProbFG > 1.0f))
	{
		printf("blockProbFG: %f is not in [0, 1].\n", blockProbFG);
	}
	dev_blockProbFG_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y] = blockProbFG;
	//printf("blockProbFG： %f.\n", blockProbFG);
	//add on 2020/9/21.
}



__device__ float BhattacharyyaDistanceBetween2GMMComponents(float mu1, float sigma1, float mu2, float sigma2)
{
	//根据论文中的公式，返回所需值
	float result;
	sigma1 = powf(sigma1, 2);
	sigma2 = powf(sigma2, 2);
	result = (1.0f / 8.0f) * (mu1 - mu2) * powf((sigma1 + sigma2) / 2.0f, -1.0f) * (mu1 - mu2) + (1.0f / 2.0f) * logf(abs((sigma1 + sigma2) / 2.0f) / sqrtf(abs(sigma1) * abs(sigma2)));

	/*
	//Test: print result value.
	if (result != 0.0f)
	printf("result: %f.\n", result);
	//Test: print result value.
	*/

	return result;
}



//laucn 48 * 16 * 12 threads.
__global__ void computePerBlockBhattacharyyaDistance_currentTimeStep(dim3 numBlocks, float *dev_tarFeatGMMmu, float *dev_tarFeatGMMsigma, float *dev_tarFeatGMMcompProp,
	float *dev_blockGMMmus_currentTimeStep, float *dev_blockGMMsigmas_currentTimeStep, float *dev_blockGMMcompProps_currentTimeStep,
	float *dev_blockProbSimilarity_currentTimeStep)
{
	//for each block b(i, j, k):
	int i = threadIdx.x + blockIdx.x * blockDim.x;	//x range: [0 - 47].
	int j = threadIdx.y + blockIdx.y * blockDim.y;	//y rage: [0 - 15].
	int k = threadIdx.z + blockIdx.z * blockDim.z;	//z range: [0 - 11].


	//1. measure the Bhattacharyya-based distance between 
	//the tarFeatGMM and this block's GMM.
	float totalDistance = 0.0f;
	for (int c1 = 0; c1 < NUMGMMCOMPONENTS; c1++)
	for (int c2 = 0; c2 < NUMGMMCOMPONENTS; c2++)
	{
		//疑点 c2
		totalDistance += dev_tarFeatGMMcompProp[c1] * dev_blockGMMcompProps_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c2 * numBlocks.x * numBlocks.y * numBlocks.z] *
			BhattacharyyaDistanceBetween2GMMComponents(dev_tarFeatGMMmu[c1], dev_tarFeatGMMsigma[c1],
			dev_blockGMMmus_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c2 * numBlocks.x * numBlocks.y * numBlocks.z],
			dev_blockGMMsigmas_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y + c2 * numBlocks.x * numBlocks.y * numBlocks.z]);
	}


	//2. assign totalDistance to this block's probability dev_blockProbSimilarity_currentTimeStep(i, j , k).
	dev_blockProbSimilarity_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y] = totalDistance;

	/*
	//Test: print totalDistance.
	if (totalDistance != 0.0f)
	{
	printf("blockProbSim: %f.\n", totalDistance);
	}
	//Test: print totalDistance.
	*/

}



//launch 48 * 16 * 12 threads.
__global__ void normalize(dim3 numBlocks, float minVal, float maxVal, float *dev_data)
{
	//for each block b(i, j, k):
	int i = threadIdx.x + blockIdx.x * blockDim.x;	//x range: [0 - 47].
	int j = threadIdx.y + blockIdx.y * blockDim.y;	//y rage: [0 - 15].
	int k = threadIdx.z + blockIdx.z * blockDim.z;	//z range: [0 - 11].


	dev_data[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y] = (dev_data[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y] - minVal) / (maxVal - minVal);
}



//launch 48 * 16 * 12 threads.
__global__ void computePerBlockProbSimilarity_currentTimeStep(dim3 numBlocks, float *dev_blockProbSimilarity_currentTimeStep)
{
	//获取基于相似度的特征概率估计
	//for each block b(i, j, k):
	int i = threadIdx.x + blockIdx.x * blockDim.x;	//x range: [0 - 47].
	int j = threadIdx.y + blockIdx.y * blockDim.y;	//y rage: [0 - 15].
	int k = threadIdx.z + blockIdx.z * blockDim.z;	//z range: [0 - 11].


	/*//1.normalize the dev_blockProbSimilarity_currentTimeStep
	float minVal, maxVal;
	findMinAndMaxVals(numBlocks, &minVal, &maxVal,dev_blockProbSimilarity_currentTimeStep);
	normalize<<<numBlocks,1>>>(numBlocks, minVal, maxVal, dev_blockProbSimilarity_currentTimeStep);*/

	//2. measure the similarity between tarFeatGMM and this block GMM by using Equation 8.
	dev_blockProbSimilarity_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y] =
		1.0f - dev_blockProbSimilarity_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y];
}



//launch 48 * 16 * 12 threads.
__global__ void obtainPerBlockProbTarFeatAndClassificationField_currentTimeStep(int xDim, int yDim, int zDim, dim3 numBlocks, dim3 blockSize,
	float *dev_blockProbFG_currentTimeStep, float *dev_blockProbSimilarity_currentTimeStep,
	float *dev_blockProbTarFeat_currentTimeStep,
	float *dev_classificationField_currentTimeStep)
{
	//计算这个分块最终分类域的概率
	//for each block b(i, j, k):
	int i = threadIdx.x + blockIdx.x * blockDim.x;	//x range: [0 - 47].
	int j = threadIdx.y + blockIdx.y * blockDim.y;	//y rage: [0 - 15].
	int k = threadIdx.z + blockIdx.z * blockDim.z;	//z range: [0 - 11].


	//1. for this block, compute its dev_blockProbTarFeat_currentTimeStep(i, j, k).
	dev_blockProbTarFeat_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y] =
		(GAMMA)* dev_blockProbSimilarity_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y] +
		(1.0f - GAMMA)* dev_blockProbFG_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y];


	//2. in this block, construct its classification field.
	//为每个体素赋予分类域的值
	//loop through each voxel in this block.
	for (int kk = 0; kk < blockSize.z; kk++)
	for (int jj = 0; jj < blockSize.y; jj++)
	for (int ii = 0; ii < blockSize.x; ii++)
	{
		//for each voxel v(ii, jj, kk) (in this block):
		//(2.1)get this voxel's global id.
		dim3 vid;
		vid.x = ii + i * blockSize.x;	//[0 - 191].
		vid.y = jj + j * blockSize.y;	//[0 - 63].
		vid.z = kk + k * blockSize.z;	//[0 - 47].


		//(2.2)its value dev_classificationField_currentTimeStep(vid.x, vid.y, vid.z) = dev_blockProbTarFeat_currentTimeStep(i, j, k).
		dev_classificationField_currentTimeStep[vid.x + vid.y * xDim + vid.z * xDim * yDim] =
			dev_blockProbTarFeat_currentTimeStep[i + j * numBlocks.x + k * numBlocks.x * numBlocks.y];

	}


}



void thresholdClassificationField_currentTimeStep(int xDim, int yDim, int zDim, float threshold, float *classificationField_currentTimeStep)
{
	for (int kk = 0; kk < zDim; kk++)
	for (int jj = 0; jj < yDim; jj++)
	for (int ii = 0; ii < xDim; ii++)
	{
		//for each voxel v(ii, jj, kk):
		//if its value < threshold, then set its value = 0.0f.
		if (classificationField_currentTimeStep[ii + jj * xDim + kk * xDim * yDim] < threshold)
		{
			classificationField_currentTimeStep[ii + jj * xDim + kk * xDim * yDim] = 0.0f;
		}
	}

}



void floatToIntConverter(int xDim, int yDim, int zDim, float *classificationField_currentTimeStep, int *classificationFieldInt_currentTimeStep)
{
	for (int k = 0; k < zDim; k++)
	for (int j = 0; j < yDim; j++)
	for (int i = 0; i < xDim; i++)
	{
		//for each voxel v(i, j, k):
		//generate binary classificationFieldInt_currentTimeStep(i, j, k), according to classificationField_currentTimeStep(i, j, k).
		if (classificationField_currentTimeStep[i + j * xDim + k *xDim * yDim] > 0.0f)
		{
			classificationFieldInt_currentTimeStep[i + j * xDim + k *xDim * yDim] = 1;
		}
	}

}

void expand(float *FG_currentTimeStep, float *blockProb_currentTimeStep,int xDim,int yDim,int zDim,dim3 numBlocks)
{
	dim3 blockSize(4, 4, 4);
	//将blockProb_currentTimeStep(48,16,12)扩展至FG_currentTimeStep(192,64,48)
	for (int i = 0; i < xDim; i++)
	{
		int ii = i / blockSize.x; //[0-48]
		for (int j = 0; j < yDim; j++)
		{
			int jj = j / blockSize.y; //[0-16]
			for (int k = 0; k < zDim; k++)
			{
				int kk = k / blockSize.z; //[0-12]
				FG_currentTimeStep[i + j*xDim + k*xDim*yDim] = blockProb_currentTimeStep[ii + jj*numBlocks.x + kk*numBlocks.x*numBlocks.y];
			}
		}
	}
}



int main()
{
	//for computing program runtime.
	clock_t start_t, end_t;
	start_t = clock();


	//declare variables.
	char dataSource[100] = "D:/Graduate_Design/Data/3DFlow/";
	char saveDataPath[150] = "D:/FeatureAware-Classification/build_GMM(Teacher-Version)/build_GMM(Teacher-Version)/NewSquareCylinder/";//特征所在的文件夹
	char fileName[200];
	//int fid = 0;

	dim3 blockSize = { 4, 4, 4 };
	float selectRegionxmin = 75, selectRegionymin = 18, selectRegionzmin = 3;
	float selectRegionxmax = 87, selectRegionymax = 23, selectRegionzmax = 43;
	//printf("blockSize.x: %d, blockSize.y: %d, blockSize.z: %d.\n", blockSize.x, blockSize.y, blockSize.z);


	//0. compute target feature center at initial time step, and
	//assign tarFeatCenter_initialTimeStep to tarFeatCenter_lastTimeStep.
	float3 tarFeatCenter_initialTimeStep;
	tarFeatCenter_initialTimeStep.x = (selectRegionxmin + selectRegionxmax) / 2.0f;
	tarFeatCenter_initialTimeStep.y = (selectRegionymin + selectRegionymax) / 2.0f;
	tarFeatCenter_initialTimeStep.z = (selectRegionzmin + selectRegionzmax) / 2.0f;
	float3 tarFeatCenter_lastTimeStep = tarFeatCenter_initialTimeStep;
	printf("tarFeatCenter_lastTimeStep: %f, %f, %f.\n", tarFeatCenter_lastTimeStep.x, tarFeatCenter_lastTimeStep.y, tarFeatCenter_lastTimeStep.z);



	//1. parse .am header at tarFeat time step (where the target feature is selected) to obtain necessary data information.
	int xDim = 0, yDim = 0, zDim = 0;
	int numVecComponents = 0;
	float xmin = 1.0f, ymin = 1.0f, zmin = 1.0f;
	float xmax = -1.0f, ymax = -1.0f, zmax = -1.0f;
	bool isUniform = false;
	sprintf(fileName, "%sflow_t%.4d.am", dataSource, FILESTARTVAL + TARFEATTIMESTEP * FILEINCREMENT);
	parseAmiraMeshHeader(fileName, &xDim, &yDim, &zDim, &numVecComponents, &xmin, &xmax, &ymin, &ymax, &zmin, &zmax, &isUniform);
	printf("Grid Dimensions: %d %d %d.\n", xDim, yDim, zDim);
	printf("BoundingBox in x-Direction: [%g ... %g]\n", xmin, xmax);
	printf("BoundingBox in y-Direction: [%g ... %g]\n", ymin, ymax);
	printf("BoundingBox in z-Direction: [%g ... %g]\n", zmin, zmax);
	printf("Uniform Grid: %s\n", isUniform ? "true" : "false");
	printf("Number of Vector Components: %d.\n", numVecComponents);



	//2. read tarFeatGMMmu.raw, tarFeatGMMsigma.raw and tarFeatGMMcompProp.raw at tarFeat time step,
	//and copy them to dev_tarFeatGMMmu, dev_tarFeatGMMsigma, dev_tarFeatGMMcompProp.
	//用上个matlab文件获得的目标特征GMM
	float *tarFeatGMMmu, *dev_tarFeatGMMmu;
	float *tarFeatGMMsigma, *dev_tarFeatGMMsigma;
	float *tarFeatGMMcompProp, *dev_tarFeatGMMcompProp;
	sprintf(fileName, "%sflow_t%.4d_tarFeatGMMmu.raw", saveDataPath, FILESTARTVAL + TARFEATTIMESTEP * FILEINCREMENT);
	tarFeatGMMmu = (float *)readVolumeData(fileName, sizeof(float), NUMGMMCOMPONENTS);
	//cout << fileName << endl;
	sprintf(fileName, "%sflow_t%.4d_tarFeatGMMsigma.raw", saveDataPath, FILESTARTVAL + TARFEATTIMESTEP * FILEINCREMENT);
	tarFeatGMMsigma = (float *)readVolumeData(fileName, sizeof(float), NUMGMMCOMPONENTS);
	sprintf(fileName, "%sflow_t%.4d_tarFeatGMMcompProp.raw", saveDataPath, FILESTARTVAL + TARFEATTIMESTEP * FILEINCREMENT);
	tarFeatGMMcompProp = (float *)readVolumeData(fileName, sizeof(float), NUMGMMCOMPONENTS);

	//copy tarFeatGMMxx to dev_tarFeatGMMxx.
	cudaMalloc((void **)&dev_tarFeatGMMmu, sizeof(float)* NUMGMMCOMPONENTS);
	cudaMemcpy(dev_tarFeatGMMmu, tarFeatGMMmu,
		sizeof(float)* NUMGMMCOMPONENTS, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_tarFeatGMMsigma, sizeof(float)* NUMGMMCOMPONENTS);
	cudaMemcpy(dev_tarFeatGMMsigma, tarFeatGMMsigma,
		sizeof(float)* NUMGMMCOMPONENTS, cudaMemcpyHostToDevice);
	cudaMalloc((void **)&dev_tarFeatGMMcompProp, sizeof(float)* NUMGMMCOMPONENTS);
	cudaMemcpy(dev_tarFeatGMMcompProp, tarFeatGMMcompProp,
		sizeof(float)* NUMGMMCOMPONENTS, cudaMemcpyHostToDevice);

	//Test: print tarFeatGMM.
	for (int c = 0; c < NUMGMMCOMPONENTS; c++)
	{
		printf("%f, %f, %f.\n", tarFeatGMMmu[c], tarFeatGMMsigma[c], tarFeatGMMcompProp[c]);
	}
	//Test: print tarFeatGMM.

	//free tarFeatGMMmu, tarFeatGMMsigma and tarFeatGMMcompProp, after they are copied to 
	//dev_tarFeatGMMmu, dev_tarFeatGMMsigma, dev_tarFeatGMMcompProp. 
	free(tarFeatGMMmu);
	free(tarFeatGMMsigma);
	free(tarFeatGMMcompProp);


	//3. read BlockGMMmus.raw, BlockGMMsigmas.raw, BlockGMMcompProps.raw at tarFeat time step.
	dim3 numBlocks = { xDim / blockSize.x, yDim / blockSize.y, zDim / blockSize.z };
	printf("numBlocks.x: %d, numBlocks.y: %d, numBlocks.z: %d.\n", numBlocks.x, numBlocks.y, numBlocks.z);

	float *blockGMMmus_tarFeatTimeStep;
	sprintf(fileName, "%sflow_t%.4d_BlockGMMmus.raw", saveDataPath, FILESTARTVAL + TARFEATTIMESTEP * FILEINCREMENT);
	blockGMMmus_tarFeatTimeStep = (float *)readVolumeData(fileName,
		sizeof(float), numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);


	float *blockGMMsigmas_tarFeatTimeStep;
	sprintf(fileName, "%sflow_t%.4d_BlockGMMsigmas.raw", saveDataPath, FILESTARTVAL + TARFEATTIMESTEP * FILEINCREMENT);
	blockGMMsigmas_tarFeatTimeStep = (float *)readVolumeData(fileName,
		sizeof(float), numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);


	float *blockGMMcompProps_tarFeatTimeStep;
	sprintf(fileName, "%sflow_t%.4d_BlockGMMcompProps.raw", saveDataPath, FILESTARTVAL + TARFEATTIMESTEP * FILEINCREMENT);
	blockGMMcompProps_tarFeatTimeStep = (float *)readVolumeData(fileName,
		sizeof(float), numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);



	//4. declare dev_blockGMMmus_lastTimeStep, dev_blockGMMsigmas_lastTimeStep, dev_blockGMMcompProps_lastTimeStep at last time step and
	//initialize them the same as in tarFeat time step:
	//dev_blockGMMmus_lastTimeStep = blockGMMmus_tarFeatTimeStep;
	//dev_blockGMMsigmas_lastTimeStep = blockGMMsigmas_tarFeatTimeStep;
	//dev_blockGMMcompProps_lastTimeStep = blockGMMcompProps_tarFeatTimeStep.
	float *dev_blockGMMmus_lastTimeStep;
	cudaMalloc((void **)&dev_blockGMMmus_lastTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
	cudaMemcpy(dev_blockGMMmus_lastTimeStep, blockGMMmus_tarFeatTimeStep,
		sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyHostToDevice);


	float *dev_blockGMMsigmas_lastTimeStep;
	cudaMalloc((void **)&dev_blockGMMsigmas_lastTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
	cudaMemcpy(dev_blockGMMsigmas_lastTimeStep, blockGMMsigmas_tarFeatTimeStep,
		sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyHostToDevice);


	float *dev_blockGMMcompProps_lastTimeStep;
	cudaMalloc((void **)&dev_blockGMMcompProps_lastTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
	cudaMemcpy(dev_blockGMMcompProps_lastTimeStep, blockGMMcompProps_tarFeatTimeStep,
		sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyHostToDevice);



	//free blockGMMxxs_tarFeatTimeStep at tarFeat time step, after they are copied to dev_blockGMMxxs_lastTimeStep.
	free(blockGMMmus_tarFeatTimeStep);
	free(blockGMMsigmas_tarFeatTimeStep);
	free(blockGMMcompProps_tarFeatTimeStep);



	//5. declare blockGMMxxs_currentTimeStep/dev_blockGMMxxs_currentTimeStep at current time step.
	float *blockGMMmus_currentTimeStep, *dev_blockGMMmus_currentTimeStep;
	blockGMMmus_currentTimeStep = (float *)malloc(sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
	cudaMalloc((void **)&dev_blockGMMmus_currentTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);


	float *blockGMMsigmas_currentTimeStep, *dev_blockGMMsigmas_currentTimeStep;
	blockGMMsigmas_currentTimeStep = (float *)malloc(sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
	cudaMalloc((void **)&dev_blockGMMsigmas_currentTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);


	float *blockGMMcompProps_currentTimeStep, *dev_blockGMMcompProps_currentTimeStep;
	blockGMMcompProps_currentTimeStep = (float *)malloc(sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
	cudaMalloc((void **)&dev_blockGMMcompProps_currentTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);



	//6. declare blockProbFG_currentTimeStep/dev_blockProbFG_currentTimeStep at current time step.
	float *blockProbFG_currentTimeStep, *dev_blockProbFG_currentTimeStep;
	blockProbFG_currentTimeStep = (float *)malloc(sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);
	cudaMalloc((void **)&dev_blockProbFG_currentTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);



	//7. declare blockProbSimilarity_currentTimeStep/dev_blockProbSimilarity_currentTimeStep at current time step.
	float *blockProbSimilarity_currentTimeStep, *dev_blockProbSimilarity_currentTimeStep;
	blockProbSimilarity_currentTimeStep = (float *)malloc(sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);
	cudaMalloc((void **)&dev_blockProbSimilarity_currentTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);



	//8. declare blockProbTarFeat_currentTimeStep/dev_blockProbTarFeat_currentTimeStep at current time step.
	float *blockProbTarFeat_currentTimeStep, *dev_blockProbTarFeat_currentTimeStep;
	blockProbTarFeat_currentTimeStep = (float *)malloc(sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);
	cudaMalloc((void **)&dev_blockProbTarFeat_currentTimeStep, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);



	//9. declare dev_classificationField_currentTimeStep at current time step.
	float *classificationField_currentTimeStep, *dev_classificationField_currentTimeStep;
	classificationField_currentTimeStep = (float *)malloc(sizeof(float)* xDim * yDim * zDim);
	cudaMalloc((void **)&dev_classificationField_currentTimeStep, sizeof(float)* xDim * yDim * zDim);



	//loop through each .am file.
	//for (int fid = 1; fid < NUMFILES; fid++)
	for (int fid = TARFEATTIMESTEP + 1; fid < NUMFILES; fid++) //从特征时间步开始的必要性：因为matlab用EM算法完成的GMM参数保存在特征时间步中，所以必须从这一时间步开始
	{
		//start a new loop.
		printf("\nLoop through flow_t%.4d.am:\n", FILESTARTVAL + fid * FILEINCREMENT);


		//10. clear memory to be 0.
		//(10.1)clear blockGMMxxs_currentTimeStep/dev_blockGMMxxs_currentTimeStep at current time step.
		memset(blockGMMmus_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		cudaMemset(dev_blockGMMmus_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		memset(blockGMMsigmas_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		cudaMemset(dev_blockGMMsigmas_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		memset(blockGMMcompProps_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		cudaMemset(dev_blockGMMcompProps_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);


		//(10.2)clear blockProbFG_currentTimeStep/dev_blockProbFG_currentTimeStep at current time step.
		memset(blockProbFG_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);
		cudaMemset(dev_blockProbFG_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);


		//(10.3)clear blockProbSimilarity_currentTimeStep/dev_blockProbSimilarity_currentTimeStep at current time step.
		memset(blockProbSimilarity_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);
		cudaMemset(dev_blockProbSimilarity_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);


		//(10.4)clear blockProbTarFeat_currentTimeStep/dev_blockProbTarFeat_currentTimeStep at current time step.
		memset(blockProbTarFeat_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);
		cudaMemset(dev_blockProbTarFeat_currentTimeStep, 0, sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z);


		//(10.5)clear classificationField_currentTimeStep/dev_classificationFiled_currentTimeStep at current time step.
		memset(classificationField_currentTimeStep, 0, sizeof(float)* xDim * yDim * zDim);
		cudaMemset(dev_classificationField_currentTimeStep, 0, sizeof(float)* xDim * yDim * zDim);


		//11. read the current .am data at current time step (starting at time step TARFEATTIMESTEP + 1).
		sprintf(fileName, "%sflow_t%.4d.am", dataSource, FILESTARTVAL + fid * FILEINCREMENT);
		float *amData_currentTimeStep;
		amData_currentTimeStep = readAmiraMeshData(fileName, xDim, yDim, zDim, numVecComponents);


		/*
		//Test: Print all data values.
		//Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
		printf("Printing all values in the same order in which they are in memory:\n");
		for (int k = 0; k < zDim; k++)
		for (int j = 0; j < yDim; j++)
		for (int i = 0; i < xDim; i++)
		{
		//Note: Random access to the value (of the first component) of the grid point (i,j,k):
		// pData[((k * yDim + j) * xDim + i) * NumComponents]
		//assert(pData[((k * yDim + j) * xDim + i) * NumComponents] == pData[Idx * NumComponents]);

		for (int c = 0; c < numVecComponents; c++)
		{
		printf("%f, ", amData_timestepx[(i + j * xDim + k * xDim * yDim) * numVecComponents + c]);
		}
		printf("\n");
		}
		//Test: Print all data values.
		*/



		//12. given amData_currentTimeStep (including 3 velocities) at current time step, 
		//compute its velocity magnitude at this time step.
		float *velMagVol_currentTimeStep, *dev_velMagVol_currentTimeStep;
		velMagVol_currentTimeStep = (float *)malloc(sizeof(float)* xDim * yDim * zDim);
		cudaMalloc((void **)&dev_velMagVol_currentTimeStep, sizeof(float)* xDim * yDim * zDim);
		for (int k = 0; k < zDim; k++)
		for (int j = 0; j < yDim; j++)
		for (int i = 0; i < xDim; i++)
		{
			//for each voxel v(i, j, k);
			float velMag_ijk = 0;
			for (int c = 0; c < numVecComponents; c++)
			{
				velMag_ijk += pow(amData_currentTimeStep[(i + j * xDim + k * xDim * yDim) * numVecComponents + c], 2);
			}

			velMag_ijk = sqrt(velMag_ijk);

			//assign velMag_ijk to velMagVol(i, j, k).
			velMagVol_currentTimeStep[i + j * xDim + k * xDim * yDim] = velMag_ijk;
		}
		printf("compute velMagVol_currentTimeStep successfully.\n");
		//copy velMagVol_currentTimeStep to dev_velMagVol_currentTimeStep.
		cudaMemcpy(dev_velMagVol_currentTimeStep, velMagVol_currentTimeStep,
			sizeof(float)* xDim * yDim * zDim, cudaMemcpyHostToDevice);


		//free amData_currentTimeStep, after velMagVol_currentTimeStep is computed. 
		free(amData_currentTimeStep);



		//13. given dev_velMagVol_currentTimeStep at current time step, (in parallel) 
		//(13.1)update the perBlockGMM at last time step to obtain the perBlockGMM at current time step;
		//(13.2)obtain the perBlockProbFG (range: [0, 1]) at current time step.
		computePerBlockGMMAndProbForeground_currentTimeStep << <numBlocks, 1 >> >(xDim, yDim, zDim, numBlocks, blockSize,
			dev_blockGMMmus_lastTimeStep, dev_blockGMMsigmas_lastTimeStep, dev_blockGMMcompProps_lastTimeStep,
			dev_velMagVol_currentTimeStep,
			dev_blockGMMmus_currentTimeStep, dev_blockGMMsigmas_currentTimeStep, dev_blockGMMcompProps_currentTimeStep,
			dev_blockProbFG_currentTimeStep);
		cudaError_t cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("computePerBlockGMMAndProbForeground_currentTimeStep kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
		}

		/* Test: update the parameters
		check the difference from 0->n and n->0
		*/
		//printf("%lf", dev_blockGMMmus_currentTimeStep[0]);
		printFirstBlockParams<<<1,1>>>(dev_blockGMMmus_currentTimeStep, dev_blockGMMsigmas_currentTimeStep, dev_blockGMMcompProps_currentTimeStep,10,11,10,numBlocks);
		//if (fid > 25) exit(0);

		//2020/10/8: test.
		//save dev_blockProbFG_currentTimeStep at current time step.
		cudaMemcpy(blockProbFG_currentTimeStep, dev_blockProbFG_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z, cudaMemcpyDeviceToHost);
		sprintf(fileName, "%sflow_t%.4d_BlockProbFG.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		//声明一个变量FG_currentTimeStep
		float *FG_currentTimeStep = (float*)malloc(sizeof(float)*xDim*yDim*zDim);
		expand(FG_currentTimeStep, blockProbFG_currentTimeStep,xDim,yDim,zDim,numBlocks);
		saveToRawFile(fileName, FG_currentTimeStep, 'f', xDim*yDim*zDim);
		free(FG_currentTimeStep);
		//2020/10/8: test.



		//14. given (i)dev_tarFeatGMMxx and (ii)dev_blockGMMxxs_currentTimeStep at current time step, (in parallel)
		//compute dev_blockProbSimilarity_currentTimeStep at current time step.
		//(14.1)(in parallel)compute perBlock Bahttacharyya-based distance (using Equation 6),
		//which is stored in dev_blockProbSimilarity_currentTimeStep.
		computePerBlockBhattacharyyaDistance_currentTimeStep << <numBlocks, 1 >> >(numBlocks, dev_tarFeatGMMmu, dev_tarFeatGMMsigma, dev_tarFeatGMMcompProp,
			dev_blockGMMmus_currentTimeStep, dev_blockGMMsigmas_currentTimeStep, dev_blockGMMcompProps_currentTimeStep,
			dev_blockProbSimilarity_currentTimeStep);
		cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("computePerBlockBhattacharyyaDistance_currentTimeStep kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
		}



		//copy dev_blockProbSimilairy_currentTimeStep to blockProbSimilarity_currentTimeStep.
		cudaMemcpy(blockProbSimilarity_currentTimeStep, dev_blockProbSimilarity_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z, cudaMemcpyDeviceToHost);


		//(14.2)find blockProbSimilarity_currentTimeStep's min. and max. values.
		float minVal_blockProbSimilarity = 1000.0f;
		float maxVal_blockProbSimilarity = -1;
		findMinAndMaxVals(numBlocks, &minVal_blockProbSimilarity, &maxVal_blockProbSimilarity, blockProbSimilarity_currentTimeStep);
		printf("minVal_blockProbSimilarity: %f, maxVal_blockProbSimilarity: %f.\n", minVal_blockProbSimilarity, maxVal_blockProbSimilarity);


		//(14.3)given blockProbSimilarity_currentTimeStep's min. and max. values, (in parallel)
		//normalize dev_blockProbSimilarity_currentTimeStep (range: [0, 1]).
		normalize << <numBlocks, 1 >> >(numBlocks, minVal_blockProbSimilarity, maxVal_blockProbSimilarity, dev_blockProbSimilarity_currentTimeStep);
		cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("normalize kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
		}


		//copy dev_blockProbSimilarity_currentTimeStep to blockProbSimilarity_currentTimeStep.
		cudaMemcpy(blockProbSimilarity_currentTimeStep, dev_blockProbSimilarity_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z, cudaMemcpyDeviceToHost);


		//(14.4)after normalization, find blockProbSimilarity_currentTimeStep's min. and max. values.
		float minVal_normBlockProbSimilarity = 1000.0f;
		float maxVal_normBlockProbSimilarity = -1.0f;
		findMinAndMaxVals(numBlocks, &minVal_normBlockProbSimilarity, &maxVal_normBlockProbSimilarity, blockProbSimilarity_currentTimeStep);
		printf("minVal_blockProbSimilarity: %f, maxVal_blockProbSimilarity: %f.\n", minVal_normBlockProbSimilarity, maxVal_normBlockProbSimilarity);



		//(14.5) (in parallel)compute perBlock similarity probability 
		//dev_blockProbSimilarity_currentTimeTime at current time step using Equation 8.
		computePerBlockProbSimilarity_currentTimeStep << <numBlocks, 1 >> >(numBlocks, dev_blockProbSimilarity_currentTimeStep);
		cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("computePerBlockProbSimilarity_currentTimeStep kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
		}
		


		//2020/10/8: test.
		//save dev_blockProbSimilarity_currentTimeStep at current time step.
		cudaMemcpy(blockProbSimilarity_currentTimeStep, dev_blockProbSimilarity_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z, cudaMemcpyDeviceToHost);
		sprintf(fileName, "%sflow_t%.4d_BlockProbSimilarity.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		//扩展blockProbSimilarity_currentTimeStep至(192，64，48)
		float *Similarity_currentTimeStep = (float*)malloc(sizeof(float)*xDim*yDim*zDim);
		expand(Similarity_currentTimeStep, blockProbSimilarity_currentTimeStep, xDim, yDim, zDim, numBlocks);
		saveToRawFile(fileName, Similarity_currentTimeStep, 'f', xDim*yDim*zDim);
		free(Similarity_currentTimeStep);
		//2020/10/8: test.



		//(14.6)find blockProbSimilarity_currentTimeStep's min. and max. values.
		cudaMemcpy(blockProbSimilarity_currentTimeStep, dev_blockProbSimilarity_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z, cudaMemcpyDeviceToHost);
		minVal_blockProbSimilarity = 1000.0f;
		maxVal_blockProbSimilarity = -1.0f;
		findMinAndMaxVals(numBlocks, &minVal_blockProbSimilarity, &maxVal_blockProbSimilarity, blockProbSimilarity_currentTimeStep);
		printf("minVal_blockProbSimilarity: %f, maxVal_blockProbSimilarity: %f.\n", minVal_blockProbSimilarity, maxVal_blockProbSimilarity);



		//15. now, given (i)dev_blockProbFG_currentTimeStep and (ii)dev_blockProbSimilarity_currentTimeStep,
		//(15.1)(in parallel)compute dev_blockProbTarFeat_currentTimeStep, and
		//(15.2)construct the feature-aware classification field dev_classificationFiled_currentTimeStep.
		obtainPerBlockProbTarFeatAndClassificationField_currentTimeStep << <numBlocks, 1 >> >(xDim, yDim, zDim, numBlocks, blockSize,
			dev_blockProbFG_currentTimeStep, dev_blockProbSimilarity_currentTimeStep,
			dev_blockProbTarFeat_currentTimeStep,
			dev_classificationField_currentTimeStep);
		cudaerr = cudaDeviceSynchronize();
		if (cudaerr != cudaSuccess)
		{
			printf("obtainPerBlockProbTarFeatAndClassificationField_currentTimeStep kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
		}


		//(15.3)find blockProbTarFeat_currentTimeStep's min. and max. values.
		cudaMemcpy(blockProbTarFeat_currentTimeStep, dev_blockProbTarFeat_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z, cudaMemcpyDeviceToHost);
		float minVal_blockProbTarFeat = 1000.0f;
		float maxVal_blockProbTarFeat = -1.0f;
		findMinAndMaxVals(numBlocks, &minVal_blockProbTarFeat, &maxVal_blockProbTarFeat, blockProbTarFeat_currentTimeStep);
		printf("minVal_blockProbTarFeat: %f, maxVal_blockProbTarFeat: %f.\n", minVal_blockProbTarFeat, maxVal_blockProbTarFeat);



		//16(optional). for current time step, save as .raw files.
		/*
		//(16.1)save dev_blockGMMxxs_currentTimeStep at current time step.
		//copy dev_blockGMMxxs_currentTimeStep to blockGMMxxs_currentTimeStep.
		cudaMemcpy(blockGMMmus_currentTimeStep, dev_blockGMMmus_currentTimeStep,
		sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyDeviceToHost);
		cudaMemcpy(blockGMMsigmas_currentTimeStep, dev_blockGMMsigmas_currentTimeStep,
		sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyDeviceToHost);
		cudaMemcpy(blockGMMcompProps_currentTimeStep, dev_blockGMMcompProps_currentTimeStep,
		sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyDeviceToHost);
		//save blockGMMxxs_currentTimeStep at current time.
		sprintf(fileName, "%sflow_t%.4d_BlockGMMmus.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		saveToRawFile(fileName, blockGMMmus_currentTimeStep, 'f', numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		sprintf(fileName, "%sflow_t%.4d_BlockGMMsigmas.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		saveToRawFile(fileName, blockGMMsigmas_currentTimeStep, 'f', numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		sprintf(fileName, "%sflow_t%.4d_BlockGMMcompProps.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		saveToRawFile(fileName, blockGMMcompProps_currentTimeStep, 'f', numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS);
		*/


		//(16.2)save velMagVol_currentTimeStep at current time step.
		sprintf(fileName, "%sflow_t%.4d_velMag.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		saveToRawFile(fileName, velMagVol_currentTimeStep, 'f', xDim * yDim * zDim);
		free(velMagVol_currentTimeStep);
		cudaFree(dev_velMagVol_currentTimeStep);


		/*
		//(16.3)save dev_probFG_currentTimeStep at current time step.
		//copy dev_probFG_currentTimeStep to probFG_currentTimeStep.
		cudaMemcpy(blockProbFG_currentTimeStep, dev_blockProbFG_currentTimeStep,
		sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z, cudaMemcpyDeviceToHost);
		//save probFG_currentTimeStep at current time step.
		sprintf(fileName, "%sflow_t%.4d_BlockProbFG.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		saveToRawFile(fileName, blockProbFG_currentTimeStep, 'f', numBlocks.x * numBlocks.y * numBlocks.z);



		//(16.4)save dev_classificationFiled_currentTimeStep at current time step.
		//copy dev_classificationField_currentTimeStep to classificationField_currentTimeStep.
		cudaMemcpy(classificationField_currentTimeStep, dev_classificationField_currentTimeStep,
		sizeof(float)* xDim * yDim * zDim, cudaMemcpyDeviceToHost);
		//save classificationField_currentTimeStep at current time step.
		sprintf(fileName, "%sflow_t%.4d_classificationFiled.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		saveToRawFile(fileName, classificationField_currentTimeStep, 'f', xDim * yDim * zDim);
		*/



		//--------tracking part--------//
		//17. so far, we obtain the classificationField_currentTimeStep/dev_classificationField_currentTimeStep at current time step.
		//thresholding the classification field at current time step.
		cudaMemcpy(classificationField_currentTimeStep, dev_classificationField_currentTimeStep,
			sizeof(float)* xDim * yDim *zDim, cudaMemcpyDeviceToHost);
		thresholdClassificationField_currentTimeStep(xDim, yDim, zDim, THRESHOLD_POSS, classificationField_currentTimeStep);



		//18. apply connected component analysis algorithm on the thresholded classification field at current time step.
		//according to classificationField_currentTimeStep(i, j, k), generate binary classificationFieldInt_currentTimeStep(i, j, k), 
		//so as to obtain the CClabelVol_currentTimeStep at current time step.
		int *classificationFieldInt_currentTimeStep = (int *)malloc(sizeof(int)* xDim * yDim * zDim);
		//clear classificationFieldInt_currentTimeStep.
		memset(classificationFieldInt_currentTimeStep, 0, sizeof(int)* xDim * yDim * zDim);
		floatToIntConverter(xDim, yDim, zDim, classificationField_currentTimeStep, classificationFieldInt_currentTimeStep);
		int *ccLabelVol_currentTimeStep = (int *)malloc(sizeof(int)* xDim * yDim * zDim);
		int numCCs = i4block_components(xDim, yDim, zDim, classificationFieldInt_currentTimeStep, ccLabelVol_currentTimeStep);
		free(classificationFieldInt_currentTimeStep);
		printf("numCCs: %d.\n", numCCs);



		//19(2020/10/7: 已检查). given numCCs and ccLabelVol_currentTimeStep at current time step, 
		//(19.1)compute the center of each connected component;
		//(19.2)compute the Euclidean distance between each connected component's center (at current time step) and the target feature center (at last time step).
		// 所有坐标值的平均值就是中心,也即重心.
		float3 *ccCenters = (float3 *)malloc(sizeof(float3)* numCCs);
		float *eucDistance = (float *)malloc(sizeof(float)* numCCs);
		for (int cc = 1; cc <= numCCs; cc++)
		{
			//for each connected component at current time step:
			//(i)find its ccxmin, ccxmax, ccymin, ccymax, cczmin and cczmax along 3 dimensions.
			/*float ccxmin = xDim, ccxmax = -1;
			float ccymin = yDim, ccymax = -1;
			float cczmin = zDim, cczmax = -1;*/
			int numOfCcVoxels = 0;//连接分量体素数量
			int sumOfX = 0;
			int sumOfY = 0;
			int sumOfZ = 0;

			//loop through each voxel v(i, j, k): 找出连接分量的中心
			for (int k = 0; k < zDim; k++)
			for (int j = 0; j < yDim; j++)
			for (int i = 0; i < xDim; i++)
			{
				if (ccLabelVol_currentTimeStep[i + j * xDim + k * xDim * yDim] == cc)
				{
					numOfCcVoxels += 1;
					sumOfX += i;
					sumOfY += j;
					sumOfZ += k;

					//for each voxel(i, j, k) belongs to the cc:
					/*if (i > ccxmax)
						ccxmax = i;
					else if (i < ccxmin)
						ccxmin = i;

					if (j > ccymax)
						ccymax = j;
					else if (j < ccymin)
						ccymin = j;

					if (k > cczmax)
						cczmax = k;
					else if (k < cczmin)
						cczmin = k;*/
				}

			}

			ccCenters[cc - 1].x = 1.0*sumOfX / numOfCcVoxels;
			ccCenters[cc - 1].y = 1.0*sumOfY / numOfCcVoxels;
			ccCenters[cc - 1].z = 1.0*sumOfZ / numOfCcVoxels;

			//(ii)for this connected component, given its ccxmin, ccxmax, ccymin, ccymax, cczmin and cczmax,
			//compute its center.
			/*ccCenters[cc - 1].x = (ccxmin + ccxmax) / 2.0f;
			ccCenters[cc - 1].y = (ccymin + ccymax) / 2.0f;
			ccCenters[cc - 1].z = (cczmin + cczmax) / 2.0f;*/


			//Test: for this connected component, print its center.
			//printf("center: %f, %f, %f.\n", ccCenters[cc-1].x, ccCenters[cc-1].y, ccCenters[cc-1].z);
			//Test: for this connected component, print its center.


			//(iii)for this connected component, compute the Euclidean distance between its center and the target feature center (at last time step).
			eucDistance[cc - 1] = sqrt(pow(ccCenters[cc - 1].x - tarFeatCenter_lastTimeStep.x, 2) +
				pow(ccCenters[cc - 1].y - tarFeatCenter_lastTimeStep.y, 2) +
				pow(ccCenters[cc - 1].z - tarFeatCenter_lastTimeStep.z, 2));
		}



		//20. find the connected component with the minimum distance to the target feature at last time step, which is considered as the 
		//target feature at current time step.
		//(20.1)find out the best match connected component (which is the target feature) at current time step.
		float minEucDistance = 10000.0f;
		int bestMatchCC_currentTimeStep = -1;
		for (int cc = 1; cc <= numCCs; cc++)
		{
			//for each connected component:
			if (eucDistance[cc - 1] < minEucDistance)
			{
				minEucDistance = eucDistance[cc - 1];
				bestMatchCC_currentTimeStep = cc;
			}
		}
		printf("best match connected component: %d; its euclidean distance: %f.\n", bestMatchCC_currentTimeStep,
			eucDistance[bestMatchCC_currentTimeStep - 1]);


		//(20.2)given the best match connected component bestMatchCC_currentTimeStep (which is the target feature) and 
		//ccLabelVol_currentTimeStep at current time step, 
		//clear the classificationField_currentTimeStep (which are not the target feature) at current time step.
		for (int k = 0; k < zDim; k++)
		for (int j = 0; j < yDim; j++)
		for (int i = 0; i < xDim; i++)
		{
			//for each voxel v(i, j, k):
			if (ccLabelVol_currentTimeStep[i + j * xDim + k * xDim * yDim] != bestMatchCC_currentTimeStep)
			{
				classificationField_currentTimeStep[i + j * xDim + k * xDim * yDim] = 0.0f;
			}
		}


		//(20.3)given the best match connected component (which is the target feature) at current time step, obtain its center.
		float3 tarFeatCenter_currentTimeStep = ccCenters[bestMatchCC_currentTimeStep - 1];


		//free memory.
		free(ccCenters);
		free(eucDistance);



		//21. save classificationFiled_currentTimeStep (range: [0, 1]), which contains the extracted target feature at current time step. 
		sprintf(fileName, "%sflow_t%.4d_classificationFiled.raw", saveDataPath, FILESTARTVAL + fid * FILEINCREMENT);
		saveToRawFile(fileName, classificationField_currentTimeStep, 'f', xDim * yDim * zDim);
		//--------tracking part--------//



		//22. for next loop,
		//(22.1)assign dev_blockGMMxxs_lastTimeStep = dev_blockGMMxxs_currentTimeStep.
		cudaMemcpy(dev_blockGMMmus_lastTimeStep, dev_blockGMMmus_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyDeviceToDevice);

		cudaMemcpy(dev_blockGMMsigmas_lastTimeStep, dev_blockGMMsigmas_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyDeviceToDevice);

		cudaMemcpy(dev_blockGMMcompProps_lastTimeStep, dev_blockGMMcompProps_currentTimeStep,
			sizeof(float)* numBlocks.x * numBlocks.y * numBlocks.z * NUMGMMCOMPONENTS, cudaMemcpyDeviceToDevice);

		//(22.2)assign tarFeatCenter_lastTimeStep = tarFeatCenter_currentTimeStep.
		tarFeatCenter_lastTimeStep = tarFeatCenter_currentTimeStep;

	}//end loop through each .am file.



	//free cpu + gpu memory.
	free(blockGMMmus_currentTimeStep);
	free(blockGMMsigmas_currentTimeStep);
	free(blockGMMcompProps_currentTimeStep);
	free(blockProbFG_currentTimeStep);
	free(blockProbSimilarity_currentTimeStep);
	free(blockProbTarFeat_currentTimeStep);
	free(classificationField_currentTimeStep);
	cudaFree(dev_blockGMMmus_lastTimeStep);
	cudaFree(dev_blockGMMsigmas_lastTimeStep);
	cudaFree(dev_blockGMMcompProps_lastTimeStep);
	cudaFree(dev_blockGMMmus_currentTimeStep);
	cudaFree(dev_blockGMMsigmas_currentTimeStep);
	cudaFree(dev_blockGMMcompProps_currentTimeStep);
	cudaFree(dev_blockProbFG_currentTimeStep);
	cudaFree(dev_blockProbSimilarity_currentTimeStep);
	cudaFree(dev_blockProbTarFeat_currentTimeStep);
	cudaFree(dev_classificationField_currentTimeStep);



	end_t = clock();
	printf("BlockGMMsIncrementalUpdate consumed time = %fs.\n\n", (end_t - start_t) / (float)CLOCKS_PER_SEC);

	system("pause");
	return 0;
}

