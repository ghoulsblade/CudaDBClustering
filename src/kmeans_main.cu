//~ #define N (1024 * 32 * 8 * 8)  // will be rounded upwards, see I0 below

#define HANDLE_ERROR(x) myLastErr = cudaGetLastError(); if (myLastErr != 0) { printf("%s : %d(%s)\n",x,(int)myLastErr,cudaGetErrorString(myLastErr)); exit(0); }
#define kReportFile				"report.txt"

//~ #define MY_DATA_FILE_BASENAME "data524288dim8clust5.txt"
//~ #define K 5
//~ #define N 524288 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "data2031616dim8clust20.txt" 
//~ #define K 20
//~ #define N 2031616 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "data4030464dim8clust40.txt" 
//~ #define K 40
//~ #define N 4030464 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "dataLIN524288dim8clust5.txt" 
//~ #define K 5
//~ #define N 524288 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "data2031616dim32clust20.txt" 
//~ #define K 20
//~ #define N 2031616 
//~ #define D 32

#define K 64 // todo : DATA_CLUST_RANDOM_NUM_CLUST below ? 
#define N 1024 * 1024 * 2
#define D 8


// 1048576 / 1024 / 32
// 524288=1024*32*16 // 16 mb

#ifdef MY_DATA_FILE_BASENAME
#define MY_DATA_FILE "data/" MY_DATA_FILE_BASENAME
#define kReportCustomString MY_DATA_FILE_BASENAME
#else
#define kReportCustomString "builtin_random"
#endif

#define DATA_PURE_RANDOM_MIN	(0.0)
#define DATA_PURE_RANDOM_MAX	(1.0)
#ifndef MY_DATA_FILE
// random data generation
//~ #define DATA_PURE_RANDOM
#define DATA_CLUST_RANDOM
//~ #define DATA_CLUST_RANDOM_NUM_CLUST 1024
#define DATA_CLUST_RANDOM_NUM_CLUST (K)
#define DATA_CLUST_RANDOM_RAD_MIN 0.02
#define DATA_CLUST_RANDOM_RAD_MAX 0.15
#define DATA_CLUST_RANDOM_CENTER_MIN 0.2
#define DATA_CLUST_RANDOM_CENTER_MAX 0.8
#endif

//~ #define DATAORG_STRUCT_OF_STREAMS  // DISABLED for index : quicksort would be tricky otherwise, and data might be too large for ram
#ifdef DATAORG_STRUCT_OF_STREAMS
#define DATAPOINT_IDX(n,d) ((d)*N + (n)) // DATAORG_STRUCT_OF_STREAMS  (might be better for gpu access, but no measurable improvements in our case, avoid bank-conflict)
#else
#define DATAPOINT_IDX(n,d) ((n)*D + (d)) // DATAORG_STREAM_OF_STRUCTS  (the "usual way",default)
#endif
#define DATAPOINT(pData,n,d) ((pData)[DATAPOINT_IDX(n,d)]) // DATAORG_STREAM_OF_STRUCTS  (the "usual way",default)


// includes, system
#include <time.h> // srand
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <assert.h>

// code
#include "src/utils.cu"
#include "src/kmeans.cu"


void runTest( int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	InitGlobalTimer();
    runTest( argc, argv);

    CUT_EXIT(argc, argv);
}


KMeansData gKMeansData;

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv) {
    printf("KMeans\n");
    unsigned int timer;
    cutCreateTimer(&timer);

    CUT_DEVICE_INIT(argc,argv);

    // initalize the random generator 
    srand(time(NULL));
	
	// random init  
	printf("randomcluster generation start\n");
	RobStartTimer();
	
	#ifdef MY_DATA_FILE
	ReadTextData(MY_DATA_FILE,gKMeansData.pPoints);
	#else
	GenerateData_ClusterRandom(gKMeansData.pPoints);
	#endif
	
	for (int iMeasureCounter=0;iMeasureCounter<1;++iMeasureCounter) {
	
	float fRandGenTime = RobStopTimer();
	
	// initial medoid assignment : extract k random points as medoids
	int k,n,d;
	for (k=0;k<K;++k) {
		n = (rand() >> 2) % N; // TODO : better random choice here ?
		//~ printf("initial med k=%d N=%d n=%d\n",k,N,n);
		for (d=0;d<D;++d) gKMeansData.pInitialMedoids[k*D+d] = gKMeansData.pPoints[n*D+d];
	}
	printf("randomcluster generation done (time=%fs)\n",fRandGenTime);
	
	float fTime_kmeans_cpu = 0;
	float fTime_kmeans_gpu = 0;
	int iRepeatGPU = 1;
	int iRepeatCPU = 1;
	
	// kmeans_gpu
	if (1) {
		printf("kmeans_gpu start\n");
		RobStartTimer();
		for (int i=0;i<iRepeatGPU;++i) kmeans_gpu(&gKMeansData);
		fTime_kmeans_gpu = RobStopTimer();
		printf("kmeans_gpu done (time=%fs)\n",fTime_kmeans_gpu);
	}
	
	// cpu
	if (1) {
		printf("kmeans_cpu start\n");
		RobStartTimer();
		for (int i=0;i<iRepeatCPU;++i) kmeans_cpu(&gKMeansData);
		fTime_kmeans_cpu = RobStopTimer();
		printf("kmeans_cpu done (time=%fs)\n",fTime_kmeans_cpu);
	}
	
	// report
	if (1) {
		char mybuf[512];
		float fSpeedUp = (fTime_kmeans_cpu > 0.0 && fTime_kmeans_gpu > 0.0) ? (fTime_kmeans_cpu / fTime_kmeans_gpu) : 0;
		sprintf(mybuf,"kmeans, K=%d e=%f gpu:%0.1fs(%s) cpu:%0.1fs(%s) speedup:%0.1f %s",(int)K,kCostDiffEpsilon,fTime_kmeans_gpu,gsInfoGPU,fTime_kmeans_cpu,gsInfoCPU,fSpeedUp,kReportCustomString);
		RobWriteReportLine(kReportFile,mybuf);
	}
	
	}
	
	// TODO : 
	/*
	typedef struct {
		float			pPoints				[N*D];
		float			pMedoids			[K*D];
		float			pMedoidCounts		[K];
		float			pClosestMedoidSqDist[N];
		unsigned int	pClosestMedoidIndex	[N];
	} KMeansData;
	*/


    // setup execution parameters
    //~ dim3 grid(size_x / BLOCK_DIM, size_y / BLOCK_DIM, 1);
    //~ dim3 threads(BLOCK_DIM, BLOCK_DIM, 1);

    // warmup so we don't time CUDA startup
    //~ transpose_naive<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
    //~ transpose<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
    
    // execute the kernel
    //~ cutStartTimer(timer);
    //~ for (int i = 0; i < numIterations; ++i)
    //~ {
        //~ transpose_naive<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
    //~ }
    //~ cudaThreadSynchronize();
    //~ cutStopTimer(timer);
    //~ float naiveTime = cutGetTimerValue(timer);
	

    // execute the kernel
    
    //~ cutResetTimer(timer);
    //~ cutStartTimer(timer);
    //~ for (int i = 0; i < numIterations; ++i)
    //~ {
        //~ transpose<<< grid, threads >>>(d_odata, d_idata, size_x, size_y);
    //~ }
    //~ cudaThreadSynchronize();
    //~ cutStopTimer(timer);
    //~ float optimizedTime = cutGetTimerValue(timer);

    //~ printf("Naive transpose average time:     %0.3f ms\n", naiveTime / numIterations);
    //~ printf("Optimized transpose average time: %0.3f ms\n\n", optimizedTime / numIterations);

    // check if kernel execution generated and error
    //~ CUT_CHECK_ERROR("Kernel execution failed");

    // copy result from device to    host
    //CUDA_SAFE_CALL( cudaMemcpy( h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost) );

    // cleanup memory
    //CUDA_SAFE_CALL(cudaFree(d_odata));
	
	printf("done2\n");
	
    CUT_SAFE_CALL( cutDeleteTimer(timer));
}



