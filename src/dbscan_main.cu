//~ #define N_FILE (1024 * 32 * 8 * 2)  //  * 8 * 4 // will be rounded upwards, see I0 below
//~ #define D 8

#define HANDLE_ERROR(x) myLastErr = cudaGetLastError(); if (myLastErr != 0) { printf("%s : %d(%s)\n",x,(int)myLastErr,cudaGetErrorString(myLastErr)); exit(0); }
#define kReportFile				"report.txt"

#define NORMALIZE_DATA_TO_MINMAX


//~ #define MY_DATA_FILE_BASENAME "data524288dim8clust5.txt"
//~ #define N_FILE 524288 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "data2031616dim8clust20.txt" 
//~ #define N_FILE 2097152   // increased a little to comply with IO^3 * kthreadblocksize
//~ #define N_FILE 2031616 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "data4030464dim8clust40.txt" 
//~ #define N_FILE 4194304 
//~ #define N_FILE 4030464 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "dataLIN524288dim8clust5.txt" 
//~ #define N_FILE 524288 
//~ #define D 8

//~ #define MY_DATA_FILE_BASENAME "data2031616dim32clust20.txt" 
//~ #define N_FILE 2097152 
//~ #define N_FILE 2031616 
//~ #define D 32

//~ #define N_FILE (1024 * 32 * 8)   
#define N_FILE (1024 * 512)    // test:2mio:2097152
#define D 8



//~ #define DBSCAN_PARAM_MINPTS		(2048)		// TODO   // currently minpts is at least 1 because self is also counted (dist=0)
#define DBSCAN_PARAM_MINPTS			(4)		// TODO   // currently minpts is at least 1 because self is also counted (dist=0)

#define DBSCAN_PARAM_EPSILON		(0.050)	// TODO
//~ #define DBSCAN_PARAM_EPSILON	(0.050)	// TODO


#define MYTEST_DISABLE_INDEX


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
#define DATA_PURE_RANDOM
#define DATA_CLUST_RANDOM
#define DATA_CLUST_RANDOM_NUM_CLUST 20
#define DATA_CLUST_RANDOM_RAD_MIN 0.02
#define DATA_CLUST_RANDOM_RAD_MAX 0.05
#define DATA_CLUST_RANDOM_CENTER_MIN 0.2
#define DATA_CLUST_RANDOM_CENTER_MAX 0.8
#endif

#define I0		(16)	// index segment-count (max:254)  // should be power of two for best performance
//#define I0	(64)	// index segment-count (max:254)

#define	kThreadBlockSize	(64) // x*64:,64,128,192,256 (see cuda-guide1:page65 , 2:page75)



//~ #define DATAORG_STRUCT_OF_STREAMS  // DISABLED for index : quicksort would be tricky otherwise, and data might be too large for ram
#ifdef DATAORG_STRUCT_OF_STREAMS
#define DATAPOINT_IDX(n,d) ((d)*N + (n)) // DATAORG_STRUCT_OF_STREAMS  (might be better for gpu access, but no measurable improvements in our case, avoid bank-conflict)
#else
#define DATAPOINT_IDX(n,d) ((n)*D + (d)) // DATAORG_STREAM_OF_STRUCTS  (the "usual way",default)
#endif
#define DATAPOINT(pData,n,d) ((pData)[DATAPOINT_IDX(n,d)]) // DATAORG_STREAM_OF_STRUCTS  (the "usual way",default)

#define DBSCAN_ENABLE_GPU 1
#define DBSCAN_ENABLE_CPU 1
//~ #define DBSCAN_ENABLE_GPU_DEBUG
#define DBSCAN_EXIT_ON_FIRST_DOUBLEASIGN 0
#define DBSCAN_FIXEDSEED 1211905618
#define DBSCAN_DUMP_STEPS		0 // dumps each step to steps/step_gpu_%05d.txt

#ifdef __DEVICE_EMULATION__
#define ROB_EMU_ACTIVE true
#else
#define ROB_EMU_ACTIVE false
#endif



//~ #define DBSCAN_PARAM_EPSILON	(0.010)	// TODO
#define DBSCAN_PARAM_SQEPSILON	(DBSCAN_PARAM_EPSILON * DBSCAN_PARAM_EPSILON)
#define DBSCAN_INDEX_OUTLIER	(-1)
#define DBSCAN_INDEX_INIT		(-2)
#define DBSCAN_INDEX_MARKED		(-3) // for cpu
#define GRIDHEIGHT 			1 // TODO, not fully/correctly used in BLOCKIDX and MYID


#define BLOCKIDX	(blockIdx.x * GRIDHEIGHT + blockIdx.y)
#define MYID		(BLOCKIDX * blockDim.x + threadIdx.x)  
//~ #define MYID		(BLOCKIDX * blockDim.x * blockDim.y + threadIdx.y * blockDim.y + threadIdx.x )  

#define DBSCAN_NUM_SEEDS					(128)
#define DBSCAN_NUM_BOUNDS 					(DBSCAN_NUM_SEEDS/kThreadBlockSize) // must be divisible without remainder
#define DBSCAN_BOUNDS_ELEMENTS_PER_BLOCK 	(kThreadBlockSize)
#define DBSCAN_CLUSTER_ADD_SCAN 			(DBSCAN_NUM_SEEDS+1) // unscanned_id + this = scanned_id
#define DBSCAN_CLUSTER_ID_INIT				(-1)
#define DBSCAN_CLUSTER_ID_UNSCANNED_FIRST	(0)
#define DBSCAN_CLUSTER_ID_UNSCANNED_LAST	(DBSCAN_NUM_SEEDS-1)
#define DBSCAN_CLUSTER_ID_NOISE				(DBSCAN_NUM_SEEDS  )
#define DBSCAN_CLUSTER_ID_SCANNED_FIRST		(DBSCAN_NUM_SEEDS+1)
#define DBSCAN_CLUSTER_ID_SCANNED_LAST		(DBSCAN_NUM_SEEDS+1+DBSCAN_NUM_SEEDS-1)
#define DBSCAN_CLUSTER_ID_FINISHED_FIRST	(DBSCAN_NUM_SEEDS+1+DBSCAN_NUM_SEEDS-1+1)
#define kStateEndValue  0x2fffFFFF


// segment sizes
#define I03 				(I0*I0*I0) // I0=32 -> 32k*9*4=1mb , I0=16 -> 4k*9*4=144k
#define SZ 					(int(N_FILE + I03-1)/I03) // round upwards
#define SY 					(SZ*I0)
#define SX					(SZ*I0*I0)
#define N		 			(SZ*I0*I0*I0) // rounded upwards, N >= N_FILE , this is the amount of data allocated
// DATASIZE for SZ = 254 =   254*64*64*64 / 1024 / 1024 * 4 * 9  = ca 2286mb
// DATASIZE for SZ =  32 =    32*64*64*64 / 1024 / 1024 * 4 * 9  = ca  288mb
// DATASIZE for SZ =   1 =     1*64*64*64 / 1024 / 1024 * 4 * 9  = ca    9mb
// SZ may not be larger than 254, increase I0 if that happens
// SZ=254 + I0=254 allows for  (254^4) lines of data, which is about 139 gigabyte, that should be enough for a while
// increase index type from uchar4 to ushort4 if not.... (DONE, as it's only 254^3...)

// index level starts
#define INDEXSTART_0		(0)
#define INDEXSTART_1		((I0+1))
#define INDEXSTART_2		((I0+1) + (I0+1)*I0)
#define INDEX_END			((I0+1) + (I0+1)*I0 + (I0+1)*I0*I0)
#define	INDEX_NUM_FLOATS	(INDEX_END)
#define	INDEX_SIZE			(INDEX_NUM_FLOATS * sizeof(float))
// (64 + 64*63 + 64*63*63) * sizeof(float) = ca 1mb : 3 level

// index position calc
#define INDEXPOS_0(x)		(INDEXSTART_0 + (x)									) // 	  x<=I0
#define INDEXPOS_1(x,y)		(INDEXSTART_1 + (y) + (x)*(I0+1)					) // x<I0 y<=I0
#define INDEXPOS_2(x,y,z)	(INDEXSTART_2 + (z) + (y)*(I0+1) + (x)*(I0+1)*I0	) // *<I0 z<=I0



// includes, system
#include <time.h> // srand
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <assert.h>

#include <sm_11_atomic_functions.h>

#define CANDIDATE_LIST_MAXLEN	(1024)
#define CANDIDATE_LIST_REFILL	(CANDIDATE_LIST_MAXLEN / 2)
#define SEEDSTATEDIM			5

#define NUM_THREADS_NEWSEEDSCAN (4*kThreadBlockSize)
#define NEWSEEDSCAN_NUM_PARAMS 4


typedef struct {
	float			pPoints				[N*D];	
	int				pClusterIDs			[N];
	float			pIndex				[INDEX_NUM_FLOATS];
	unsigned int	pSeedStates			[DBSCAN_NUM_SEEDS*SEEDSTATEDIM]; // (atomicops)
	unsigned int	pFinalChainIDs		[DBSCAN_NUM_SEEDS]; // (constant memory, values >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST)
	unsigned int	pCandidateLists		[DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN];
	unsigned int	pConnectionMatrix	[DBSCAN_NUM_SEEDS * DBSCAN_NUM_SEEDS];	
	unsigned int	pHelperData			[DBSCAN_NUM_SEEDS + NEWSEEDSCAN_NUM_PARAMS + NUM_THREADS_NEWSEEDSCAN];	
} DBScanData;

	
	
void	DBScan_WriteToFile	(DBScanData* pDBScanData,const char* szFilePath);

// code
#include "src/utils.cu"
#include "src/robintlist.cu"
#include "src/dbscan.cu"
#include "src/dbscan_cpu.cu"
#include "src/dbscan_idx.cu"


void runTest( int argc, char** argv);


typedef struct {
	int				a[200][10];
} MultiDimArrayTestStruct;	
	

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv) 
{
	/*
	MultiDimArrayTestStruct p;
	#define MultiDimArrayTest(a,b) printf("%s-%s=%d\n",#a,#b,a-b);
	MultiDimArrayTest(&p.a[0][0],&p.a[0][1]) //~ &p.a[0][0]-&p.a[0][1]=-1
	MultiDimArrayTest(&p.a[0][0],&p.a[1][0]) //~ &p.a[0][0]-&p.a[1][0]=-10
	*/
	
	printf("running in %s mode\n",ROB_EMU_ACTIVE?"EMULATION":"REAL-HARDWARE");
	printf("__DEVICE_EMULATION__ -> %d\n",ROB_EMU_ACTIVE?1:0);
	
	InitGlobalTimer();
	do {
		runTest( argc, argv);
	} while (DBSCAN_EXIT_ON_FIRST_DOUBLEASIGN) ;

    CUT_EXIT(argc, argv);
}

DBScanData gDBScanData;

void	DBScan_WriteToFile	(DBScanData* pDBScanData,const char* szFilePath) {
	FILE* fp = fopen(szFilePath,"w");
	if (!fp) { printf("DBScan_WriteToFile : warning, couldn't open file for dump : %s\n",szFilePath); return; }
	for (int n=0;n<N;++n) {
		for (int d=0;d<D;++d) fprintf(fp,"%f ",DATAPOINT(pDBScanData->pPoints,n,d));
		fprintf(fp,"%d\n",pDBScanData->pClusterIDs[n]);
	}
	fclose(fp);
}


char gsInfoGPU[512] = "";

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest( int argc, char** argv) {
    printf("DBScan\n");
	
    unsigned int timer;
    cutCreateTimer(&timer);

    CUT_DEVICE_INIT(argc,argv);

    // initalize the random generator
	unsigned int iMySeed = time(NULL);
	#ifdef DBSCAN_FIXEDSEED
	iMySeed = DBSCAN_FIXEDSEED;
	#endif
	
	printf("iMySeed=%d\n",iMySeed);
    srand(iMySeed);
	
	float fTime_dbscan_idx = 0;
	
	// random init 
	printf("randomcluster generation / fileread start\n");
	#ifdef MY_DATA_FILE
	ReadTextData(MY_DATA_FILE,gDBScanData.pPoints);
	#else
	GenerateData_ClusterRandom(gDBScanData.pPoints);
	#endif
	printf("randomcluster generation / fileread done\n");
	printf("index generation start\n");
	#ifdef DATAORG_STRUCT_OF_STREAMS
	printf("ERROR : index doesn't support DATAORG_STRUCT_OF_STREAMS\n");
	exit(0);
	#endif
	RobStartTimer();
	IndexStructure_Generate(gDBScanData.pPoints,gDBScanData.pIndex);
	fTime_dbscan_idx = RobStopTimer();
	printf("index generation done\n");
	
	if (0) DBScan_WriteToFile(&gDBScanData,"out_init.txt"); 
	
	float fTime_dbscan_cpu = 0;
	float fTime_dbscan_gpu = 0;
	
	// dbscan_gpu
	if (DBSCAN_ENABLE_GPU) {
		printf("dbscan_gpu start\n");
		RobStartTimer();
		dbscan_gpu(&gDBScanData);
		fTime_dbscan_gpu = RobStopTimer();
		printf("dbscan_gpu done (time=%fs)\n",fTime_dbscan_gpu);
		
		
		// count clusters
		float fTime_dbscan_gpu_count_cluster;
		printf("dbscan_gpu_count_clusters start\n");
		RobStartTimer();
		RobIntList	oList_ClusterIDs;
		RobIntList_Init(&oList_ClusterIDs);
		for (int n=0;n<N;++n) {
			int iClusterID = gDBScanData.pClusterIDs[n];
			bool bFound = false;
			for (int i=0;i<oList_ClusterIDs.iSize;++i) 
				if (oList_ClusterIDs.pData[i] == iClusterID) { bFound = true; break; }
			if (!bFound) RobIntList_Push(&oList_ClusterIDs,iClusterID);
		}
		
		printf("dbscan_gpu_count_clusters : %d\n",oList_ClusterIDs.iSize);
		fTime_dbscan_gpu_count_cluster = RobStopTimer();
		printf("dbscan_gpu_count_clusters done (time=%fs)\n",fTime_dbscan_gpu_count_cluster);
			
		sprintf(gsInfoGPU,"clust=%d|%s",(int)oList_ClusterIDs.iSize,gsInfoGPUaux);
			
		if (1) DBScan_WriteToFile(&gDBScanData,"out_gpu.txt"); 
	}
	
	// cpu
	if (DBSCAN_ENABLE_CPU) {
		printf("dbscan_cpu start\n");
		RobStartTimer();
		dbscan_cpu(&gDBScanData);
		fTime_dbscan_cpu = RobStopTimer();
		printf("dbscan_cpu done (time=%fs)\n",fTime_dbscan_cpu);
		
		if (1) DBScan_WriteToFile(&gDBScanData,"out_cpu.txt"); 
	}
	
	// report
	if (1) {
		char mybuf[512]; 
		float fSpeedUp = (fTime_dbscan_cpu > 0.0 && fTime_dbscan_gpu > 0.0) ? (fTime_dbscan_cpu / fTime_dbscan_gpu) : 0;
		//~ sprintf(mybuf,"kmeans, K=%d e=%f gpu:%0.1fs(%s) cpu:%0.1fs(%s) speedup:%0.1f %s",(int)K,kCostDiffEpsilon,
		//~ fTime_kmeans_gpu,gsInfoGPU,fTime_kmeans_cpu,gsInfoCPU,fSpeedUp,kReportCustomString);
		
		const char* sNoIdxAdd = "";
		#ifdef MYTEST_DISABLE_INDEX 
		sNoIdxAdd = "NOIDX";
		#endif
		
		sprintf(mybuf,"dbscan %s, e=%f e^2=%f kThreadBlockSize=%d I0=%d minpts=%d index:%0.1fs gpu:%0.1fs(%s) cpu:%0.1fs(%s) speedup:%0.1f %s",
			sNoIdxAdd,
			(float)DBSCAN_PARAM_EPSILON,(float)DBSCAN_PARAM_SQEPSILON,(int)kThreadBlockSize,(int)I0,(int)DBSCAN_PARAM_MINPTS,
			fTime_dbscan_idx,fTime_dbscan_gpu,gsInfoGPU,fTime_dbscan_cpu,gsInfoCPU,fSpeedUp,
			kReportCustomString);
		RobWriteReportLine(kReportFile,mybuf);
	}
	
	printf("done2\n");
	
    CUT_SAFE_CALL( cutDeleteTimer(timer));
}

