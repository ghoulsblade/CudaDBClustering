#define N (1024 * 32 * 1 / 8) 
#define D 2

#define HANDLE_ERROR(x) myLastErr = cudaGetLastError(); if (myLastErr != 0) { printf("%s : %d(%s)\n",x,(int)myLastErr,cudaGetErrorString(myLastErr)); exit(0); }
#define kReportFile				"report.txt"
	
#define DATAORG_STRUCT_OF_STREAMS  // DISABLED for index : quicksort would be tricky otherwise, and data might be too large for ram
#ifdef DATAORG_STRUCT_OF_STREAMS
#define DATAPOINT_IDX(n,d) ((d)*N + (n)) // DATAORG_STRUCT_OF_STREAMS  (might be better for gpu access, but no measurable improvements in our case, avoid bank-conflict)
#else
#define DATAPOINT_IDX(n,d) ((n)*D + (d)) // DATAORG_STREAM_OF_STRUCTS  (the "usual way",default)
#endif
#define DATAPOINT(pData,n,d) ((pData)[DATAPOINT_IDX(n,d)]) // DATAORG_STREAM_OF_STRUCTS  (the "usual way",default)

// random data generation
//~ #define DATA_PURE_RANDOM
//~ #define DATA_PURE_RANDOM_MIN	(0.0)
//~ #define DATA_PURE_RANDOM_MAX	(8.0)
#define DATA_CLUST_RANDOM
#define DATA_CLUST_RANDOM_NUM_CLUST 20
#define DATA_CLUST_RANDOM_RAD_MIN 0.02
#define DATA_CLUST_RANDOM_RAD_MAX 0.05
#define DATA_CLUST_RANDOM_CENTER_MIN 0.2
#define DATA_CLUST_RANDOM_CENTER_MAX 0.8

#define DBSCAN_ENABLE_GPU 1
#define DBSCAN_ENABLE_CPU 1
//~ #define DBSCAN_ENABLE_GPU_DEBUG
#define DBSCAN_EXIT_ON_FIRST_DOUBLEASIGN 0
#define DBSCAN_FIXEDSEED 1211905618
#define DBSCAN_DUMP_STEPS		1 // dumps each step to steps/step_gpu_%05d.txt

#ifdef __DEVICE_EMULATION__
#define ROB_EMU_ACTIVE true
#else
#define ROB_EMU_ACTIVE false
#endif

// includes, system
#include <time.h> // srand
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <assert.h>

#include <sm_11_atomic_functions.h>


#define DBSCAN_PARAM_EPSILON	(0.010)	// TODO
#define DBSCAN_PARAM_SQEPSILON	(DBSCAN_PARAM_EPSILON * DBSCAN_PARAM_EPSILON)
#define DBSCAN_PARAM_MINPTS		(4)		// TODO   // currently minpts is at least 1 because self is also counted (dist=0)
#define DBSCAN_INDEX_OUTLIER	(-1)
#define DBSCAN_INDEX_INIT		(-2)
#define DBSCAN_INDEX_MARKED		(-3) // for cpu
#define GRIDHEIGHT 			1 // TODO 
#define	kThreadBlockSize	(16) // x*64:,64,128,192,256 (see cuda-guide page 65)


#define BLOCKIDX	(blockIdx.x * GRIDHEIGHT + blockIdx.y)
#define MYID		(BLOCKIDX * blockDim.x + threadIdx.x)  

#define DBSCAN_NUM_SEEDS					(16)
#define DBSCAN_CLUSTER_ADD_SCAN 			(DBSCAN_NUM_SEEDS+1) // unscanned_id + this = scanned_id
#define DBSCAN_CLUSTER_ID_INIT				(-1)
#define DBSCAN_CLUSTER_ID_UNSCANNED_FIRST	(0)
#define DBSCAN_CLUSTER_ID_UNSCANNED_LAST	(DBSCAN_NUM_SEEDS-1)
#define DBSCAN_CLUSTER_ID_NOISE				(DBSCAN_NUM_SEEDS  )
#define DBSCAN_CLUSTER_ID_SCANNED_FIRST		(DBSCAN_NUM_SEEDS+1)
#define DBSCAN_CLUSTER_ID_SCANNED_LAST		(DBSCAN_NUM_SEEDS+1+DBSCAN_NUM_SEEDS-1)
#define DBSCAN_CLUSTER_ID_FINISHED_FIRST	(DBSCAN_NUM_SEEDS+1+DBSCAN_NUM_SEEDS-1+1)

typedef struct {
	float			pPoints			[N*D];
	int				pSeedList		[DBSCAN_NUM_SEEDS];
	int				pClusterIDs		[N];
	int				pConnections	[DBSCAN_NUM_SEEDS * DBSCAN_NUM_SEEDS];
} DBScanData;

	
	
void	DBScan_WriteToFile	(DBScanData* pDBScanData,const char* szFilePath);

// code
#include "src/utils.cu"
#include "src/robintlist.cu"
#include "src/dbscan_noidx.cu"
#include "src/dbscan_noidx_cpu.cu"


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
		for (int d=0;d<D;++d) fprintf(fp,"%f ",pDBScanData->pPoints[n + d*N]);
		fprintf(fp,"%d\n",pDBScanData->pClusterIDs[n]);
	}
	fclose(fp);
}


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
	
	// random init 
	printf("randomcluster generation start\n");
	GenerateData_ClusterRandom(gDBScanData.pPoints);
	printf("randomcluster generation done\n");
	
	if (1) DBScan_WriteToFile(&gDBScanData,"out_init.txt"); 
	
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
		sprintf(mybuf,"dbscan, e^2=%f minpts=%d t_gpu=%fs t_cpu=%fs",(float)DBSCAN_PARAM_SQEPSILON,(int)DBSCAN_PARAM_MINPTS,fTime_dbscan_gpu,fTime_dbscan_cpu);
		RobWriteReportLine(kReportFile,mybuf);
	}
	
	printf("done2\n");
	
    CUT_SAFE_CALL( cutDeleteTimer(timer));
}

