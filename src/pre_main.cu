/// LMU Muenchen ProjektArbeit Robert Noll 2007 : Similarity Join mittels Grafikprozessor

#ifdef _WIN32
#define NOMINMAX 
#endif

// includes
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cutil.h>
#include <cuda_runtime.h>
#include <sm_11_atomic_functions.h>

// note : for threads you need windows.h from the platform sdk : 
// http://www.microsoft.com/downloads/details.aspx?FamilyId=A55B6B43-E24F-4EA3-A93E-40C0EC4F68E5&displaylang=en
#include "threads.h"

#undef assert
//~ #define assert(code) printf("assert %s : (%s)\n",(int)(code) ? "passed" : "FAILED",#code); if (!(int)(code)) exit(0);
#define assert(code) if (!(int)(code)) { printf("assert FAILED : (%s)\n",#code); exit(0); }

// input parameters
#define N_FILE_1MB_PER_DIM		 (1024*1024*8/32)			// = 9mb  for D=9 floats..
							//	 (1024*1024*8/32)			// = 9mb  for D=9 floats..
							//	 N_FILE_1MB_PER_DIM			= 9mb  for 9 floats..
							//	 N_FILE_1MB_PER_DIM*10		= 90mb
							//	 N_FILE_1MB_PER_DIM*10*4	= 360mb
							
							
#define N_FILE			 		(N_FILE_1MB_PER_DIM) // 9*128 = 1152mb / 4 = 288
#define	D						(9)
#define TEST_EPSILON			0.5

#define ENABLE_GPU_IDX3
#define ENABLE_CPU_IDX3

// filepath (todo : read from commandline)
//~ #define kDataFilePath			"data/coarse-strong-ultra-100-96_10000.txt"	// ??? format unknown.. very big lines...
#define kDataFilePath			"data/Corel_ColorMoments_9d.ascii"	// 68040 lines : n=65536
//~ #define kDataFilePath			"data/smalldata.ascii" 			// 16383 lines : n=16384
#define kReportFile				"report.txt"

// constants
#define MB	 					(1024 * 1024) // bytes per megabyte
#define	GRIDHEIGHT				4  
#define	kThreadBlockSize		(64) // x*64:,64,128,192,256 (see cuda-guide page 65)
#define	kWrapSize				(32) // some code in the kernel might depend on this being exactly 32
#define SIZE_RESULTLIST			(64 * MB) // 16 mb depends on graka mem, but not needed dynamically
#define kMaxResults				(SIZE_RESULTLIST/sizeof(ulong2)) // 1 result = 2 uint32 indices
#define kLastValidResultIndex	(kMaxResults-1)
#define kGPU_COUNT				2


#define I0		(32)	// index segment-count (max:254)  // should be power of two for best performance
//#define I0	(64)	// index segment-count (max:254)

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
#define INDEXSTART_0	(0)
#define INDEXSTART_1	((I0+1))
#define INDEXSTART_2	((I0+1) + (I0+1)*I0)
#define INDEX_END		((I0+1) + (I0+1)*I0 + (I0+1)*I0*I0)
#define	SIZE_INDEX		(INDEX_END * sizeof(float))
// (64 + 64*63 + 64*63*63) * sizeof(float) = ca 1mb : 3 level

// index position calc
#define INDEXPOS_0(x)		(INDEXSTART_0 + (x)									) // 	  x<=I0
#define INDEXPOS_1(x,y)		(INDEXSTART_1 + (y) + (x)*(I0+1)					) // x<I0 y<=I0
#define INDEXPOS_2(x,y,z)	(INDEXSTART_2 + (z) + (y)*(I0+1) + (x)*(I0+1)*I0	) // *<I0 z<=I0

// n = iNumLines = number of data-points
#define DATASIZE_IN_RAW			(N*D*sizeof(float))
#define DATASIZE_IN_STATE		((N/kThreadBlockSize)*sizeof(uint4))
#define DATASIZE_IN_INDEX		(SIZE_INDEX)
#define DATASIZE_IN_TOTAL		(DATASIZE_IN_RAW + DATASIZE_IN_STATE + DATASIZE_IN_INDEX)
#define DATASIZE_OUT_ATOM		(sizeof(unsigned int) * kThreadBlockSize * 2)
#define DATASIZE_OUT_LIST		(SIZE_RESULTLIST)
#define DATASIZE_OUT_TOTAL		(DATASIZE_OUT_ATOM + DATASIZE_OUT_LIST)

#define kStateEndValue  0x7fffFFFF




// forward declarations
int		ReadTextData		(const char* szFilePath,float* pData);
#include "utils.cu"
#include "idx_kernel.cu"
#include "idx_prepare.cu"
#include "idx_cpu.cu"
#include "nestedloop.cu"

float*	gpDataIn_Raw	= 0;	
float*	gpDataIn_Index	= 0;	

// ##### ##### ##### ##### ##### dataparts for threads

#define kMaxPartCount		I0
#define kMaxPartDataSize	(9 * 4 * 11 * MB)  // = 396mb = ok on both devices

class cDataPart { public :
	int		iPartID;
	int		iPartElementsFirst;	
	int		iPartElementsLast;	
	int		iPartViewFirst;
	int		iPartViewLast;
	int		iViewDataOffsetInFloats; ///< global offset
	int		iViewDataOffsetInPoints; ///< global offset
	int		iViewDataSize; ///< in bytes
	int		iElementOffsetInPoints; ///< offset relative to part start
	int		iElementCount; ///< in datapoints
	int		iResultCount;
	float	fEpsilon;
	
	cDataPart();
	bool	Prepare	(int p,int iPartCount,float e);
	void	Run		(); // executed in thread
	
};

cDataPart::cDataPart() {}
	
bool	cDataPart::Prepare		(int p,int iPartCount,float e) {
	iPartID = p;
	fEpsilon = e;
	iResultCount = 0;
	
	// determine the datarange that will be managed by this part/thread
	int iMaxPartLen = (I0 + iPartCount - 1)/iPartCount; // round up
	iPartElementsFirst	= p*iMaxPartLen; // each element gets its own thread later
	iPartElementsLast	= min(iPartElementsFirst+iMaxPartLen-1,I0-1); 
	
	// determine the view = the area in the data that has to be scanned, e.g. elements + epsilon border
	iPartViewFirst	= 0;	// the view is the data range that has to be examined
	iPartViewLast	= I0-1; // inclusive
	#define GET_MIN_0(x)	gpDataIn_Index[INDEXPOS_0(x)]
	#define GET_MAX_0(x)	GET_MIN_0((x+1))
	float fMin		= GET_MIN_0(iPartElementsFirst	) - fEpsilon; 
	float fMax		= GET_MAX_0(iPartElementsLast	) + fEpsilon; // max(i)=min(i+1)
	while (iPartViewFirst	< I0-1	&& GET_MAX_0(iPartViewFirst)	< fMin) ++iPartViewFirst;
	while (iPartViewLast	> 0		&& GET_MIN_0(iPartViewLast)		> fMax) --iPartViewLast;
	assert(iPartViewFirst 	<= iPartElementsFirst);
	assert(iPartViewLast	>= iPartElementsLast);
	
	// calc mem usage and positions
	iViewDataSize				= (iPartViewLast+1     - iPartViewFirst		)	* SX * D * sizeof(float); // in bytes
	iElementCount				= (iPartElementsLast+1 - iPartElementsFirst	)	* SX; // in datapoints
	iViewDataOffsetInPoints		= iPartViewFirst								* SX;
	iViewDataOffsetInFloats		= iPartViewFirst								* SX * D;
	iElementOffsetInPoints		= (iPartElementsFirst - iPartViewFirst)			* SX;
	
	// print a little debug info
	printf("parts=%d [%d]: elem[%d,%d] view[%d,%d] datasize=%dMB\n",iPartCount,p,
		iPartElementsFirst,iPartElementsLast,
		iPartViewFirst,iPartViewLast,iViewDataSize/MB);
	
	// check if this part fits in vram
	if (iViewDataSize > kMaxPartDataSize) return false;
	return true;
}

void	cDataPart::Run		() { // executed in thread
	int iGPUId = iPartID % kGPU_COUNT;
	CUDA_SAFE_CALL(cudaSetDevice(iGPUId));
	
	// alloc and init device memory (vram)
	unsigned int*	pDataOut_Atom	= (unsigned int*)	malloc(DATASIZE_OUT_ATOM);
	ulong2*			pDataOut_List	= (ulong2*)			malloc(DATASIZE_OUT_LIST);
	float*			pDataInD_Raw	= NULL;
	float*			pDataInD_Index	= NULL;
	uint4*			pDataInD_State	= NULL;
	unsigned int*	pDataOutD_Atom	= NULL;
	ulong2*			pDataOutD_List	= NULL;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataOutD_Atom,	DATASIZE_OUT_ATOM	)); assert(pDataOutD_Atom);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataOutD_List,	DATASIZE_OUT_LIST	)); assert(pDataOutD_List);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataInD_Index,	DATASIZE_IN_INDEX	)); assert(pDataInD_Index);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataInD_State,	DATASIZE_IN_STATE	)); assert(pDataInD_State);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataInD_Raw,		iViewDataSize		)); assert(pDataInD_Raw);  // biggest last
	CUDA_SAFE_CALL( cudaMemcpy(pDataInD_Raw,	gpDataIn_Raw+iViewDataOffsetInFloats,	iViewDataSize,	cudaMemcpyHostToDevice	)); 
	CUDA_SAFE_CALL( cudaMemcpy(pDataInD_Index,	gpDataIn_Index,	DATASIZE_IN_INDEX,						cudaMemcpyHostToDevice	)); 
	CUDA_SAFE_CALL( cudaMemset(pDataInD_State,	0,				DATASIZE_IN_STATE 	)); 
	CUDA_SAFE_CALL( cudaMemset(pDataOutD_Atom,	0,				DATASIZE_OUT_ATOM 	)); 
	
	// prepare device environment
	dim3  block_size;
	dim3  grid_size;
	grid_size.x		= (iElementCount / kThreadBlockSize / GRIDHEIGHT);
	grid_size.y		= GRIDHEIGHT;
	block_size.x	= kThreadBlockSize;
	unsigned int mem_shared = 0; // this is for dynamic alloc of shared mem, we alloc statically
	
	#define kMaxGridSize (63*1024)
	assert(grid_size.x <= kMaxGridSize && "grid_size.x larger than supported (cudaGetDeviceProperties:maxGridSize: 63k currently)"); 
	
	// precalc params as much as possible, i think they are only needed once for the whole program, not for every thread
	do {
		// run kernel
		mykernel<<<grid_size, block_size, mem_shared >>>(
			pDataInD_Raw,
			pDataInD_Index,
			pDataInD_State,
			pDataOutD_Atom,
			pDataOutD_List, 
			fEpsilon,
			fEpsilon*fEpsilon,
			iElementOffsetInPoints,
			iViewDataOffsetInPoints);
		CUT_SAFE_CALL( cudaThreadSynchronize()); // kernel start is async, so wait for finish here
		printf("datapart %d : exec kernel on device finished one step\n",iPartID);

		// copy result from device to host
		CUDA_SAFE_CALL( cudaMemcpy(pDataOut_Atom,	pDataOutD_Atom, DATASIZE_OUT_ATOM,	cudaMemcpyDeviceToHost	)); // get counter
		CUDA_SAFE_CALL( cudaMemset(pDataOutD_Atom,	0,				DATASIZE_OUT_ATOM							)); // clear counter
		int iNumResults = min(kMaxResults,pDataOut_Atom[0]);
		int iListSize = iNumResults * sizeof(ulong2);
		CUDA_SAFE_CALL( cudaMemcpy(pDataOut_List,	pDataOutD_List, iListSize, 			cudaMemcpyDeviceToHost	)); // get list
		printf("datapart %d : receive results from device\n",iPartID);
		iResultCount += iNumResults;
		
		printf("atom[0]=%d atom[1]=%d iNumResults=%d kMaxResults=%d\n",pDataOut_Atom[0],pDataOut_Atom[1],iNumResults,kMaxResults);
		
		// todo : do something with the data here, e.g. printf, or write to file ?
	
		if (pDataOut_Atom[0] < kMaxResults) break; // detect if finished
	} while (1);
	
	// release device memory
	CUDA_SAFE_CALL(cudaFree(pDataInD_Raw));
	CUDA_SAFE_CALL(cudaFree(pDataInD_Index));
	CUDA_SAFE_CALL(cudaFree(pDataInD_State));
	CUDA_SAFE_CALL(cudaFree(pDataOutD_Atom));
	CUDA_SAFE_CALL(cudaFree(pDataOutD_List));
	
	free(pDataOut_Atom);
	free(pDataOut_List);
}
	
cDataPart*	gDataParts[kMaxPartCount];


////////////////////////////////////////////////////////////////////////////////
// GPU thread
////////////////////////////////////////////////////////////////////////////////

static CUT_THREADPROC gpuThread(int * piPartID) {
	//gDataParts[*piPartID]->Run();
    CUT_THREADEND;
}

// ##### ##### ##### ##### ##### main


int main( int argc, char** argv) {
    CUT_DEVICE_INIT();
	
	int count = 0;
	CUDA_SAFE_CALL( cudaGetDeviceCount(&count));
	assert(count >= kGPU_COUNT);
	
	
	
	float	fEpsilon = TEST_EPSILON; // todo : read from arg ?
	float	fSqEpsilon = fEpsilon*fEpsilon;
	float	fTimeGPU = 1;
	float	fTimeCPU = 1;
	int		iResultsGPU = 0;
	int		iResultsCPU = 0;
	
	// print device infos
	if (0) PrintDeviceInfos();
	
	
	// size calc and alloc
	gpDataIn_Raw					= (float*)			malloc(DATASIZE_IN_RAW);
	gpDataIn_Index					= (float*)			malloc(DATASIZE_IN_INDEX);
	
	// print some infos
	printf("N=%d\n",N);
	printf("SX=%d\n",SX);
	printf("SY=%d\n",SY);
	printf("SZ=%d\n",SZ);
	printf("I0=%d\n",I0);
	printf("DATASIZE_IN_RAW=%dkb\n",		DATASIZE_IN_RAW		/1024);
	printf("DATASIZE_IN_STATE=%dkb\n",		DATASIZE_IN_STATE	/1024);
	printf("DATASIZE_IN_INDEX=%dkb\n",		DATASIZE_IN_INDEX	/1024);
	printf("DATASIZE_IN_TOTAL=%dkb\n",		DATASIZE_IN_TOTAL	/1024);
	printf("DATASIZE_OUT_TOTAL=%dkb\n",		DATASIZE_OUT_TOTAL	/1024);
	
		
	// read file
	RobStartTimer();
	ReadTextData(kDataFilePath,gpDataIn_Raw);			// read raw data from file
	printf("%4.2f sec : reading data from file\n",RobStopTimer());
	
	// generate index
	RobStartTimer();
	IndexStructure_Generate(gpDataIn_Raw,gpDataIn_Index);	// gen index and sort raw data
	printf("%4.2f sec : generating index data\n",RobStopTimer());
	
	
    if (1) {
		RobStartTimer();			printf("starting gpu...\n");
		
        int 		threadIds[kGPU_COUNT];
        CUTThread	threads[kGPU_COUNT];
		int iPartCount = 0;
		
		{ for (int i=0;i<kMaxPartCount;++i) gDataParts[i] = new cDataPart();  }
		
		// determine how the data will be subdivided
		bool bOK = false;
		for (iPartCount=kGPU_COUNT;iPartCount <= kMaxPartCount;iPartCount*=2) {
			bOK = true;
			for (int p=0;p<iPartCount;++p) if (gDataParts[p]->Prepare(p,iPartCount,fEpsilon)) { bOK = false; break; }
			if (bOK) break;
		}
		if (!bOK) { printf("failed to find a partitioning that fits in vram, try lowering epsilon, or increase I0\n"); exit(1); }
		//~ for (int j=0;j<I0/2;++j) printf("%d:%+0.1f  %d:%+0.1f\n",j,GET_MIN_0(j),j+I0/2,GET_MIN_0(j+I0/2));

		gDataParts[0]->Run();
		
		// start threads
		{ for (int i=0;i<iPartCount;++i) {
			int iThreadIdx = i % kGPU_COUNT;
            threadIds[iThreadIdx] = i;
            threads[iThreadIdx] = cutStartThread((CUT_THREADROUTINE)gpuThread, (void *)&threadIds[iThreadIdx]);
			if (iThreadIdx == kGPU_COUNT-1) cutWaitForThreads(threads, kGPU_COUNT); // Wait for all the threads to finish.
		} }
		
		iResultsGPU = 0;
		{ for (int i=0;i<iPartCount;++i) iResultsGPU += gDataParts[i]->iResultCount; }
		
		printf("all threads finished\n");
		fTimeGPU = RobStopTimer();	printf("%4.2f sec : gpu\n",fTimeGPU);
    }
	
	if (0) { // NestedLoop
		RobStartTimer();			printf("check : NestedLoop...\n");
		iResultsCPU = NestedLoop(gpDataIn_Raw,fSqEpsilon); 
		fTimeCPU = RobStopTimer();	printf("%4.2f sec : check : NestedLoop\n",fTimeCPU);
	}
	
	if (0) { // Idx_CPU
		RobStartTimer();		printf("check : with index on cpu...\n");
		iResultsCPU = Idx_CPU(gpDataIn_Raw,gpDataIn_Index,fEpsilon,fSqEpsilon); 
		fTimeCPU = RobStopTimer(); printf("%4.2f sec : check : with index on cpu\n",fTimeCPU);
	}
		
	
	if (1) { // write report
		FILE* fp = fopen(kReportFile,"a");
		if (fp) {
			int iGPU_IDX3 = 0;
			int iCPU_IDX3 = 0;
			#ifdef ENABLE_GPU_IDX3
			iGPU_IDX3 = 1;
			#endif
			#ifdef ENABLE_CPU_IDX3
			iCPU_IDX3 = 1;
			#endif
			
			float fTolerance = 0.01; // 1 % tolerance due to less exact 
			float fRelError = float(abs(iResultsCPU - iResultsGPU)) / float( (iResultsCPU > 0) ? iResultsCPU : 1 );
			bool bOK = fRelError < fTolerance; // error relative to "correct" cpu results...
			float fErrorPercent = ceil(fRelError * 1000.0) * 0.1;
			#define REPORT "gpu/cpu=%8.1f %s err<=%0.1f%% N=%d size=%dMB I0=%d i3:%d,%d tgpu=%0.1fs tcpu=%0.1f\n", \
				fTimeGPU/fTimeCPU,bOK?" ok ":"MISS", fErrorPercent, \
				N,DATASIZE_IN_RAW/MB,I0,iGPU_IDX3,iCPU_IDX3,fTimeGPU,fTimeCPU
			printf(			REPORT );
			fprintf( fp,	REPORT );
			fclose(fp);
		}
	}
	
	
	// release memory
    free(gpDataIn_Raw);
    free(gpDataIn_Index);

    CUT_EXIT(argc, argv);
	return 0;
}


