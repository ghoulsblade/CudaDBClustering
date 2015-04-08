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

#define N_FILE_1MB_PER_DIM		 (1024*1024 / 4)			// = 9mb  for D=9 floats..
							//	 N_FILE_1MB_PER_DIM			= 9mb  for 9 floats..
							//	 N_FILE_1MB_PER_DIM*10		= 90mb
							//	 N_FILE_1MB_PER_DIM*10*4	= 360mb
							
							
// input parameters  
#define	D						(8) 
#define N_FILE			 		(N_FILE_1MB_PER_DIM * 1 / D) 
#define TEST_EPSILON			1.0
//~ #define GPU_NESTED_LOOP			// nested loop on gpu, don't use index
//~ #define ENABLE_GPU_WRITEOUTLIST	
//~ #define ENABLE_GPU_WRITEOUTLIST_BIG_RESULT_LIST // activates code for the case that the result list doesn't fit in vram and has to be written out multiple times	

#define ENABLE_MEASUREMENT_GPU
//~ #define ENABLE_MEASUREMENT_CPU_IDX		// don't enable both cpu_idx and nest at the same time
//~ #define ENABLE_MEASUREMENT_CPU_NEST		// don't enable both cpu_idx and nest at the same time

#define ENABLE_GPU_IDX3
#define ENABLE_CPU_IDX3

// filepath (todo : read from commandline)
//~ #define kDataFilePath			"data/coarse-strong-ultra-100-96_10000.txt"	// ??? format unknown.. very big lines...
#define kDataFilePath			"data/Corel_ColorMoments_9d.ascii"	// 68040 lines : n=65536
//~ #define kDataFilePath			"data/Corel_CoocTexture_16d.ascii"	// 68040 lines : n=65536
//~ #define kDataFilePath			"data/smalldata.ascii" 			// 16383 lines : n=16384
#define kReportFile				"report.txt"


// random data generation
//~ #define DATA_PURE_RANDOM
//~ #define DATA_PURE_RANDOM_MIN	(0.0)
//~ #define DATA_PURE_RANDOM_MAX	(8.0)
#define DATA_CLUST_RANDOM
#define DATA_CLUST_RANDOM_NUM_CLUST 6
#define DATA_CLUST_RANDOM_RAD_MIN 1.0
#define DATA_CLUST_RANDOM_RAD_MAX 2.0
#define DATA_CLUST_RANDOM_CENTER_MIN 2.0
#define DATA_CLUST_RANDOM_CENTER_MAX 6.0


// constants
#define MB	 					(1024 * 1024) // bytes per megabyte
#define	GRIDHEIGHT				4  
#define	kThreadBlockSize		(64) // x*64:,64,128,192,256 (see cuda-guide page 65)
#define	kWrapSize				(32) // some code in the kernel might depend on this being exactly 32
#define SIZE_RESULTLIST			(64 * MB) // 16 mb depends on graka mem, but not needed dynamically
#define kMaxResults				(SIZE_RESULTLIST/sizeof(ulong2)) // 1 result = 2 uint32 indices
#define kLastValidResultIndex	(kMaxResults-1)
#define kGPU_COUNT				1


#define I0		(16)	// index segment-count (max:254)  // should be power of two for best performance
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
#define DATASIZE_IN_STATE		((N/kThreadBlockSize)*sizeof(uint4))  // this is too big, but easier this way
#define DATASIZE_IN_INDEX		(SIZE_INDEX)
#define DATASIZE_IN_TOTAL		(DATASIZE_IN_RAW + DATASIZE_IN_STATE + DATASIZE_IN_INDEX)
#define DATASIZE_OUT_ATOM		(sizeof(unsigned int) * 2)
#define DATASIZE_OUT_LIST		(SIZE_RESULTLIST)
#define DATASIZE_OUT_TOTAL		(DATASIZE_OUT_ATOM + DATASIZE_OUT_LIST)

#define kStateEndValue  0x2fffFFFF




// forward declarations
int		ReadTextData		(const char* szFilePath,float* pData);
#include "utils.cu"
#include "idx_kernel.cu"
#include "idx_prepare.cu"
#include "idx_cpu.cu"
#include "nestedloop.cu"
#include <time.h>

float*	gpDataIn_Raw	= 0;	
float*	gpDataIn_Index	= 0;	

// ##### ##### ##### ##### ##### dataparts for threads

#define kMaxPartCount		I0
#define kMaxPartDataSize	(9 * 4 * 11 * MB)  // = 396mb = ok on both devices

typedef struct { 
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
	int		iNumThreadBlocks;
	int		iDataSizeBounds;
	float	fEpsilon;
	
	unsigned int*	pDataOut_Atom;
	ulong2*			pDataOut_List;
	float3*			pDataIn_Bounds;
	uint4*			pDataInDebug_State;
} cDataPart;
cDataPart	gDataParts[kMaxPartCount];
	
bool	cDataPart_Prepare		(cDataPart* p,int iPartID,int iPartCount,float e) {
	p->iPartID = iPartID;
	p->fEpsilon = e;
	p->iResultCount = 0;
	
	p->pDataOut_Atom		= 0;
	p->pDataOut_List		= 0;
	p->pDataIn_Bounds		= 0;
	p->pDataInDebug_State	= 0;
	
	// determine the datarange that will be managed by this part/thread
	int iMaxPartLen 	= (I0 + iPartCount - 1)/iPartCount; // round up
	p->iPartElementsFirst	= iPartID*iMaxPartLen; // each element gets its own thread later
	p->iPartElementsLast	= min(p->iPartElementsFirst+iMaxPartLen-1,I0-1); 
	
	//~ // determine the view = the area in the data that has to be scanned, e.g. elements + epsilon border
	p->iPartViewFirst	= 0;	// the view is the data range that has to be examined
	p->iPartViewLast	= I0-1; // inclusive
	#define GET_MIN_0(x)	gpDataIn_Index[INDEXPOS_0(x)]
	#define GET_MAX_0(x)	GET_MIN_0((x+1))
	float fMin		= GET_MIN_0(p->iPartElementsFirst	) - p->fEpsilon; 
	float fMax		= GET_MAX_0(p->iPartElementsLast	) + p->fEpsilon; // max(i)=min(i+1)
	while (p->iPartViewFirst	< I0-1	&& GET_MAX_0(p->iPartViewFirst)	< fMin) ++p->iPartViewFirst;
	while (p->iPartViewLast		> 0		&& GET_MIN_0(p->iPartViewLast)	> fMax) --p->iPartViewLast;
	assert(p->iPartViewFirst 	<= p->iPartElementsFirst);
	assert(p->iPartViewLast		>= p->iPartElementsLast);    
	
	// calc mem usage and positions 
	p->iViewDataSize				= (p->iPartViewLast+1     - p->iPartViewFirst		)	* SX * D * sizeof(float); // in bytes
	p->iElementCount				= (p->iPartElementsLast+1 - p->iPartElementsFirst	)	* SX; // in datapoints
	p->iViewDataOffsetInPoints		=  p->iPartViewFirst									* SX;
	p->iViewDataOffsetInFloats		=  p->iPartViewFirst									* SX * D;
	p->iElementOffsetInPoints		= (p->iPartElementsFirst - p->iPartViewFirst)			* SX;
	p->iNumThreadBlocks				= p->iElementCount / kThreadBlockSize;
	p->iDataSizeBounds				= p->iNumThreadBlocks*2*sizeof(float3);
	
	
	// print a little debug info
	printf("parts=%d [%d]: elem[%d,%d] view[%d,%d] datasize=%dMB\n",iPartCount,iPartID,
		p->iPartElementsFirst,p->iPartElementsLast,
		p->iPartViewFirst,p->iPartViewLast,p->iViewDataSize/MB);
	
	//~ // check if this part fits in vram
	if (p->iViewDataSize > kMaxPartDataSize) return false;
	return true;
}

/// malloc might not be threadsafe
void	cDataPart_Alloc			(cDataPart* p) {
	p->pDataIn_Bounds		= (float3*)			malloc(p->iDataSizeBounds); 
	p->pDataOut_Atom		= (unsigned int*)	malloc(DATASIZE_OUT_ATOM);
	#ifdef ENABLE_GPU_WRITEOUTLIST
	p->pDataOut_List		= (ulong2*)			malloc(DATASIZE_OUT_LIST);
	#endif
	p->pDataInDebug_State	= (uint4*)			malloc(DATASIZE_IN_STATE);
}

void	cDataPart_Free			(cDataPart* p) {
	#define FREE_AND_ZERO(x) if (x) free(x); x = 0;
	FREE_AND_ZERO(p->pDataIn_Bounds);
	FREE_AND_ZERO(p->pDataOut_Atom);
	FREE_AND_ZERO(p->pDataOut_List);
	FREE_AND_ZERO(p->pDataInDebug_State);
}


////////////////////////////////////////////////////////////////////////////////
// GPU thread
////////////////////////////////////////////////////////////////////////////////

static CUT_THREADPROC gpuThread (int* piPartID) {
	int iPartID = *piPartID;
	cDataPart* p = &gDataParts[iPartID];
	int iGPUId = iPartID % kGPU_COUNT;
	CUDA_SAFE_CALL(cudaSetDevice(iGPUId));
	
	printf("run:%d,%d (...,e=%f,%d,%d) off=%d,size=%d,el=%d\n",
		iPartID,iGPUId,
		p->fEpsilon,p->iElementOffsetInPoints,p->iViewDataOffsetInPoints,
		p->iViewDataOffsetInFloats,p->iViewDataSize,p->iElementCount);
	
	#define HANDLE_ERROR(x) myLastErr = cudaGetLastError(); if (myLastErr != 0) printf("%s : %d(%s)\n",x,(int)myLastErr,(myLastErr != 0) ? cudaGetErrorString(myLastErr) : "ok");
	
	cudaError_t myLastErr; 
	
	uint4* pDataInDebug_State = p->pDataInDebug_State;
	
	// calculate bounds of individual threadblocks
	// this part is only done once at startup, so performance doesn't really matter much
	if (1) {
		float* pElements = &gpDataIn_Raw[p->iViewDataOffsetInFloats + p->iElementOffsetInPoints*D];
		float e = p->fEpsilon;
		float3 vMin,vMax,vCur;
		
		// foreach threadblock...
		for (int iBlockIdx=0;iBlockIdx < p->iNumThreadBlocks;++iBlockIdx) {
			// calculate the bounds for the wrap, for the first 3 dimensions
			{ for (int d=0;d<kThreadBlockSize;++d) { // compiler should loop-unroll
				vCur.x = pElements[(iBlockIdx * kThreadBlockSize + d)*D + 0];
				vCur.y = pElements[(iBlockIdx * kThreadBlockSize + d)*D + 1];
				vCur.z = pElements[(iBlockIdx * kThreadBlockSize + d)*D + 2];
				if (d > 0) { 
					vMin.x = min(vMin.x,vCur.x);
					vMin.y = min(vMin.y,vCur.y);
					vMin.z = min(vMin.z,vCur.z);
					vMax.x = max(vMax.x,vCur.x);
					vMax.y = max(vMax.y,vCur.y);
					vMax.z = max(vMax.z,vCur.z);
				} else {
					vMin = vCur;
					vMax = vCur;
				}
			}}
			
			// add epsilon to the edges of the bounds
			vMin.x = vMin.x - e;
			vMin.y = vMin.y - e;
			vMin.z = vMin.z - e;
			vMax.x = vMax.x + e;
			vMax.y = vMax.y + e;
			vMax.z = vMax.z + e;
			
			p->pDataIn_Bounds[iBlockIdx*2+0] = vMin;
			p->pDataIn_Bounds[iBlockIdx*2+1] = vMax;
		}
	}
	
	
	// alloc and init device memory (vram)
	float*			pDataInD_Raw	= NULL;
	float*			pDataInD_Index	= NULL; 
	uint4*			pDataInD_State	= NULL;
	float3*			pDataInD_Bounds	= NULL;
	unsigned int*	pDataOutD_Atom	= NULL;
	ulong2*			pDataOutD_List	= NULL;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataOutD_Atom,	DATASIZE_OUT_ATOM	)); assert(pDataOutD_Atom);
	#ifdef ENABLE_GPU_WRITEOUTLIST
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataOutD_List,	DATASIZE_OUT_LIST	)); assert(pDataOutD_List);
	#endif
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataInD_Index,	DATASIZE_IN_INDEX	)); assert(pDataInD_Index);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataInD_State,	DATASIZE_IN_STATE	)); assert(pDataInD_State);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataInD_Bounds,	p->iDataSizeBounds	)); assert(pDataInD_Bounds);
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pDataInD_Raw,		p->iViewDataSize	)); assert(pDataInD_Raw);  // biggest last
	CUDA_SAFE_CALL( cudaMemcpy(pDataInD_Raw,	&gpDataIn_Raw[p->iViewDataOffsetInFloats],	p->iViewDataSize,	cudaMemcpyHostToDevice	)); 
	CUDA_SAFE_CALL( cudaMemcpy(pDataInD_Index,	gpDataIn_Index,		DATASIZE_IN_INDEX,							cudaMemcpyHostToDevice	)); 
	CUDA_SAFE_CALL( cudaMemcpy(pDataInD_Bounds,	p->pDataIn_Bounds,	p->iDataSizeBounds,							cudaMemcpyHostToDevice	)); 
	CUDA_SAFE_CALL( cudaMemset(pDataInD_State,	0,				DATASIZE_IN_STATE 	)); 
	CUDA_SAFE_CALL( cudaMemset(pDataOutD_Atom,	0,				DATASIZE_OUT_ATOM 	)); HANDLE_ERROR("memset")
	
	// prepare device environment
	dim3  block_size;
	dim3  grid_size;
	grid_size.x		= p->iNumThreadBlocks / GRIDHEIGHT;
	grid_size.y		= GRIDHEIGHT;
	grid_size.z		= 1;
	block_size.x	= kThreadBlockSize;
	block_size.y	= 1; 
	block_size.z	= 1;
	unsigned int mem_shared = 0; // this is for dynamic alloc of shared mem, we alloc statically
	unsigned int iMyAtoms[2];
	assert(DATASIZE_OUT_ATOM == sizeof(iMyAtoms));
	iMyAtoms[0] = 0;
	iMyAtoms[1] = 0;
	
	#define kMaxGridSize (63*1024)
	assert(grid_size.x <= kMaxGridSize && "grid_size.x larger than supported (cudaGetDeviceProperties:maxGridSize: 63k currently)"); 
	
	
	printf("run_:%d,%d\n",iPartID,iGPUId);
	bool bPrintFirst = true;
	
	// precalc params as much as possible, i think they are only needed once for the whole program, not for every thread
	do {
		// run kernel
		
		mykernel<<<grid_size, block_size, mem_shared >>>(
			pDataInD_Raw,
			pDataInD_Index,
			pDataInD_State,
			pDataInD_Bounds,
			pDataOutD_Atom,
			pDataOutD_List,
			p->fEpsilon,
			p->fEpsilon*p->fEpsilon,
			p->iElementOffsetInPoints,
			p->iViewDataOffsetInPoints);
		//~ CUT_SAFE_CALL( cudaThreadSynchronize()); // kernel start is async, so wait for finish here
		//~ HANDLE_ERROR("threadsync")
		
		// copy result from device to host
		//~ CUDA_SAFE_CALL( cudaMemcpy(pDataInDebug_State,	pDataInD_State, DATASIZE_IN_STATE,	cudaMemcpyDeviceToHost	)); HANDLE_ERROR("cudaMemcpy state")
		CUDA_SAFE_CALL( cudaMemcpy(iMyAtoms, 			pDataOutD_Atom, DATASIZE_OUT_ATOM,	cudaMemcpyDeviceToHost	)); HANDLE_ERROR("cudaMemcpy atom") // get counter
		CUDA_SAFE_CALL( cudaMemset(pDataOutD_Atom,		0,	 DATASIZE_OUT_ATOM							)); HANDLE_ERROR("cudaMemset atom") // clear counter
		int iNumResults = min(kMaxResults,iMyAtoms[0]);  
		//~ int iNumResults = min(kMaxResults,p->pDataOut_Atom[0]); 
		//~ printf("datapart %d : atom[0]=%d atom[1]=%d iNumResults=%d kMaxResults=%d\n",p->iPartID,p->pDataOut_Atom[0],p->pDataOut_Atom[1],iNumResults,(int)kMaxResults);
		//~ printf("datapart %d : atom[0]=%d atom[1]=%d iNumResults=%d kMaxResults=%d\n",p->iPartID,iMyAtoms[0],iMyAtoms[1],iNumResults,(int)kMaxResults);
		
		#ifdef ENABLE_GPU_WRITEOUTLIST
		int iListSize = iNumResults * sizeof(ulong2);
		CUDA_SAFE_CALL( cudaMemcpy(p->pDataOut_List,	pDataOutD_List, iListSize, 			cudaMemcpyDeviceToHost	)); // get list
		#endif
		
		//~ printf("datapart %d : receive results from device\n",p->iPartID);
		p->iResultCount += iNumResults;  
		
		// todo : do something with the data here, e.g. printf, or write to file ?
		
		//~ int b =  grid_size.x*grid_size.y;
		//~ for (int i=0;i<grid_size.x*grid_size.y+4;++i) if (pDataInDebug_State[i].x == kStateEndValue) --b;
		//~ printf("[%d,%d,%d,%d]:%d\n",iPartID,a,iNumResults,(int)iMyAtoms[1],b);
		
		if (iMyAtoms[1] == 0 && iNumResults == 0) break; // detect end
		//~ if (--a == 0) break;
		if (bPrintFirst) { bPrintFirst = false; printf("run_firstdone:%d,%d\n",iPartID,iGPUId); }
		break;
	} while (1); 
	
	printf("datapart %d : atom[0]=%d atom[1]=%d iNumResults=%d kMaxResults=%d\n",p->iPartID,iMyAtoms[0],iMyAtoms[1],p->iResultCount,(int)kMaxResults);
		
	
	// release device memory 
	CUDA_SAFE_CALL(cudaFree(pDataInD_Raw));
	CUDA_SAFE_CALL(cudaFree(pDataInD_Index));
	CUDA_SAFE_CALL(cudaFree(pDataInD_State));
	CUDA_SAFE_CALL(cudaFree(pDataOutD_Atom));
	//~ CUDA_SAFE_CALL(cudaFree(pDataOutD_List));
	
	
    CUT_THREADEND; 
}








// ##### ##### ##### ##### ##### main


int main( int argc, char** argv) {
	#ifdef ENABLE_MEASUREMENT_CPU_IDX
	#ifdef ENABLE_MEASUREMENT_CPU_NEST
	printf("don't enable ENABLE_MEASUREMENT_CPU_IDX and _NEST at the same time !\n");
	exit(0);
	#endif
	#endif
	
    CUT_DEVICE_INIT();
	
	const char* sCPU_CalcType = "none";
	const char* sGPU_CalcType = "none";
	
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
	
	#ifdef ENABLE_MEASUREMENT_GPU
    if (1) {
        int threadIds[kGPU_COUNT];
		int iPartID,iGPUIndex;
		int iPartCount = 0;
        CUTThread*	threads = (CUTThread *)malloc(sizeof(CUTThread) * kGPU_COUNT);
		RobStartTimer();printf("starting gpu...\n");
		
		#ifdef GPU_NESTED_LOOP
		sGPU_CalcType  = "nest";
		#else
		sGPU_CalcType  = "idx";
		#endif
		
		// determine how the data will be subdivided
		bool bOK = false;
		for (iPartCount=kGPU_COUNT;iPartCount <= kMaxPartCount;iPartCount*=2) {
			bOK = true;
			for (iPartID=0;iPartID<iPartCount;++iPartID) 
				if (!cDataPart_Prepare(&gDataParts[iPartID],iPartID,iPartCount,fEpsilon)) { bOK = false; break; }
			if (bOK) break;
		}
		if (!bOK) { printf("failed to find a partitioning that fits in vram, try lowering epsilon, or increase I0\n"); exit(1); }
		for (iPartID=0;iPartID<iPartCount;++iPartID) cDataPart_Alloc(&gDataParts[iPartID]);
		//~ for (int j=0;j<I0/2;++j) printf("%d:%+0.1f  %d:%+0.1f\n",j,GET_MIN_0(j),j+I0/2,GET_MIN_0(j+I0/2));
		
		#ifdef GPU_NESTED_LOOP
		assert(iPartCount == 1 && "nested loop on gpu must currently be on a single GPU");
		#endif
		
		// start threads
		for (iPartID=0;iPartID<iPartCount;++iPartID) {
			iGPUIndex = iPartID % kGPU_COUNT;
            threadIds[iGPUIndex] = iPartID;
			gpuThread(&threadIds[iGPUIndex]);
            //~ threads[iGPUIndex] = cutStartThread((CUT_THREADROUTINE)gpuThread, (void *)&threadIds[iGPUIndex]);
			//~ if (iGPUIndex == kGPU_COUNT-1) cutWaitForThreads(threads, kGPU_COUNT); // Wait for all the threads to finish.
		}
		
		iResultsGPU = 0;
		for (iPartID=0;iPartID<iPartCount;++iPartID) iResultsGPU += gDataParts[iPartID].iResultCount;
		
		for (iPartID=0;iPartID<iPartCount;++iPartID) cDataPart_Free(&gDataParts[iPartID]);
        free(threads);
		printf("all threads finished\n");
		fTimeGPU = RobStopTimer();	printf("%4.2f sec : gpu\n",fTimeGPU);
    }
	#endif
	
	
	#ifdef ENABLE_MEASUREMENT_CPU_NEST
	if (1) { // NestedLoop
		RobStartTimer();			printf("check : NestedLoop...\n");
		iResultsCPU = NestedLoop(gpDataIn_Raw,fSqEpsilon); 
		fTimeCPU = RobStopTimer();	printf("%4.2f sec : check : NestedLoop\n",fTimeCPU);
		sCPU_CalcType  = "nest";
	}
	#endif
	
	#ifdef ENABLE_MEASUREMENT_CPU_IDX
	if (1) { // Idx_CPU
		RobStartTimer();		printf("check : with index on cpu...\n");
		iResultsCPU = Idx_CPU(gpDataIn_Raw,gpDataIn_Index,fEpsilon,fSqEpsilon); 
		fTimeCPU = RobStopTimer(); printf("%4.2f sec : check : with index on cpu\n",fTimeCPU);
		sCPU_CalcType  = "idx";
	}
	#endif
		
	
	// time-text
	char myTimeText[256] = "";
	time_t mytime;
	time(&mytime);
	strftime(myTimeText,255,"%Y.%m.%d_%H.%M.%S",localtime(&mytime));
	
	
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
			
			#ifdef DATA_PURE_RANDOM
			const char* szDataType = "unirand";
			#else
				#ifdef DATA_CLUST_RANDOM
				const char* szDataType = "clustrand";
				#else
				const char* szDataType = "file";
				#endif
			#endif
			
			float fTolerance = 0.01; // 1 % tolerance due to less exact 
			float fRelError = float(abs(iResultsCPU - iResultsGPU)) / float( (iResultsCPU > 0) ? iResultsCPU : 1 );
			bool bOK = fRelError < fTolerance; // error relative to "correct" cpu results...
			float fErrorPercent = ceil(fRelError * 1000.0) * 0.1;
			#define REPORT "%s gpu/cpu=%8.1f %s err<=%0.1f%% N=%d size=%dMB %s I0=%d i3:%d,%d e=%0.1f d=%d tgpu(%s)=%0.1fs=%0.1fm=%0.1fh tcpu(%s)=%0.1f res=%d/%d\n", \
				myTimeText,fTimeGPU/fTimeCPU,bOK?" ok ":"MISS", fErrorPercent, \
				N,DATASIZE_IN_RAW/MB,szDataType,I0,iGPU_IDX3,iCPU_IDX3, \
				(float)TEST_EPSILON,(int)D,sGPU_CalcType,fTimeGPU,fTimeGPU/60,fTimeGPU/3600,sCPU_CalcType,fTimeCPU, iResultsCPU,iResultsGPU
			printf(			REPORT );
			fprintf( fp,	REPORT );
			fclose(fp);  
		}
	}
	
	
	if (0) { // write sample data
		char mySampleDataPath[256] = "";
		sprintf(mySampleDataPath,"sampledata/sampledata_%s_%dMB.txt",myTimeText,(int)DATASIZE_IN_RAW/MB);
		FILE* fp = fopen(mySampleDataPath,"a");
		if (fp) {
			fprintf(fp,"START e=%f rcpu=%d tcpu=%f\n",fEpsilon,iResultsCPU,fTimeCPU);
			for (int i=0;i<N;++i) {
				for (int d=0;d<D;++d) fprintf(fp,"%f,",gpDataIn_Raw[i*D + d]);
				fprintf(fp,"\n");
			}
			fclose(fp);
		}
	}
	
	// release memory
    free(gpDataIn_Raw);
    free(gpDataIn_Index);

    CUT_EXIT(argc, argv);
	return 0;
}


