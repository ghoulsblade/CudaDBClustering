#include "robintlist.cu"

#define DBSCAN_PARAM_EPSILON	(0.010)	// TODO
#define DBSCAN_PARAM_SQEPSILON	(DBSCAN_PARAM_EPSILON * DBSCAN_PARAM_EPSILON)
#define DBSCAN_PARAM_MINPTS		(4)		// TODO   // currently minpts is at least 1 because self is also counted (dist=0)
#define DBSCAN_INDEX_OUTLIER	(-1)
#define DBSCAN_INDEX_INIT		(-2)
#define GRIDHEIGHT 			1 // TODO 
#define	kThreadBlockSize	(64) // x*64:,64,128,192,256 (see cuda-guide page 65)


#define BLOCKIDX	(blockIdx.x * GRIDHEIGHT + blockIdx.y)
#define MYID		(BLOCKIDX * blockDim.x + threadIdx.x)  

/// kernel code : this gets executed on the GPU
__global__ static void dbscan_kernel_main (
	float*			pIn_Points,
	int*			pOut_ParentIndex) {
	
	int iMyLocalID = MYID;
	
	// read own coordinates
	float element[D];
	{for (int d=0;d<D;++d) element[d] = pIn_Points[iMyLocalID*D + d];} // compiler should loop-unroll
	
	int iNumPoints = 0;
	int iParentIndex = iMyLocalID;
	float a;
	float fSqDist;
	for (int n=0;n<N;++n) {
		fSqDist = 0.0;
		{for (int d=0;d<D;++d) { a = pIn_Points[n*D + d] - element[d]; fSqDist += a*a; }} // compiler should loop-unroll
		if (fSqDist <= DBSCAN_PARAM_SQEPSILON) { iNumPoints += 1; iParentIndex = min(iParentIndex,n); }
	}
	
	if (iNumPoints < DBSCAN_PARAM_MINPTS) iParentIndex = DBSCAN_INDEX_OUTLIER;
	pOut_ParentIndex[iMyLocalID] = iParentIndex;
}

/// kernel code : this gets executed on the GPU
__global__ static void dbscan_kernel_cluster_chains (
	float*			pIn_Points,
	int*			pOut_ParentIndex) {
	
	int iMyLocalID = MYID;
	
	// read own coordinates
	float element[D];
	{for (int d=0;d<D;++d) element[d] = pIn_Points[iMyLocalID*D + d];} // compiler should loop-unroll
	
	int iParentIndex = pOut_ParentIndex[iMyLocalID];
	int iOtherParentIndex;
	float a;
	float fSqDist;
	for (int n=0;n<N;++n) {
		fSqDist = 0.0;
		{for (int d=0;d<D;++d) { a = pIn_Points[n*D + d] - element[d]; fSqDist += a*a; }} // compiler should loop-unroll
		if (fSqDist <= DBSCAN_PARAM_SQEPSILON && iParentIndex != DBSCAN_INDEX_OUTLIER) { 
			iOtherParentIndex = pOut_ParentIndex[n];
			if (iOtherParentIndex < iParentIndex && iOtherParentIndex != DBSCAN_INDEX_OUTLIER)
				atomicMin((unsigned int*)&pOut_ParentIndex[iParentIndex],pOut_ParentIndex[iOtherParentIndex]);
		}
	}
}

// in this iteration we can look up if our neighbors are core-points or not
// also only a fraction of the original data has to be processed
__global__ static void dbscan_kernel_rescan (
	float*			pIn_Points,
	int*			pIn_RescanIndex,
	int*			pOut_ParentIndex) {
		
	int iMyLocalID = pIn_RescanIndex[MYID];

	// read own coordinates
	float element[D];
	{for (int d=0;d<D;++d) element[d] = pIn_Points[iMyLocalID*D + d];} // compiler should loop-unroll
	
	int iParentIndex = iMyLocalID;
	float a;
	float fSqDist;
	for (int n=0;n<N;++n) if (pOut_ParentIndex[n] != DBSCAN_INDEX_OUTLIER) { // only consider core neighbors this time
		fSqDist = 0.0;
		{for (int d=0;d<D;++d) { a = pIn_Points[n*D + d] - element[d]; fSqDist += a*a; }} // compiler should loop-unroll
		if (fSqDist <= DBSCAN_PARAM_SQEPSILON) iParentIndex = min(iParentIndex,n);
	}
	
	pOut_ParentIndex[iMyLocalID] = iParentIndex;
}

typedef struct {
	float			pPoints				[N*D];
	int				pParentIndex		[N]; ///< also used as clusterindex
} DBScanData;

void	dbscan_gpu	(DBScanData* p) {
	int	iParent,iGrandParent,n;
	cudaError_t myLastErr; 
	
	//~ float*			pPoints				= p->pPoints;
	int*			pParentIndex		= p->pParentIndex;
	
	// allocate and init gpu buffers
	float*			pGPUIn_Data = 0;
	int*			pGPUOut_ParentIndex = 0;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_Data,			sizeof(p->pPoints)		));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUOut_ParentIndex,	sizeof(p->pParentIndex)	));
	
	// copy data from ram to vram
	CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_Data,	p->pPoints,	sizeof(p->pPoints),	cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy init")
	
	// grid_size, block_size, mem_shared
	dim3  grid_size;
	dim3  block_size;
	unsigned int mem_shared = 0; // this is for dynamic alloc of shared mem, we alloc statically
	grid_size.x		= N / kThreadBlockSize / GRIDHEIGHT;  // TODO : make sure  N is a multiple of kThreadBlockSize
	grid_size.y		= GRIDHEIGHT;
	grid_size.z		= 1;
	block_size.x	= kThreadBlockSize;
	block_size.y	= 1; 
	block_size.z	= 1;
	
	// call kernel
	dbscan_kernel_main<<< grid_size, block_size, mem_shared >>>(
		pGPUIn_Data,
		pGPUOut_ParentIndex);	
	
	// read back results
	CUDA_SAFE_CALL( cudaMemcpy(pParentIndex, pGPUOut_ParentIndex, sizeof(p->pParentIndex), cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy pParentIndex")
	
	// prepare list for rescanning if neccessary
	RobIntList	oList_RescanBorder;
	RobIntList_Init(&oList_RescanBorder);
	
	// detect parent-borderpoint that need to be rescanned
	for (n=0;n<N;++n) {
		iParent = pParentIndex[n];
		if (DBSCAN_INDEX_OUTLIER == iParent) continue; // n is outlier itself
		if (DBSCAN_INDEX_OUTLIER != pParentIndex[iParent]) continue; // n has valid parent, no need to rescan
			
		// iParent is an outlier, so n has to be rescanned to find a core-point-parent near it
		// if no other corepoint near it can be found n is a single-corepoint-cluster
		RobIntList_Push(&oList_RescanBorder,n);
	}
	
	// rescan border points which have wrong parents (oList_RescanBorder)
	if (oList_RescanBorder.iSize > 0) {
		printf("dbscan_gpu : rescan list %d/%d : %f\n",(int)oList_RescanBorder.iSize,(int)N,(float)oList_RescanBorder.iSize / (float)N);
		// the number of kernel-threads must be a multiple of kThreadBlockSize, pad by repeating last point
		int iPadValue	= oList_RescanBorder.pData[oList_RescanBorder.iSize-1];
		while ((oList_RescanBorder.iSize % (kThreadBlockSize * GRIDHEIGHT)) != 0) RobIntList_Push(&oList_RescanBorder,iPadValue);
		
		// copy oList_RescanBorder to gpu memory (max size = N*sizeof(int)
		#define DBSCAN_RESCANLIST_SIZE (oList_RescanBorder.iSize*sizeof(int))
		int*			pGPUIn_RescanList = 0;
		CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_RescanList, DBSCAN_RESCANLIST_SIZE));
		
		// copy data from ram to vram
		CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_RescanList,	oList_RescanBorder.pData,	DBSCAN_RESCANLIST_SIZE,	cudaMemcpyHostToDevice	)); HANDLE_ERROR("cudaMemcpy rescanlist")
	
		// grid_size, block_size, mem_shared
		dim3  grid_size;
		dim3  block_size;
		unsigned int mem_shared = 0; // this is for dynamic alloc of shared mem, we alloc statically
		grid_size.x		= oList_RescanBorder.iSize / kThreadBlockSize / GRIDHEIGHT; 
		grid_size.y		= GRIDHEIGHT;
		grid_size.z		= 1;
		block_size.x	= kThreadBlockSize;
		block_size.y	= 1; 
		block_size.z	= 1;
		
		// call kernel
		dbscan_kernel_rescan<<< grid_size, block_size, mem_shared >>>(
			pGPUIn_Data,
			pGPUIn_RescanList,
			pGPUOut_ParentIndex);	
		
		// read back results
		CUDA_SAFE_CALL( cudaMemcpy(pParentIndex, pGPUOut_ParentIndex, sizeof(p->pParentIndex), cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy rescan pParentIndex")
		
		CUDA_SAFE_CALL(cudaFree(pGPUIn_RescanList));
	}
	
	RobIntList_Destroy(&oList_RescanBorder);
	
	// trace parents to root of clusters/chains
	// no recursion needed because of iParent <= n
	for (n=0;n<N;++n) {
		iParent = pParentIndex[n];
		if (iParent == DBSCAN_INDEX_OUTLIER) continue;
		assert(iParent <= n); // TODO : disable this after testing // due to the way it is calculated : min(...)
		if (iParent == n) continue; // already min
		iGrandParent = pParentIndex[iParent];
		assert(iGrandParent != DBSCAN_INDEX_OUTLIER); // TODO : disable this after testing // prevented by rescan
		pParentIndex[n] = iGrandParent;
	}
	
	
	// count current chains 
	/*
	
		RobIntList	oList_ClusterIDs;
		RobIntList_Init(&oList_ClusterIDs);
		for (int n=0;n<N;++n) {
			int iClusterID = gDBScanData.pParentIndex[n];
			bool bFound = false;
			for (int i=0;i<oList_ClusterIDs.iSize;++i) 
				if (oList_ClusterIDs.pData[i] == iClusterID) { bFound = true; break; }
			if (!bFound) RobIntList_Push(&oList_ClusterIDs,iClusterID);
		}
		printf("dbscan_gpu_count_clusters : %d\n",oList_ClusterIDs.iSize);
	*/
	
	// now we have connected chains, but a single cluster can consist of more thena one chain, 
	// so we have to scan all points again to see which chains are adjacted.  the price we have to pay for parallelism
	// TODO : still doesn work 
	if (0) {
		#define DBSCAN_RESCANLIST_SIZE (oList_RescanBorder.iSize*sizeof(int))
		int*			pGPUIn_RescanList = 0;
		CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_RescanList, DBSCAN_RESCANLIST_SIZE));
		
		
		
		
	
		// copy data from ram to vram
		CUDA_SAFE_CALL( cudaMemcpy(pGPUOut_ParentIndex,	pParentIndex,	sizeof(p->pParentIndex),	cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy chainclust pParentIndex up")
		
		// call kernel
		dbscan_kernel_cluster_chains<<< grid_size, block_size, mem_shared >>>(
			pGPUIn_Data,
			pGPUOut_ParentIndex);	
		
		// read back results
		CUDA_SAFE_CALL( cudaMemcpy(pParentIndex, pGPUOut_ParentIndex, sizeof(p->pParentIndex), cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy chainclust pParentIndex down")
	}
	
	
	// trace parents to root of clusters/chains
	// no recursion needed because of iParent <= n
	for (n=0;n<N;++n) {
		iParent = pParentIndex[n];
		if (iParent == DBSCAN_INDEX_OUTLIER) continue;
		assert(iParent <= n); // TODO : disable this after testing // due to the way it is calculated : min(...)
		if (iParent == n) continue; // already min
		iGrandParent = pParentIndex[iParent];
		assert(iGrandParent != DBSCAN_INDEX_OUTLIER); // TODO : disable this after testing // prevented by rescan
		pParentIndex[n] = iGrandParent;
	}
	
	
	// release GPU-memory
	CUDA_SAFE_CALL(cudaFree(pGPUIn_Data));
	CUDA_SAFE_CALL(cudaFree(pGPUOut_ParentIndex));
}



