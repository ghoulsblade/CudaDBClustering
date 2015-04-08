
typedef struct {
	float			pPoints			[N*D];
	float			pIndex			[INDEX_NUM_FLOATS];
	float3			pBounds			[DBSCAN_NUM_BOUNDS*2]; // min,max for each threadblock
	int				pSeedList		[DBSCAN_NUM_SEEDS];
	int				pClusterIDs		[N];
	int				pConnections	[DBSCAN_NUM_SEEDS * DBSCAN_NUM_SEEDS];
} DBScanData;


// clusterid range : 
// atomicMin assigned_and_scanned(128) noise assigned_but_not_scanned(128) init oldcluster(big,won't be connected)
// atomicMax init assigned_but_not_scanned(128) noise assigned_and_scanned(128) oldcluster(big,won't be connected)


/*
int atomicMin(int* address, int val);
int atomicMax(int* address, int val);
int   atomicExch(int*   address, int   val);
uint  atomicExch(uint*  address, uint  val);
float atomicExch(float* address, float val);

*/

//~ #define EMU_CHECKBOUNDS(name,val,maxcount) if ((val) < 0 || (val) >= (maxcount)) { printf("OUT OF BOUNDS %s %s %d\n",name,#val,(int)(val)); exit(0); }
#define EMU_CHECKBOUNDS(name,val,maxcount)
#define DEBUGCHECK_START()  // for marking lines
#define DEBUGCHECK_END()  // for marking lines

#define ENABLE_STEP_PRINT 1

//~ #define PROFILE_TIME_ENABLE
#ifdef PROFILE_TIME_ENABLE
	#define PROFILE_TIME_SECTION_START() ProfileTimerStartSection()
	#define PROFILE_TIME_SECTION_STEP(name) float name = ProfileTimerStartSection()
	#define PROFILE_TIME_SECTION_SUM(name) name += ProfileTimerStartSection()
#else
	#define PROFILE_TIME_SECTION_START()
	#define PROFILE_TIME_SECTION_STEP(name)
	#define PROFILE_TIME_SECTION_SUM(name) 
#endif


bool DBScan_CheckConnection			(int* pConnections,int i,int j) { 
	EMU_CHECKBOUNDS("DBScan_CheckConnection",i,DBSCAN_NUM_SEEDS)
	EMU_CHECKBOUNDS("DBScan_CheckConnection",j,DBSCAN_NUM_SEEDS)
	return pConnections[i*DBSCAN_NUM_SEEDS + j] || pConnections[j*DBSCAN_NUM_SEEDS + i]; 
}

void DBScan_SpreadConnection		(int* iFinishedClusterIDAssignment,int* pConnections);
void DBScan_SpreadConnection_Aux	(int* iFinishedClusterIDAssignment,int* pConnections,int i);

void PrintConnectionMatrix (const char* szHeader,int* iFinishedClusterIDAssignment,int* pConnections);


// function used by kernel, inlined automatically
__device__ void DBScan_Connect (int* pOut_Connections,int a,int b) {
	//~ EMU_CHECKBOUNDS("DBScan_Connect",a,DBSCAN_NUM_SEEDS)
	//~ EMU_CHECKBOUNDS("DBScan_Connect",b,DBSCAN_NUM_SEEDS)
	//~ pOut_Connections[a*DBSCAN_NUM_SEEDS + b] = 1;   // min(a,b) max(a,b)
	//if (a >= 0 && a < DBSCAN_NUM_SEEDS && b >= 0 && b < DBSCAN_NUM_SEEDS) // TODO : this check should not be neccessary
	atomicMax(&pOut_Connections[a*DBSCAN_NUM_SEEDS + b],1);
} // multiple write : order undefined, but one is guaranteed to succeed


// function used by kernel, inlined automatically
__device__ void DBScan_SpreadToNeighbour (int* pOut_Connections,int* pOut_ClusterID,int iMyUnscannedClusterID,int iMyScannedClusterID,int n) {
	/*
	EMU_CHECKBOUNDS("DBScan_SpreadToNeighbour",iMyUnscannedClusterID,DBSCAN_NUM_SEEDS)
	EMU_CHECKBOUNDS("DBScan_SpreadToNeighbour",n,N)
	*/
	int iOldClusterID = atomicMax(&pOut_ClusterID[n],iMyUnscannedClusterID); // TODO : minmax shouldn't be neccessary
	//~ int iOldClusterID = atomicMax(&pOut_ClusterID[max(0,min(N-1,N-1))],iMyUnscannedClusterID); // TODO : minmax shouldn't be neccessary
	if (iOldClusterID >= DBSCAN_CLUSTER_ID_SCANNED_FIRST) iOldClusterID = iOldClusterID - DBSCAN_CLUSTER_ID_SCANNED_FIRST;
		
	if (iOldClusterID >= DBSCAN_CLUSTER_ID_UNSCANNED_FIRST &&
		iOldClusterID <= DBSCAN_CLUSTER_ID_UNSCANNED_LAST &&
		iOldClusterID != iMyUnscannedClusterID) atomicMax(&pOut_Connections[iOldClusterID*DBSCAN_NUM_SEEDS + iMyUnscannedClusterID],1); // DBScan_Connect
	
	/*
	
	// do nothin if iOldClusterID corresponds to own cluster already (+- scan)
	// connect if scanned or unscanned 
	if (iOldClusterID >= DBSCAN_CLUSTER_ID_UNSCANNED_FIRST &&
		iOldClusterID <= DBSCAN_CLUSTER_ID_UNSCANNED_LAST &&
		iOldClusterID != iMyUnscannedClusterID) DBScan_Connect(pOut_Connections,iOldClusterID-DBSCAN_CLUSTER_ID_UNSCANNED_FIRST,iMyUnscannedClusterID);
	
	if (iOldClusterID >= DBSCAN_CLUSTER_ID_SCANNED_FIRST &&
		iOldClusterID <= DBSCAN_CLUSTER_ID_SCANNED_LAST &&
		iOldClusterID != iMyScannedClusterID) DBScan_Connect(pOut_Connections,iOldClusterID-DBSCAN_CLUSTER_ID_SCANNED_FIRST,iMyUnscannedClusterID);
	*/
	
}

#ifdef __DEVICE_EMULATION__
	#define ARRAY_INDEX(idx,arrsize) max(0,idx)
	//~ #define ARRAY_INDEX(idx,arrsize) MyCheckIndex(idx,arrsize,__LINE__)
#else
	#define ARRAY_INDEX(idx,arrsize) MyGPUCheckIndex(idx,arrsize,__LINE__,pOut_Debug)
	//~ #define ARRAY_INDEX(idx,arrsize) (idx)
	//~ #define ARRAY_INDEX(idx,arrsize) (max(0,min((arrsize)-1,idx)))
#endif

__device__ int MyGPUCheckIndex	(int idx,int arrsize,int iLINE,int* pOut_Debug) {
	//~ if (idx < 0 || idx >= arrsize) pOut_Debug[MYID] = iLINE;
	//~ return max(0,min((arrsize)-1,idx));
	return max(0,idx);
	//~ return idx;
}

int	MyCheckIndex (int idx,int arrsize,int iLINE) {
	if (idx < 0 || idx >= arrsize) printf("mycheckindex(%d,%d) failed in line %d\n",idx,arrsize,iLINE);
	return idx;
}

/// kernel code : this gets executed on the GPU
__global__ static void dbscan_kernel_main (
	float*			pIn_Points,			// N*D
	float*			pIn_Index,			// INDEX_NUM_FLOATS
	float3*			pIn_Bounds,			// DBSCAN_NUM_BOUNDS*2
	int*			pIn_SeedList,		// DBSCAN_NUM_SEEDS
	int*			pOut_ClusterID,		// N
	int*			pOut_Connections,	// DBSCAN_NUM_SEEDS * DBSCAN_NUM_SEEDS
	int				iNumberOfPoints,
	int*			pOut_Debug			// N
	) {
	
	EMU_CHECKBOUNDS("DBScan_Kernel",MYID,DBSCAN_NUM_SEEDS)
	int iPointID = pIn_SeedList[ARRAY_INDEX(MYID,DBSCAN_NUM_SEEDS)]; // the index of the "current" point/seed
		
	__shared__ float	pfSharedReadCache[D];
		
	// read thread-block bounds
	__shared__ float3 vMin;
	__shared__ float3 vMax;
	if (threadIdx.x == 0) vMin = pIn_Bounds[BLOCKIDX*2 + 0];
	if (threadIdx.x == 1) vMax = pIn_Bounds[BLOCKIDX*2 + 1];
		
	// original code
	//~ if (iPointID >= 0) {
		EMU_CHECKBOUNDS("DBScan_Kernel",iPointID,N)
		int iMyUnscannedClusterID;
		if (iPointID >= 0)
				iMyUnscannedClusterID = max(0,min(DBSCAN_NUM_SEEDS-1,pOut_ClusterID[ARRAY_INDEX(iPointID,N)])); // will be in the "assigned_but_not_scanned" range
		else	iMyUnscannedClusterID = 0;
		int iMyScannedClusterID		= iMyUnscannedClusterID + DBSCAN_CLUSTER_ADD_SCAN;
		EMU_CHECKBOUNDS("DBScan_Kernel",iMyUnscannedClusterID,DBSCAN_NUM_SEEDS)
		#define NEIGHBOUR_COUNT (DBSCAN_PARAM_MINPTS-1)
		#define CON_SIZE (DBSCAN_NUM_SEEDS*DBSCAN_NUM_SEEDS)
		int iNeighbours[NEIGHBOUR_COUNT];
		int iNumPoints = 0;
		int n;
		
		// read own coordinates
		float element[D];
		float fSqDist,a;
		int d;
		for (d=0;d<D;++d) element[d] = pIn_Points[ARRAY_INDEX(DATAPOINT_IDX(max(0,iPointID),d),N*D)]; // compiler should loop-unroll
		
		#define K_I_0(a) (pIn_Index[INDEXPOS_0(				((int)x)+(a))])
		#define K_I_1(a) (pIn_Index[INDEXPOS_1((int)x,			((int)y)+(a))])
		#define K_I_2(a) (pIn_Index[INDEXPOS_2((int)x,	(int)y,	((int)z)+(a))])
		#define K_INIT_INDEX n = ((int)x)*SX + ((int)y)*SY + ((int)z)*SZ; // oldname : iMyLocalDataIndex
		
		#define GPU_IDX1		(K_I_0(1) >= vMin.x && K_I_0(0) <= vMax.x) 
		#define GPU_IDX2		(K_I_1(1) >= vMin.y && K_I_1(0) <= vMax.y) 
		#define GPU_IDX3		(K_I_2(1) >= vMin.z && K_I_2(0) <= vMax.z) 
		#ifndef ENABLE_GPU_IDX3
		#undef	GPU_IDX3	
		#define	GPU_IDX3		(1)
		#endif
		
		//~ for (n=0;n<iNumberOfPoints;++n) {
		
		__syncthreads();
		
		for (int x=0;x<I0;++x) if (GPU_IDX1) {
		for (int y=0;y<I0;++y) if (GPU_IDX2) { 
		for (int z=0;z<I0;++z) if (GPU_IDX3) { K_INIT_INDEX // n init here
		for (int w=0;w<SZ;++w,++n) {
			
			
			//~ #ifdef __DEVICE_EMULATION__ 
			//~ if (threadIdx.x == 0 && (n % 1024) == 0) printf("%d/%d (%0.1f%%)\n",n,N,100.0*float(n)/float(N));
			//~ #endif
			// load coordinates of next point to shared memory
			__syncthreads();
			if (threadIdx.x < D) pfSharedReadCache[ARRAY_INDEX(threadIdx.x,D)] = pIn_Points[ARRAY_INDEX(DATAPOINT_IDX(n,threadIdx.x),N*D)];
			__syncthreads();
			
			// check distance
			fSqDist = 0.0;
			for (d=0;d<D;++d) { a = pfSharedReadCache[d] - element[d]; fSqDist += a*a; } // compiler should loop-unroll
			if (fSqDist <= DBSCAN_PARAM_SQEPSILON) { 
				if (iNumPoints < NEIGHBOUR_COUNT) {
					// until we know that self is a core-point, only remember neightboors, don't spread yet
					iNeighbours[ARRAY_INDEX(iNumPoints,NEIGHBOUR_COUNT)] = n;
				} else {
					// spread cluster, and connect to other clusters
					if (iPointID >= 0) {
						int iOldClusterID = atomicMax(&pOut_ClusterID[ARRAY_INDEX(n,N)],iMyUnscannedClusterID);
						if (iOldClusterID >= DBSCAN_CLUSTER_ID_SCANNED_FIRST) iOldClusterID = iOldClusterID - DBSCAN_CLUSTER_ID_SCANNED_FIRST;
							
						// mark temporary-clusters as connected
						if (iOldClusterID >= DBSCAN_CLUSTER_ID_UNSCANNED_FIRST &&
							iOldClusterID <= DBSCAN_CLUSTER_ID_UNSCANNED_LAST &&
							iOldClusterID != iMyUnscannedClusterID) pOut_Connections[ARRAY_INDEX(iOldClusterID*DBSCAN_NUM_SEEDS + iMyUnscannedClusterID,CON_SIZE)] = 1;
					}
				}
				iNumPoints += 1;
			}
		}
		}}}
		
		#undef K_I_0
		#undef K_I_1
		#undef K_I_2
		#undef K_INIT_INDEX
		#undef GPU_IDX1
		#undef GPU_IDX2
		#undef GPU_IDX3
		
		if (iPointID >= 0) {
			if (iNumPoints < DBSCAN_PARAM_MINPTS) {
				pOut_ClusterID[ARRAY_INDEX(iPointID,N)] = DBSCAN_CLUSTER_ID_NOISE;
			} else {
				pOut_ClusterID[ARRAY_INDEX(iPointID,N)] = iMyScannedClusterID;
				if (iPointID >= 0) for (int m=0;m<NEIGHBOUR_COUNT;++m) {
					n = iNeighbours[ARRAY_INDEX(m,NEIGHBOUR_COUNT)];
					int iOldClusterID = atomicMax(&pOut_ClusterID[ARRAY_INDEX(n,N)],iMyUnscannedClusterID);
					if (iOldClusterID >= DBSCAN_CLUSTER_ID_SCANNED_FIRST) iOldClusterID = iOldClusterID - DBSCAN_CLUSTER_ID_SCANNED_FIRST;
						
					// mark temporary-clusters as connected
					if (iOldClusterID >= DBSCAN_CLUSTER_ID_UNSCANNED_FIRST &&
						iOldClusterID <= DBSCAN_CLUSTER_ID_UNSCANNED_LAST &&
						iOldClusterID != iMyUnscannedClusterID) pOut_Connections[ARRAY_INDEX(iOldClusterID*DBSCAN_NUM_SEEDS + iMyUnscannedClusterID,CON_SIZE)] = 1;
				}
			}
		}
	//~ }
}




void	dbscan_gpu	(DBScanData* p) {
	int	i,j,n,b;
	cudaError_t myLastErr; 
	int* pConnections = p->pConnections;
	int* pClusterIDs = p->pClusterIDs;
	
	#ifndef __DEVICE_EMULATION__
		CUDA_SAFE_CALL(cudaSetDevice(0));
	#endif
	
	#ifdef DBSCAN_ENABLE_GPU_DEBUG
	#define GPU_DEBUG_INTCOUNT DBSCAN_NUM_SEEDS
	int pDebug[GPU_DEBUG_INTCOUNT];
	#endif
	
	// allocate and init gpu buffers
	float*			pGPUIn_Data = 0;
	float*			pGPUIn_Index = 0;
	float3*			pGPUIn_Bounds = 0;
	int*			pGPUIn_SeedList = 0;
	int*			pGPUOut_ClusterID = 0;
	int*			pGPUOut_Connections = 0;
	int*			pGPUOut_Debug = 0;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_Data,			sizeof(p->pPoints)		));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_Index,			sizeof(p->pIndex)		));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_Bounds,		sizeof(p->pBounds)		));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_SeedList,		sizeof(p->pSeedList)	));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUOut_ClusterID,	sizeof(p->pClusterIDs)	));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUOut_Connections,	sizeof(p->pConnections)	));
	#ifdef DBSCAN_ENABLE_GPU_DEBUG
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUOut_Debug,		sizeof(pDebug)	));
	#endif
	
	// copy data from ram to vram
	CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_Data,		p->pPoints,		sizeof(p->pPoints),		cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy init data") 
	CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_Index,	p->pIndex,		sizeof(p->pIndex),		cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy init index")  
	
	
	// grid_size, block_size, mem_shared
	dim3  grid_size;
	dim3  block_size;
	unsigned int mem_shared = 0; // this is for dynamic alloc of shared mem, we alloc statically
	grid_size.x		= DBSCAN_NUM_SEEDS / kThreadBlockSize / GRIDHEIGHT;  // TODO : make sure  N is a multiple of kThreadBlockSize
	grid_size.y		= GRIDHEIGHT;
	grid_size.z		= 1;
	block_size.x	= kThreadBlockSize;
	block_size.y	= 1; 
	block_size.z	= 1;
	
	
	#define MB(a) ((int)(a)/1024/1024)
	printf("alloc %d %d %d %d  gridsize_x=%d\n",MB(p->pPoints),MB(p->pSeedList),MB(p->pClusterIDs),MB(p->pConnections),grid_size.x);
	
	
	// init clusterids
	for (n=0;n<N;++n) pClusterIDs[n] = DBSCAN_CLUSTER_ID_INIT;
	
	// init seedlist randomly
	for (i=0;i<DBSCAN_NUM_SEEDS;++i) {
		n = i;
		/*
		// random distribution is bad for locality within threadgroup when using index
		bool bAlreadyUsed;
		do {
			n = rand() % N;
			bAlreadyUsed = false;
			for (j=0;j<i;++j) if (p->pSeedList[j] == n) bAlreadyUsed = true;
		} while (bAlreadyUsed) ;
		*/
		EMU_CHECKBOUNDS("seedlist random init",n,N)
		p->pSeedList[i] = n;
		pClusterIDs[n] = i; // mark as UNSCANNED   
	}
	
	// "old" cluster ids, only assigned if one of them is finished
	int iNextFinishedClusterID = 1;
	int iFinishedClusterIDAssignment[DBSCAN_NUM_SEEDS];
	for (i=0;i<DBSCAN_NUM_SEEDS;++i) iFinishedClusterIDAssignment[i] = -1;
	
	// init connection list 
	for (i=0;i<DBSCAN_NUM_SEEDS*DBSCAN_NUM_SEEDS;++i) pConnections[i] = 0;
		
	
	//~ DEBUGCHECK_START() for (n=0;n<N;++n) if (pClusterIDs[n] != DBSCAN_CLUSTER_ID_INIT) printf("%d:%d ",n,pClusterIDs[n]); DEBUGCHECK_END()
	printf("\n");
	
	// mainloop
	bool bUpload_Connection	= true;
	bool bUpload_ClusterID	= true;
	bool bSeedAlive; // TODO : abort condition
	int iStepCounter = 0;
	int iCheckedPoints = 0;
	do {
		++iStepCounter;
		if (DBSCAN_EXIT_ON_FIRST_DOUBLEASIGN && iStepCounter > 6) break;
		//~ if (iStepCounter > 1) break;
		//~ printf("step\n");
		bSeedAlive = false;
		
		PROFILE_TIME_SECTION_START();
		
		// init bounds from seedlist
		float* pPointData = p->pPoints;
		float e = DBSCAN_PARAM_EPSILON;
		for (b=0;b<DBSCAN_NUM_BOUNDS;++b) { // one bounds-entry for every threadblock
			float3 vMin = make_float3(0,0,0);
			float3 vMax = make_float3(0,0,0);
			bool bFirst = true;
			for (i=0;i<DBSCAN_BOUNDS_ELEMENTS_PER_BLOCK;++i) {
				int n = p->pSeedList[b*DBSCAN_BOUNDS_ELEMENTS_PER_BLOCK+i];
				if (n < 0) continue; // last round, not all seeds used, skip the unused ones
				float3 vCur = make_float3(	DATAPOINT(pPointData,n,0),
											DATAPOINT(pPointData,n,1),
											DATAPOINT(pPointData,n,2) );
				vMin = bFirst ? vCur : min3(vMin,vCur);
				vMax = bFirst ? vCur : max3(vMax,vCur);
				bFirst = false;
			}
			p->pBounds[b*2+0] = make_float3(vMin.x-e,vMin.y-e,vMin.z-e);
			p->pBounds[b*2+1] = make_float3(vMax.x+e,vMax.y+e,vMax.z+e);
		}
		
		// upload data to vram
		if (bUpload_Connection)	CUDA_SAFE_CALL( cudaMemcpy(pGPUOut_Connections,	p->pConnections,sizeof(p->pConnections),cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy pConnections")
		if (bUpload_ClusterID)	CUDA_SAFE_CALL( cudaMemcpy(pGPUOut_ClusterID,	p->pClusterIDs,	sizeof(p->pClusterIDs),	cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy pClusterIDs")
		if (true)				CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_SeedList,		p->pSeedList,	sizeof(p->pSeedList),	cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy pSeedList")
		if (true)				CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_Bounds,		p->pBounds,		sizeof(p->pBounds),		cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy bounds")
		
		#ifdef DBSCAN_ENABLE_GPU_DEBUG
		CUDA_SAFE_CALL( cudaMemset(pGPUOut_Debug,	0,				sizeof(pDebug) 	)); 
		#endif
			
		PROFILE_TIME_SECTION_STEP(t_upload);
		
		// call kernel
		dbscan_kernel_main<<< grid_size, block_size, mem_shared >>>(
			pGPUIn_Data,
			pGPUIn_Index,
			pGPUIn_Bounds,
			pGPUIn_SeedList,
			pGPUOut_ClusterID,
			pGPUOut_Connections,
			N,
			pGPUOut_Debug);	
		CUDA_SAFE_CALL( cudaThreadSynchronize());HANDLE_ERROR("cudaThreadSynchronize")
		//~ printf("round finished\n");
		iCheckedPoints += DBSCAN_NUM_SEEDS;
		
		PROFILE_TIME_SECTION_STEP(t_kernel);
		
		// read back results
		CUDA_SAFE_CALL( cudaMemcpy(p->pClusterIDs, 	pGPUOut_ClusterID,		sizeof(p->pClusterIDs),		cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy pClusterIDs")
		CUDA_SAFE_CALL( cudaMemcpy(p->pConnections,	pGPUOut_Connections,	sizeof(p->pConnections),	cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy pGPUOut_Connections")
		#ifdef DBSCAN_ENABLE_GPU_DEBUG
		CUDA_SAFE_CALL( cudaMemcpy(pDebug,	pGPUOut_Debug,	sizeof(pDebug),	cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy pGPUOut_Connections")
		for (n=0;n<GPU_DEBUG_INTCOUNT;++n) if (pDebug[n] != 0) printf("GPU_DEBUG:%d\n",pDebug[n]);
		#endif
		
		PROFILE_TIME_SECTION_STEP(t_download);
		
		// iterate over all points to find unscanned points and fresh seeds
		int iFreshSeeds[DBSCAN_NUM_SEEDS]; // only needed when a cluster is complete, but collect while iterating
		int iSplitSeeds[DBSCAN_NUM_SEEDS]; // only needed when a cluster is complete, but collect while iterating
		int iFreshSeedCount = 0;
		int iSplitSeedCount = 0;
		for (i=0;i<DBSCAN_NUM_SEEDS;++i) iFreshSeeds[i] = -1;
		for (i=0;i<DBSCAN_NUM_SEEDS;++i) p->pSeedList[i] = -1;
		int c_init			= 0;
		int c_unscanned		= 0;
		int c_scanned		= 0;
		int c_finished		= 0;
		int c_activeseeds	= 0;
		
		for (n=0;n<N;++n) {
			int iClusterID = pClusterIDs[n];
			if (iClusterID == DBSCAN_CLUSTER_ID_INIT) ++c_init;
			if (iClusterID == DBSCAN_CLUSTER_ID_NOISE) 			++c_finished;
			if (iClusterID >= DBSCAN_CLUSTER_ID_FINISHED_FIRST)	++c_finished;
			if (iClusterID >= DBSCAN_CLUSTER_ID_SCANNED_FIRST &&
				iClusterID <= DBSCAN_CLUSTER_ID_SCANNED_LAST) ++c_scanned;
			
			if (iClusterID >= DBSCAN_CLUSTER_ID_UNSCANNED_FIRST &&
				iClusterID <= DBSCAN_CLUSTER_ID_UNSCANNED_LAST) {
				++c_unscanned;
				EMU_CHECKBOUNDS("seedscan",iClusterID,DBSCAN_NUM_SEEDS)
				if (p->pSeedList[iClusterID] == -1) {
					p->pSeedList[iClusterID] = n;
				} else if (iSplitSeedCount < DBSCAN_NUM_SEEDS) {
					iSplitSeeds[iSplitSeedCount++] = n;
				}
				bSeedAlive = true;
			}
			if (iClusterID == DBSCAN_CLUSTER_ID_INIT && iFreshSeedCount < DBSCAN_NUM_SEEDS) {
				EMU_CHECKBOUNDS("seedscan",iFreshSeedCount,DBSCAN_NUM_SEEDS)
				iFreshSeeds[iFreshSeedCount++] = n;
				bSeedAlive = true;
			}
		}
		for (i=0;i<DBSCAN_NUM_SEEDS;++i) if (p->pSeedList[i] != -1) ++c_activeseeds;
		if (ENABLE_STEP_PRINT) printf("seedscan init:%d unscanned:%d scanned:%d finished:%d activeseeds:%d\n",c_init,c_unscanned,c_scanned,c_finished,c_activeseeds);
		
		PROFILE_TIME_SECTION_STEP(t_seedscan);
		
		// detect if cluster was finished
		bUpload_Connection = false;
		bool bConnectionSpreadDone = false;
		//~ PrintConnectionMatrix("vor spread",iFinishedClusterIDAssignment,pConnections);
		
		//~ float t_finish_spread1 = 0;
		//~ float t_finish_spread2 = 0;
		//~ float t_finish_id = 0;
		//~ float t_finish_clear = 0;
		for (i=0;i<DBSCAN_NUM_SEEDS;++i) if (p->pSeedList[i] == -1) {
			//~ printf("seedfinish %d\n",i);
			// spread old id assignments over current connections 
			if (!bConnectionSpreadDone) { 
				bConnectionSpreadDone = true; 
				//~ printf("spread\n");
				DBScan_SpreadConnection(iFinishedClusterIDAssignment,pConnections);
				PROFILE_TIME_SECTION_SUM(t_finish_spread1);
				//~ PrintConnectionMatrix("nach spread",iFinishedClusterIDAssignment,pConnections);
			}
			// check if this seed already belongs to a finished cluster
			int iFinishedClusterID = iFinishedClusterIDAssignment[i];
			
			// if no cluster id is assigned via connections, create a new one
			if (iFinishedClusterID == -1) {
				//~ printf("create new cluster %d\n",iNextFinishedClusterID);
				iFinishedClusterID = iNextFinishedClusterID;
				iFinishedClusterIDAssignment[i] = iFinishedClusterID;
				//~ DBScan_SpreadConnection(iFinishedClusterIDAssignment,pConnections);
				DBScan_SpreadConnection_Aux(iFinishedClusterIDAssignment,pConnections,i); // i is finished, so spread to connected before cleanup
				++iNextFinishedClusterID;
				PROFILE_TIME_SECTION_SUM(t_finish_spread2);
			}
			
			// overwrite scanned with new FinishedClusterID, no unscanned of this type left
			int iFinishedClusterID_Written = iFinishedClusterID + DBSCAN_CLUSTER_ID_FINISHED_FIRST;
			int iScanned = i + DBSCAN_CLUSTER_ID_SCANNED_FIRST;
			for (n=0;n<N;++n) if (pClusterIDs[n] == iScanned) pClusterIDs[n] = iFinishedClusterID_Written;
			PROFILE_TIME_SECTION_SUM(t_finish_id);
					// TODO : active seeds connected to i might still run into these finished ones ?
					// no, otherwise it would not have finished ?  only happens if others have overwritten new seeds by this one,
					// but then they already know about the connection
			
			// clear connection matrix for i (and re-upload)
			int irow		= i*DBSCAN_NUM_SEEDS;
			int irow_end	= irow + DBSCAN_NUM_SEEDS;
			int jrow		= i;
			for (;irow<irow_end;++irow,jrow+=DBSCAN_NUM_SEEDS) {
				EMU_CHECKBOUNDS("finish_seed",i,DBSCAN_NUM_SEEDS)
				EMU_CHECKBOUNDS("finish_seed",j,DBSCAN_NUM_SEEDS)
				pConnections[irow] = 0;
				pConnections[jrow] = 0;
			}
			bUpload_Connection = true;
			PROFILE_TIME_SECTION_SUM(t_finish_clear);
			
			// start fresh seed
			if (iFreshSeedCount > 0) {
				// at the beginning there are still lots of points that haven't been touched at all, so start seperate seeds
				EMU_CHECKBOUNDS("finish_seed",iFreshSeedCount-1,DBSCAN_NUM_SEEDS)
				int n = iFreshSeeds[--iFreshSeedCount];
				p->pSeedList[i] = n;
				pClusterIDs[n] = i; // mark as UNSCANNED   
			} else if (iSplitSeedCount > 0) {
				// later when there are no untouched points, but still clusters that need to be expanded, so expand them faster by splitting,
				// so that there are multiple seeds per "cluster"
				EMU_CHECKBOUNDS("finish_seed",iSplitSeedCount-1,DBSCAN_NUM_SEEDS)
				int n = iSplitSeeds[--iSplitSeedCount];
				int splitroot_unscanned_id = pClusterIDs[n];
				p->pSeedList[i] = n;
				pClusterIDs[n] = i; // mark as UNSCANNED   
				EMU_CHECKBOUNDS("finish_seed",splitroot_unscanned_id,DBSCAN_NUM_SEEDS)
				// we did split, so we are connected to our parent
				j = splitroot_unscanned_id;
				pConnections[i*DBSCAN_NUM_SEEDS + j] = 1;
				pConnections[j*DBSCAN_NUM_SEEDS + i] = 1;
			} else {
				p->pSeedList[i] = -1; // algorithm is almost finished
			}
			iFinishedClusterIDAssignment[i] = -1;
		}
		
		PROFILE_TIME_SECTION_STEP(t_finish_seed);
		
		#ifdef PROFILE_TIME_ENABLE
			//~ printf("spread1=%3.1f spread2=%3.1f id=%3.1f clear=%3.1f\n",t_finish_spread1,t_finish_spread2,t_finish_id,t_finish_clear);
			printf("upload=%3.1f kernel=%3.1f download=%3.1f seedscan=%3.1f finish_seed=%3.1f\n",t_upload,t_kernel,t_download,t_seedscan,t_finish_seed);
		#endif
		
		//~ break; // TODO : DEBUG
		
		
		if (DBSCAN_DUMP_STEPS) {
			char mystr[256];
			sprintf(mystr,"steps/step_gpu_%05d.txt",iStepCounter);
			DBScan_WriteToFile(p,mystr); 
		}
		
	} while (bSeedAlive);
	
	printf("N-iCheckedPoints=%d N=%d  fac=%0.1f\n",(int)(N-iCheckedPoints),(int)N,float(iCheckedPoints)/float(N));
	printf("finished, next iNextFinishedClusterID %d\n",iNextFinishedClusterID);	
	// release GPU-memory
	CUDA_SAFE_CALL(cudaFree(pGPUIn_Data));
	CUDA_SAFE_CALL(cudaFree(pGPUIn_Index));
	CUDA_SAFE_CALL(cudaFree(pGPUIn_Bounds));
	CUDA_SAFE_CALL(cudaFree(pGPUIn_SeedList));
	CUDA_SAFE_CALL(cudaFree(pGPUOut_ClusterID));
	CUDA_SAFE_CALL(cudaFree(pGPUOut_Connections));
}

void PrintConnectionMatrix (const char* szHeader,int* iFinishedClusterIDAssignment,int* pConnections) {
	int i,j;
	printf("conmatrix %s\n",szHeader);
	for (i=0;i<DBSCAN_NUM_SEEDS;++i) printf("%d ",iFinishedClusterIDAssignment[i]);
	printf("\n");
	for (i=0;i<DBSCAN_NUM_SEEDS;++i) {
		//~ for (int j=0;j<DBSCAN_NUM_SEEDS;++j) printf("%d ",DBScan_CheckConnection(pConnections,i,j)?1:0);
		for (j=0;j<DBSCAN_NUM_SEEDS;++j) printf("%d",pConnections[i*DBSCAN_NUM_SEEDS + j]);
		printf("\n");
	}
	
}

// spreads iFinishedClusterIDAssignment over direct and indirect connections (a<->b<->c)
void DBScan_SpreadConnection (int* iFinishedClusterIDAssignment,int* pConnections) {
	for (int i=0;i<DBSCAN_NUM_SEEDS;++i) if (iFinishedClusterIDAssignment[i] != -1) {
		DBScan_SpreadConnection_Aux(iFinishedClusterIDAssignment,pConnections,i);
	}
}

void DBScan_SpreadConnection_Aux (int* iFinishedClusterIDAssignment,int* pConnections,int i) {
	EMU_CHECKBOUNDS("cpuspread",i,DBSCAN_NUM_SEEDS)
	for (int j=0;j<DBSCAN_NUM_SEEDS;++j) {
		if (j == i) continue;
		if (DBScan_CheckConnection(pConnections,i,j)) {
			// j and i are connected
			if (iFinishedClusterIDAssignment[j] != iFinishedClusterIDAssignment[i]) {
				if (iFinishedClusterIDAssignment[j] != -1) {
					//~ printf("SpreadConnection : unexpected double assignment %d,%d (%d,%d)\n",iFinishedClusterIDAssignment[j],iFinishedClusterIDAssignment[i],j,i);
					
					//PrintConnectionMatrix("unexpected",iFinishedClusterIDAssignment,pConnections);
					if (DBSCAN_EXIT_ON_FIRST_DOUBLEASIGN) exit(0);
					//~ continue;
				}
				iFinishedClusterIDAssignment[j]  = iFinishedClusterIDAssignment[i];
				// fresh connection, spread
				DBScan_SpreadConnection_Aux(iFinishedClusterIDAssignment,pConnections,j);
			}
		}
	}
}
