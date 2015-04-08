#define TEST_KERNEL_NEWSEED	0
#define TEST_KERNEL_REFILL	0

#define kThreadBlockMinPtsScale 32

#define GPU_POINT_COORDS(pData,point_id,d) (pData[ARRAY_INDEX((point_id)*D + (d),N*D)])
#define ARRAY_INDEX(a,arrlen) (a) // can be used for checking

#define DBSCAN_WITHIDX_CLUSTER_ID_INIT					(0)
#define DBSCAN_WITHIDX_CLUSTER_ID_NOISE					(1)
#define DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_FIRST		(2)
#define DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST	(DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_FIRST+N)
#define DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_2_FINAL(x)	(x + N)
#define DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(x)	(x - N)

#define INF_32 (0x7fffffff)      //  ; 32 bit max value, signed
#define MY_ASSERT(x,msg) if (!(x)) { printf("ASSERT FAILED (%s) : %s\n",#x,msg); exit(0); }


//~ #define DBSCAN_ID_LOOKUP_IN_VRAM
//~ #define DBSCAN_ID_LOOKUP_IN_VRAM_CHECK

//~ #define EMU_CHECKBOUNDS(name,index,num) 
#define EMU_CHECKBOUNDS(name,index,num) if (index < 0 || index >= num) { printf("EMU_CHECKBOUNDS(%s,%d,%d) failed\n",name,(int)index,(int)num); exit(0); }

#define PROFILE_TIME_ENABLE
#ifdef PROFILE_TIME_ENABLE
	#define PROFILE_TIME_SECTION_START() ProfileTimerStartSection()
	#define PROFILE_TIME_SECTION_STEP(name) float name = ProfileTimerStartSection()
	#define PROFILE_TIME_SECTION_SUM(name) name += ProfileTimerStartSection()
#else
	#define PROFILE_TIME_SECTION_START()
	#define PROFILE_TIME_SECTION_STEP(name)
	#define PROFILE_TIME_SECTION_SUM(name) 
#endif



/*
int atomicMin(int* address, int val);
int atomicMax(int* address, int val);

int   atomicExch(int*   address, int   val);
uint  atomicExch(uint*  address, uint  val);
float atomicExch(float* address, float val);

unsigned int atomicInc(unsigned int* address,
                       unsigned int val);
reads the 32-bit word old located at the address address in global or shared
memory, computes ((old >= val) ? 0 : (old+1)), and stores the result
back to memory at the same address. These three operations are performed in one
atomic transaction. The function returns old.

unsigned int atomicDec(unsigned int* address,
                       unsigned int val);


int atomicCAS(int* address, int compare, int val);
unsigned int atomicCAS(unsigned int* address,
                       unsigned int compare,
                       unsigned int val);
unsigned long long int atomicCAS(unsigned long long int* address,
                                 unsigned long long int compare,
                                 unsigned long long int val);
writes (old == compare ? val : old) , and returns old



Atomic functions operating on shared memory and atomic functions operating on
64-bit words are only available for devices of compute capability 1.2 and above.


*/

__constant__ unsigned int gConst_pFinalChainIDs[DBSCAN_NUM_SEEDS];

	
char gsInfoGPUaux[512] = "";


// ***** ***** ***** ***** ***** main kernel
#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
#define DBSCAN_LOOKUP_PARAM ,pClusterIDLookup
#else
#define DBSCAN_LOOKUP_PARAM 
#endif


// function used by kernel, inlined automatically
__device__ void DBScanGPU_TryMarkAsCandidate (	const unsigned int iCurPointID,
												unsigned int*	pPointState,
												const unsigned int	iCandidateID,
												const unsigned int iSeedID,
												unsigned int* piSeedState_NotListedLen,
												unsigned int* piSeedState_ListLen,
												unsigned int* pCandidateLists,
												unsigned int*	pConnectionMatrix	// DBSCAN_NUM_SEEDS^2
												#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
												,unsigned int* pClusterIDLookup // [N]
												#endif
												) 
{
	unsigned int iOldState = atomicCAS(&pPointState[iCurPointID],DBSCAN_WITHIDX_CLUSTER_ID_INIT,iCandidateID); // this also marks a few candidates if ownpoint is an outlier
	//~ if (iCurPointID == 126530) printf("DEBUG002:set:old=%d,cid=%d,seed=%d\n",iOldState,iCandidateID,iSeedID);

	if (iOldState == DBSCAN_WITHIDX_CLUSTER_ID_INIT) { // otherwise just count and don't do anything
		
		// claimed as candidate, add to list
		// pointstate already set, need to add to list now
		unsigned int iMyListIndex = atomicInc(piSeedState_ListLen,0xffffffff);
		if (iMyListIndex < CANDIDATE_LIST_MAXLEN) 
				pCandidateLists[iSeedID*CANDIDATE_LIST_MAXLEN+iMyListIndex] = iCurPointID;
		else	atomicInc(piSeedState_NotListedLen,0xffffffff);
			
	} else if (iOldState != DBSCAN_WITHIDX_CLUSTER_ID_NOISE) { // connection with other cluster detected	
		// iOldState can be candidate or final, transform to one of them for lookup table
		if (iOldState < DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST) 
			iOldState = DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_2_FINAL(iOldState);
	
		#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
		if (iOldState != DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_2_FINAL(iCandidateID) &&
			iOldState >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST &&
			iOldState < DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST+N) {
			// lookup in global mem.. faster for big seednum
			unsigned int iOtherSeedID = pClusterIDLookup[iOldState-DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST];
		#else 
		if (iOldState != DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_2_FINAL(iCandidateID)) { // this could save us the lookup, rather probably case also
			// use lookup table to get seedid for connection matrix
			unsigned int iOtherSeedID = 0xffffffff;
			#pragma unroll // unroll next loop
			for (int d=0;d<DBSCAN_NUM_SEEDS;++d) // warning, slow for big seednum
				if (gConst_pFinalChainIDs[d] == iOldState) iOtherSeedID = d;		
		#endif
				
			// set bit in connection matrix.. atomic not needed as at least one of concurrent writes is guaranteed to succeed
			if (iOtherSeedID < DBSCAN_NUM_SEEDS) {
				//~ atomicCAS(&pConnectionMatrix[iSeedID*DBSCAN_NUM_SEEDS + iOtherSeedID],0,1);
				pConnectionMatrix[iSeedID*DBSCAN_NUM_SEEDS + iOtherSeedID] = 1;
			}
			
		#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
		}
		#else 
		}	
		#endif
	}
}

/// kernel code : this gets executed on the GPU
__global__ static void dbscan_kernel_main (
	float*			pPointCoords,		// N*D
	unsigned int*	pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
	float*			pIndex,				// INDEX_NUM_FLOATS
	unsigned int*	pSeedStates,		// DBSCAN_NUM_SEEDS * x  (notlisted,listlen,iNeighBorCount  : atomicops)
	//~ unsigned int*	pFinalChainIDs,		// DBSCAN_NUM_SEEDS (constant memory, values >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST)
	unsigned int*	pCandidateLists,	// DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN (fresh-seeds)
	unsigned int*	pConnectionMatrix	// DBSCAN_NUM_SEEDS^2
	#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
	,unsigned int* pClusterIDLookup // [N]
	#endif
	) 
{
	#define NEIGHBOR_BUFFER_SIZE (DBSCAN_PARAM_MINPTS-1)
	__shared__ unsigned int	iOwnPointID;
	__shared__ unsigned int	iNeighborCount;
	__shared__ unsigned int	iNeighbors[NEIGHBOR_BUFFER_SIZE];
	__shared__ float		fOwnPoint[D];
	__shared__ float3 		vMin;
	__shared__ float3 		vMax;
	int d,n;
	
	// prepare variables
	const unsigned int	iSeedID 		= BLOCKIDX; // in [0;DBSCAN_NUM_SEEDS[
	const unsigned int	iCandidateID	= DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(gConst_pFinalChainIDs[iSeedID]);
	//~ if (threadIdx.x == 0) printf("iSeedID=%d, iCandidateID=%d\n",(int)iSeedID,(int)iCandidateID);
	unsigned int*	piSeedState_NotListedLen	= &pSeedStates[iSeedID*SEEDSTATEDIM + 0];
	unsigned int*	piSeedState_ListLen			= &pSeedStates[iSeedID*SEEDSTATEDIM + 1];
	unsigned int*	piSeedState_NeighborCount	= &pSeedStates[iSeedID*SEEDSTATEDIM + 2]; // ("atomics not in shared mem for compute 1.1"); 
	
	// get iOwnPointID from seedstate(cpu) or from seedlist
	if (threadIdx.x == 0) {
		if (*piSeedState_ListLen > CANDIDATE_LIST_MAXLEN) // cut/limit overincrementation before polling
			*piSeedState_ListLen = CANDIDATE_LIST_MAXLEN;
		iOwnPointID = pSeedStates[iSeedID*SEEDSTATEDIM + 3]; 
		if (iOwnPointID == INF_32) {
			// no seed from cpu, take one from list
			unsigned int ll = *piSeedState_ListLen;
			if (ll > 0) {
				unsigned int llm1 = ll - 1;
				*piSeedState_ListLen = llm1;
				iOwnPointID = pCandidateLists[ARRAY_INDEX(
					iSeedID*CANDIDATE_LIST_MAXLEN + llm1,
					DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN)];
			} else {
				//return; // seed terminated and not revivable, algorithm nearing it's finish
				// but exit function only after syncthreads
			}
		} else {
			pSeedStates[iSeedID*SEEDSTATEDIM + 3] = INF_32;
		} 
	}
	__syncthreads();
	if (iOwnPointID == INF_32) return;
	
	
	// read own point coordinates
	if (threadIdx.x < D) fOwnPoint[threadIdx.x] = GPU_POINT_COORDS(pPointCoords,iOwnPointID,threadIdx.x);
	__syncthreads();
	
	// calculate epsilon area used for index
	if (threadIdx.x == 0) {
		vMin.x = fOwnPoint[0] - DBSCAN_PARAM_EPSILON;
		vMin.y = fOwnPoint[1] - DBSCAN_PARAM_EPSILON;
		vMin.z = fOwnPoint[2] - DBSCAN_PARAM_EPSILON;
		vMax.x = fOwnPoint[0] + DBSCAN_PARAM_EPSILON;
		vMax.y = fOwnPoint[1] + DBSCAN_PARAM_EPSILON;
		vMax.z = fOwnPoint[2] + DBSCAN_PARAM_EPSILON;
		iNeighborCount = 0;
		*piSeedState_NeighborCount = 0;
	}
	
	#pragma unroll // unroll next loop
	for (d=0;d<kThreadBlockMinPtsScale;++d) {
		if (threadIdx.x*kThreadBlockMinPtsScale + d < NEIGHBOR_BUFFER_SIZE) 
			iNeighbors[threadIdx.x*kThreadBlockMinPtsScale + d] = 0xffffffff; // mark as invalid
	}
	__syncthreads();
	
	
	#define K_I_0(a) (pIndex[INDEXPOS_0(				((int)x)+(a))])
	#define K_I_1(a) (pIndex[INDEXPOS_1((int)x,			((int)y)+(a))])
	#define K_I_2(a) (pIndex[INDEXPOS_2((int)x,	(int)y,	((int)z)+(a))])
	#define K_INIT_INDEX n = (x)*SX + (y)*SY + (z)*SZ; // oldname : iMyLocalDataIndex
	
	#define GPU_IDX1		(K_I_0(1) >= vMin.x && K_I_0(0) <= vMax.x) 
	#define GPU_IDX2		(K_I_1(1) >= vMin.y && K_I_1(0) <= vMax.y) 
	#define GPU_IDX3		(K_I_2(1) >= vMin.z && K_I_2(0) <= vMax.z) 
	#ifndef ENABLE_GPU_IDX3
	#undef	GPU_IDX3	
	#define	GPU_IDX3		(1)
	#endif
	
	
	// iteration : 299
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
	float* iy;
	float* iz;
	float* iw;
	
	#ifdef MYTEST_DISABLE_INDEX 
	for (x=0;x<I0;++x) {
	for (y=0;y<I0;++y) { 
	for (z=0;z<I0;++z) { K_INIT_INDEX // n init here 
	for (w=0;w<SZ;w+=kThreadBlockSize,n+=kThreadBlockSize) {
	#else
	for (iy = &pIndex[INDEXPOS_0(0    )], x=0;x<I0;++x,++iy) if (iy[1] >= vMin.x && *iy <= vMax.x) {
	for (iz = &pIndex[INDEXPOS_1(x,0  )], y=0;y<I0;++y,++iz) if (iz[1] >= vMin.y && *iz <= vMax.y) { 
	for (iw = &pIndex[INDEXPOS_2(x,y,0)], z=0;z<I0;++z,++iw) if (iw[1] >= vMin.z && *iw <= vMax.z) { K_INIT_INDEX // n init here 
	for (w=0;w<SZ;w+=kThreadBlockSize,n+=kThreadBlockSize) {
	#endif
	
	/*
	// iteration : other try, but was slower .. 305 for 500k
	unsigned int n1;
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
	float* iy;
	float* iz;
	float* iw;
	for (iy = &pIndex[INDEXSTART_0],        x=0;x<I0*(I0+1)	;x+=(I0+1),++iy)	if (iy[1] >= vMin.x && *iy <= vMax.x) {
	for (iz = &pIndex[INDEXSTART_1+x],      y=0;y<I0*(I0+1)	;y+=(I0+1),++iz)	if (iz[1] >= vMin.y && *iz <= vMax.y) { n1 = (x/(I0+1))*SX + (y/(I0+1))*SY;
	for (iw = &pIndex[INDEXSTART_2+y+x*I0], z=0;z<I0*SZ		;z+=SZ,++iw)		if (iw[1] >= vMin.z && *iw <= vMax.z) { n = n1 + z; // n init here 
	for (w=0;w<SZ;w+=kThreadBlockSize,n+=kThreadBlockSize) {
	*/
	
	//~ for (unsigned int x=0;x<I0;++x) if (GPU_IDX1) {
	//~ for (unsigned int y=0;y<I0;++y) if (GPU_IDX2) { 
	//~ for (unsigned int z=0;z<I0;++z) if (GPU_IDX3) { K_INIT_INDEX // n init here 
	//~ for (unsigned int w=0;w<SZ;w+=kThreadBlockSize,n+=kThreadBlockSize) {
	//~ for (unsigned int nend = n + SZ;n<nend;n+=kThreadBlockSize) {  // slower than normal, no an improvement
		// read current point
		unsigned int iCurPointID = n + threadIdx.x;
		
		// calc distance
		float fSqDist = 0; // 1110.0 + fOwnPoint[0];
		#pragma unroll // unroll next loop
		for (d=0;d<D;++d) { float a = GPU_POINT_COORDS(pPointCoords,iCurPointID,d) - fOwnPoint[d]; fSqDist += a*a; }
		
		// check distance
		if (fSqDist <= DBSCAN_PARAM_SQEPSILON) {
			// self is also counted (iCurPointID == iOwnPointID) here for simplicity
			// until we know that self is a core-point, only remember neightbors, don't spread yet (atomic op on shared memory)
			unsigned int myNeighborIndex = 0x7fffffff;
			if (iNeighborCount < NEIGHBOR_BUFFER_SIZE) { // otherwise there are already minpts-1 OTHER neighbors besides me
				myNeighborIndex = atomicInc(piSeedState_NeighborCount,0xffffffff); // inc by 1 and return old
				if (myNeighborIndex == DBSCAN_PARAM_MINPTS) 
					iNeighborCount = DBSCAN_PARAM_MINPTS;
			}
			
			if (myNeighborIndex < NEIGHBOR_BUFFER_SIZE) {
				// not enough points yet, save in buffer for later
				iNeighbors[myNeighborIndex] = iCurPointID;
			} else {
				// index>=NEIGHBOR_BUFFER_SIZE(=(DBSCAN_PARAM_MINPTS-1)) so iNeighborCount>=DBSCAN_PARAM_MINPTS
				DBScanGPU_TryMarkAsCandidate(iCurPointID,pPointState,iCandidateID,iSeedID,
					piSeedState_NotListedLen,piSeedState_ListLen,pCandidateLists,pConnectionMatrix DBSCAN_LOOKUP_PARAM);
			}
		}
	#ifdef MYTEST_DISABLE_INDEX 
	}}}}
	#else
	}}}}
	#endif
	
	#undef K_I_0
	#undef K_I_1
	#undef K_I_2
	#undef K_INIT_INDEX
	#undef GPU_IDX1
	#undef GPU_IDX2
	#undef GPU_IDX3
	
	// wait until all are finished, so we know if it's a corepoint
	__syncthreads();
	
	// process stored neighbors
	#pragma unroll // unroll next loop:  
	for (d=0;d<kThreadBlockMinPtsScale;++d) {
		if (iNeighborCount >= DBSCAN_PARAM_MINPTS && 
			threadIdx.x*kThreadBlockMinPtsScale + d < NEIGHBOR_BUFFER_SIZE && 
			iNeighbors[threadIdx.x*kThreadBlockMinPtsScale + d] < N) {
			DBScanGPU_TryMarkAsCandidate(iNeighbors[threadIdx.x*kThreadBlockMinPtsScale + d],pPointState,iCandidateID,iSeedID,
				piSeedState_NotListedLen,piSeedState_ListLen,pCandidateLists,pConnectionMatrix DBSCAN_LOOKUP_PARAM);
		}
	}
	
	__syncthreads();
	
	// mark self as either confirmed-core-point or noise
	if (threadIdx.x == 0) {
		//~ if (iSeedID == 3) printf("DEBUG002:finalseed:%d>=%d\n",*piSeedState_NeighborCount,(int)DBSCAN_PARAM_MINPTS);
		//~ if (iOwnPointID == 126530) printf("DEBUG002:final:%d>=%d\n",*piSeedState_NeighborCount,(int)DBSCAN_PARAM_MINPTS);
		if (iNeighborCount >= DBSCAN_PARAM_MINPTS)
				pPointState[iOwnPointID] = DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_2_FINAL(iCandidateID);
		else	pPointState[iOwnPointID] = DBSCAN_WITHIDX_CLUSTER_ID_NOISE;
			
		if (*piSeedState_ListLen > CANDIDATE_LIST_MAXLEN)
			*piSeedState_ListLen = CANDIDATE_LIST_MAXLEN; // clip if over-incremented and didn't fit into list
	}
}




// ***** ***** ***** ***** ***** newseeds


/// helper kernel, searches new seeds
/// kernel code : this gets executed on the GPU
__global__ static void dbscan_kernel_newseeds (
	unsigned int*	pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
	unsigned int*	pHelperData,		// DBSCAN_NUM_SEEDS + x
	unsigned int	iNewSeedsNeeded
	) 
{
	#define NEWSEEDSCAN_SECTIONSIZE (N/NUM_THREADS_NEWSEEDSCAN)
	unsigned int		iMyID = MYID; // #define MYID		(BLOCKIDX * blockDim.x + threadIdx.x)  
	unsigned int*		piNextScannedOffset		= &pHelperData[DBSCAN_NUM_SEEDS+NEWSEEDSCAN_NUM_PARAMS+iMyID];
	unsigned int*		piNumFound				= &pHelperData[DBSCAN_NUM_SEEDS+0];
	unsigned int		 i						= *piNextScannedOffset;
	
	// used sharedmem to allow quick abort
	__shared__ unsigned int	iNumFoundCache;
	if (threadIdx.x == 0) iNumFoundCache = 0;
	__syncthreads();
	
	// check for earlyout
	if (i >= NEWSEEDSCAN_SECTIONSIZE) return; // this block is finished
	const unsigned int	 base					= iMyID * NEWSEEDSCAN_SECTIONSIZE;
	
	// scanloop
	for (;i<NEWSEEDSCAN_SECTIONSIZE;++i) {
		if (iNumFoundCache >= iNewSeedsNeeded) break; // stop scanning, this point will be scanned again next time
		unsigned int n = i + base;
		if (pPointState[n] == DBSCAN_WITHIDX_CLUSTER_ID_INIT) {
			unsigned int iMyIndex = atomicInc(piNumFound,0xffffffff);
			if (iMyIndex < iNewSeedsNeeded) {
				pHelperData[iMyIndex] = n;
			} else {
				iNumFoundCache = iNewSeedsNeeded; // abort, we've got enough
				break;  // couldn't register, this point has to be scanned again next time
			}
		}
		// else point cannot be fresh seed, so index can safely be incremented
	}
	//			increment if point is not init -> cannot be fresh seed ever again
	//			increment if point is     init and could     be registered as new seed
	// DON'T	increment if point is     init and could NOT be registered as new seed -> has to be scanned again next time
	// abort if we found enough seeds
	*piNextScannedOffset = i;
	// piNextScannedOffset is unique for every thread, not just for every threadblock , so no synching is neccessary.
	// the number of threads running in parallel here is rather limited, only 4 * threadblocksize
}


// ***** ***** ***** ***** ***** refill


/// helper kernel, refills candidate lists
/// kernel code : this gets executed on the GPU
__global__ static void dbscan_kernel_refill (
	unsigned int*	pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
	unsigned int*	pSeedStates,		// DBSCAN_NUM_SEEDS * x  (notlisted,listlen,iNeighBorCount  : atomicops)
	unsigned int*	pCandidateLists		// DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN (fresh-seeds)
	) 
{
	const unsigned int	iSeedID 		= BLOCKIDX; // in [0;DBSCAN_NUM_SEEDS[
	const unsigned int	iCandidateID	= DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(gConst_pFinalChainIDs[iSeedID]);
	//~ if (threadIdx.x == 0) printf("iSeedID=%d, iCandidateID=%d\n",(int)iSeedID,(int)iCandidateID);
	unsigned int*	piSeedState_NotListedLen	= &pSeedStates[iSeedID*SEEDSTATEDIM + 0];
	unsigned int*	piSeedState_ListLen			= &pSeedStates[iSeedID*SEEDSTATEDIM + 1];
	unsigned int 	iMaxTake 					= min(*piSeedState_NotListedLen,CANDIDATE_LIST_REFILL);
	if (*piSeedState_ListLen > 0) return;		// still seeds in list, nothing to do
	if (*piSeedState_NotListedLen == 0) return;	// no candidates left / fresh seed-chain
	__shared__ unsigned int	iNumFound;
	if (threadIdx.x == 0) iNumFound = 0; // piSeedState_ListLen
	__syncthreads();
	
	const unsigned int	iStep = kThreadBlockSize; // total number of threads for this seed
	
	// iterate over points
	for (int n=threadIdx.x; n<N && iNumFound < iMaxTake ; n+=iStep) {
		if (pPointState[n] == iCandidateID) {
			unsigned int iMyIndex = atomicInc(piSeedState_ListLen,0xffffffff);  // has to be cut down in the end
			if (iMyIndex < iMaxTake) {
				atomicDec(piSeedState_NotListedLen,INF_32);
				pCandidateLists[iSeedID*CANDIDATE_LIST_MAXLEN + iMyIndex] = n;
				if (iMyIndex + 1 >= iMaxTake) iNumFound = iMaxTake; // abort, we've got enough
			}
		}
	}
	
	__syncthreads();
	// cut down over-incrementation
	if (threadIdx.x == 0) {
		if (*piSeedState_ListLen > iMaxTake)
			*piSeedState_ListLen = iMaxTake;
	}
}






// ***** ***** ***** ***** ***** utils : final id


unsigned int	giFinalClusterIDs[N];
unsigned int*	gpFinalChainIDs = 0;
unsigned int*	gpConnectionMatrix = 0;
unsigned int	giLastFinalClusterID = 0;


unsigned int	GetFinalClusterIDIndexFromSeedID	(int iSeedID) {
	unsigned int iFinalChainID = gpFinalChainIDs[iSeedID];
	MY_ASSERT(iFinalChainID >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST,"GetFinalClusterIDIndexFromSeedID too low");
	unsigned int iMyIndex = iFinalChainID - DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST;
	MY_ASSERT(iMyIndex < N,"GetFinalClusterIDIndexFromSeedID too high");
	return iMyIndex;
}
	
unsigned int	GetFinalClusterIDBySeedID	(int iSeedID) {
	return	giFinalClusterIDs[GetFinalClusterIDIndexFromSeedID(iSeedID)];
}
void			SetFinalClusterIDBySeedID	(int iSeedID,unsigned int iFinalClusterID) {
			giFinalClusterIDs[GetFinalClusterIDIndexFromSeedID(iSeedID)] = iFinalClusterID;
}
	

// ***** ***** ***** ***** ***** utils : connection



bool DBScan_CheckConnection			(int i,int j) { 
	EMU_CHECKBOUNDS("DBScan_CheckConnection",i,DBSCAN_NUM_SEEDS)
	EMU_CHECKBOUNDS("DBScan_CheckConnection",j,DBSCAN_NUM_SEEDS)
	return	gpConnectionMatrix[i*DBSCAN_NUM_SEEDS + j] || 
			gpConnectionMatrix[j*DBSCAN_NUM_SEEDS + i]; 
}

void DBScan_ClearConnection (int i) {
	for (int j=0;j<DBSCAN_NUM_SEEDS;++j) {
		gpConnectionMatrix[i*DBSCAN_NUM_SEEDS + j] = 0;
		gpConnectionMatrix[j*DBSCAN_NUM_SEEDS + i] = 0;
	}
}

void DBScan_SetConnection (int i,int j) {
	EMU_CHECKBOUNDS("DBScan_SetConnection i",i,DBSCAN_NUM_SEEDS)
	EMU_CHECKBOUNDS("DBScan_SetConnection j",j,DBSCAN_NUM_SEEDS)
	gpConnectionMatrix[i*DBSCAN_NUM_SEEDS + j] = 1;
	gpConnectionMatrix[j*DBSCAN_NUM_SEEDS + i] = 1;
}


void DBScan_SpreadConnection_DebugDump (const char* szMsg) {
	return;
	printf("########## %s\n",szMsg);
	for (int c=0;c<=giLastFinalClusterID;++c) {
		bool bFound = false;
		for (int x=0;x<DBSCAN_NUM_SEEDS;++x) {
			unsigned int cid = GetFinalClusterIDBySeedID(x);
			if (cid == c) {
				if (!bFound) {bFound = true; printf("c:%5d:",c);}
				printf("%d,",x);
			}
		}
		if (bFound) printf("\n");
	}
}

int giDBScanClusterDoubleAssignmentCounter = 0;
				
void DBScan_SpreadConnection_Aux (int i) {
	unsigned int iFinalClusterIDA = GetFinalClusterIDBySeedID(i);
	MY_ASSERT(iFinalClusterIDA != INF_32,"DBScan_SpreadConnection_Aux on seed without clusterid ?");
	EMU_CHECKBOUNDS("cpuspread",i,DBSCAN_NUM_SEEDS)
	for (int j=0;j<DBSCAN_NUM_SEEDS;++j) {
		if (j == i) continue;
		if (DBScan_CheckConnection(i,j)) { // j and i are connected
			unsigned int iFinalClusterIDB = GetFinalClusterIDBySeedID(j);
			if (iFinalClusterIDB != iFinalClusterIDA) {
				if (iFinalClusterIDB != INF_32) {
					++giDBScanClusterDoubleAssignmentCounter;
					printf("warning : DBScan_SpreadConnection_Aux unexpected double assignment : i=%d,j=%d,a=%d,b=%d\n",(int)i,(int)j,(int)iFinalClusterIDA,(int)iFinalClusterIDB);
					//MY_ASSERT(0,"DBScan_SpreadConnection_Aux unexpected double assignment"); // fatal ? only during debug probably
				}
				SetFinalClusterIDBySeedID(j,iFinalClusterIDA);
				DBScan_SpreadConnection_Aux(j); // spread
			}
		}
	}
}


// spreads ClusterID Assignments over direct and indirect connections (a<->b<->c)
void DBScan_SpreadConnection () {
	//~ printf("DBScan_SpreadConnection start\n");
	for (int i=0;i<DBSCAN_NUM_SEEDS;++i) {
		if (GetFinalClusterIDBySeedID(i) != INF_32)
			DBScan_SpreadConnection_Aux(i);
	}
	//~ printf("DBScan_SpreadConnection end\n");
}






// ***** ***** ***** ***** ***** DBScanVerifyCandidates


//~ #define DBSCAN_VERIFY_CANDIDATES(sectionname) DBScanVerifyCandidates(p,sectionname,gpu_pPointState,gpu_pSeedStates,gpu_pCandidateLists);
#ifndef DBSCAN_VERIFY_CANDIDATES
#define DBSCAN_VERIFY_CANDIDATES(sectionname)
#endif

void	DBScanVerifyCandidates	(DBScanData* p,const char* szSectionName,unsigned int* gpu_pPointState,unsigned int* gpu_pSeedStates,unsigned int* gpu_pCandidateLists) {
	cudaError_t myLastErr; 
	unsigned int*	pPointState			= (unsigned int*)malloc(sizeof(p->pClusterIDs));
	unsigned int*	pSeedStates			= (unsigned int*)malloc(sizeof(p->pSeedStates));
	unsigned int*	pCandidateLists		= (unsigned int*)malloc(sizeof(p->pCandidateLists));
	
	// download data from vram
	CUDA_SAFE_CALL( cudaMemcpy(pPointState,gpu_pPointState,sizeof(p->pClusterIDs),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pPointState candidate-verify")
	CUDA_SAFE_CALL( cudaMemcpy(pSeedStates,gpu_pSeedStates,sizeof(p->pSeedStates),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pSeedStates candidate-verify")
	CUDA_SAFE_CALL( cudaMemcpy(pCandidateLists,gpu_pCandidateLists,sizeof(p->pCandidateLists),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pCandidateLists candidate-verify")
	
	
	// count candidates
	int c_candidates[DBSCAN_NUM_SEEDS];
	int c_candidates_last[DBSCAN_NUM_SEEDS];
	int n,iSeedID;
	for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) c_candidates[iSeedID] = 0;
	for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) c_candidates_last[iSeedID] = -1;
		
	//const unsigned int	iCandidateID	= DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(gConst_pFinalChainIDs[iSeedID]);
	//~ CUDA_SAFE_CALL( cudaMemcpyToSymbol(gConst_pFinalChainIDs, gpFinalChainIDs, sizeof(p->pFinalChainIDs))); HANDLE_ERROR("cudaMemcpy pFinalChainIDs") // const mem
		
			
	for (n=0;n<N;++n) {
		unsigned int iState = pPointState[n]; // iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
		if (       iState >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST) {
		} else if (iState >= DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_FIRST) {
			int iFoundSeedID = -1;
			for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
				unsigned int iFinalChainID = gpFinalChainIDs[iSeedID];
				unsigned int iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
				if (iCandidateID == iState) {
					++c_candidates[iSeedID];
					c_candidates_last[iSeedID] = n;
					iFoundSeedID = iSeedID;
				}
			}
			if (iFoundSeedID == -1) {
				printf("DBScanVerifyCandidates(%s) failed to find seed state=%d\n",szSectionName,iState);
				exit(0);
			}
		} else if (iState == DBSCAN_WITHIDX_CLUSTER_ID_INIT) {
		} else if (iState == DBSCAN_WITHIDX_CLUSTER_ID_NOISE) {
		}
	}
	
	
	for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
		unsigned int iSeedState_NotListedLen	= pSeedStates[iSeedID*SEEDSTATEDIM + 0];
		unsigned int iSeedState_ListLen			= pSeedStates[iSeedID*SEEDSTATEDIM + 1];
		unsigned int iOwnPointID 				= pSeedStates[iSeedID*SEEDSTATEDIM + 3]; 
		unsigned int iRecordedCount = iSeedState_NotListedLen + iSeedState_ListLen;
		unsigned int iState = 0xffffffff;
		if (iOwnPointID != INF_32) {
			iRecordedCount += 1;
			iState = pPointState[iOwnPointID];
			unsigned int iFinalChainID = gpFinalChainIDs[iSeedID];
			unsigned int iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
			if (iState != iCandidateID) {
				printf("DBScanVerifyCandidates(%s) failed prime candidate bad, iSeedID=%d n=%d state=%d\n",szSectionName,iSeedID,iOwnPointID,iState);
				exit(0);
			}
		}
		
		int iRealCount = c_candidates[iSeedID];
		
		if (iRealCount != iRecordedCount) {
			printf("DBScanVerifyCandidates(%s) failed, iSeedID=%d iOwnPointID=%d:%d lastreal=%d realcount=%d : %d=iRecordedCount=NL:%d+L:%d mismatch\n",szSectionName,iSeedID,iOwnPointID,iState,c_candidates_last[iSeedID],iRealCount,iRecordedCount,iSeedState_NotListedLen,iSeedState_ListLen);
			exit(0);
		}
	}
	
	free(pPointState);
	free(pSeedStates);
	free(pCandidateLists);
}

// ***** ***** ***** ***** ***** cpu main
//~ unsigned int	iVRamWriterUINT;
//~ #define VRAM_WRITE_UINT(p,v) { iVRamWriterUINT = v; CUDA_SAFE_CALL(cudaMemcpy(p,&iVRamWriterUINT,sizeof(iVRamWriterUINT),cudaMemcpyHostToDevice)); HANDLE_ERROR("VRAM_WRITE_UINT" #p) } // ","##v
	
void			VRAM_WRITE_UINT(unsigned int* p,unsigned int v) {
	cudaError_t myLastErr; 
	CUDA_SAFE_CALL(cudaMemcpy(p,&v,sizeof(v),cudaMemcpyHostToDevice)); HANDLE_ERROR("cudaMemcpy VRAM_WRITE_UINT");
}
unsigned int	VRAM_READ_UINT(unsigned int* p) {
	cudaError_t myLastErr; 
	unsigned int v = 0;
	CUDA_SAFE_CALL(cudaMemcpy(&v,p,sizeof(v),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy VRAM_READ_UINT"); 
	return v;
}


void DBScanAssignFinalChainID (unsigned int iSeedID,unsigned int iFinalChainID,unsigned int *gpu_pClusterIDLookup) {
	#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
	unsigned int iOldFinalChainID = gpFinalChainIDs[iSeedID];
	int oldn = iOldFinalChainID	-DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST;
	int newn = iFinalChainID	-DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST;
	if (oldn >= 0 && oldn < N) VRAM_WRITE_UINT(&gpu_pClusterIDLookup[oldn],INF_32);
	if (newn >= 0 && newn < N) VRAM_WRITE_UINT(&gpu_pClusterIDLookup[newn],iSeedID);
	#endif
	gpFinalChainIDs[iSeedID] = iFinalChainID;
}


#define VRAM_READWRITE_UNIT_TEST


void	dbscan_gpu	(DBScanData* p) {
	// a few sanity checks for parameters
	printf("dbscan_gpu SZ=%d kThreadBlockSize=%d\n",(int)SZ,(int)kThreadBlockSize);
	if ((SZ % kThreadBlockSize) != 0) {
		printf("##### ERROR, SZ(%d) must be a multiple of kThreadBlockSize(%d)\n",(int)SZ,(int)kThreadBlockSize);
		printf("##### try increasing the number of datapoints, or decreasing IO(%d)\n",(int)I0);
		// because of kernel : for (unsigned int w=0;w<SZ;w+=kThreadBlockSize,n+=kThreadBlockSize) {
		exit(0);
	}
	if ((kThreadBlockSize % 64) != 0) { 
		printf("threadblocksize should be multiple of 64.. 64 is ok if there are enough blocks running in parallel\n");
		// cuda manual, number of parallel running threads etc, recommended for register access or so
		exit(0); 
	}
	if ((N % kThreadBlockSize) != 0) { 
		printf("N=%d should be a multiple of kThreadBlockSize=%d\n",(int)N,(int)kThreadBlockSize); 
		exit(0); 
	}
	//~ if (kThreadBlockSize < DBSCAN_PARAM_MINPTS) { 
	if (kThreadBlockSize * kThreadBlockMinPtsScale < DBSCAN_PARAM_MINPTS) { 
		printf("(kThreadBlockSize * kThreadBlockMinPtsScale) must be >= DBSCAN_PARAM_MINPTS, other case not yet implemented (processing stored neightbors)\n");
		// kernel : neightbors
		exit(0); 
	}
	if (kThreadBlockSize < D) { 
		printf("kThreadBlockSize must be >= D, other case not yet implemented (reading in mainpoint)\n");
		// kernel : reading in mainpoint
		exit(0); 
	}
	if (GRIDHEIGHT != 1) { 
		printf("error, GRIDHEIGHT=1 assumed for MYID and BLOCKIDX implementation\n");
		// MYID and BLOCKIDX
		exit(0); 
	}
	if ((DBSCAN_NUM_SEEDS % kThreadBlockSize) != 0) {
		printf("DBSCAN_NUM_SEEDS(%d) must be a multiple of kThreadBlockSize(%d)\n",(int)DBSCAN_NUM_SEEDS,(int)kThreadBlockSize);
		exit(0); 
	}
	//~ if ((DBSCAN_NUM_SEEDS % (GRIDHEIGHT * kThreadBlockSize)) != 0) {
		//~ printf("DBSCAN_NUM_SEEDS(%d) must be a multiple of (GRIDHEIGHT(%d) * kThreadBlockSize(%d))\n",(int)GRIDHEIGHT,(int)kThreadBlockSize,(int)kThreadBlockSize);
		//~ // grid_size_one_thread_per_seed.x = DBSCAN_NUM_SEEDS / GRIDHEIGHT / kThreadBlockSize; UNUSED
		//~ exit(0); 
	//~ }
	
	
	
	// vars
	int	i,j,n;
	cudaError_t myLastErr; 
	#ifndef __DEVICE_EMULATION__
		//~ CUDA_SAFE_CALL(cudaSetDevice(0)); /// GT 8500
		CUDA_SAFE_CALL(cudaSetDevice(1)); /// GTX 280
	#endif
	
	
	// final cluster ids
	for (i=0;i<N;++i) giFinalClusterIDs[i] = INF_32; // giFinalClusterIDs[pFinalChainIDs[iSeedID] = iFinalChainID] = iFinalClusterID;
	
	// shortcuts
	bool	bFreshSeedsLeft = true;
	unsigned int*	pSeedStates		= p->pSeedStates; // (atomicops)
	unsigned int*	pHelperData		= p->pHelperData; // newseed
	unsigned int*	pPointState		= (unsigned int*)p->pClusterIDs; // for final evaluation
	gpFinalChainIDs					= p->pFinalChainIDs; // old : pFinalChainIDs
	gpConnectionMatrix				= p->pConnectionMatrix;
	
		
	
	
	// allocate and init gpu buffers
	#define ALLOCATE_GPU_BUFFER(type,name,datasize) type name = 0; CUDA_SAFE_CALL(cudaMalloc((void**)&name,datasize));
	ALLOCATE_GPU_BUFFER(float*			,gpu_pPointCoords,		sizeof(p->pPoints));	// N*D
	ALLOCATE_GPU_BUFFER(unsigned int*	,gpu_pPointState,		sizeof(p->pClusterIDs));// N  (outlier,candidate:chain-id,finished:chain-id)
	ALLOCATE_GPU_BUFFER(float*			,gpu_pIndex,			sizeof(p->pIndex));		// INDEX_NUM_FLOATS
	ALLOCATE_GPU_BUFFER(unsigned int*	,gpu_pSeedStates,		sizeof(p->pSeedStates));// DBSCAN_NUM_SEEDS * x  (notlisted,listlen,iNeighBorCount  : atomicops)
	//~ ALLOCATE_GPU_BUFFER(unsigned int*	,gpu_pFinalChainIDs,	sizeof(p->pFinalChainIDs));// DBSCAN_NUM_SEEDS (constant memory, values >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST)
	ALLOCATE_GPU_BUFFER(unsigned int*	,gpu_pCandidateLists,	sizeof(p->pCandidateLists));// DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN (fresh-seeds)
	ALLOCATE_GPU_BUFFER(unsigned int*	,gpu_pConnectionMatrix,	sizeof(p->pConnectionMatrix));// DBSCAN_NUM_SEEDS^2
	ALLOCATE_GPU_BUFFER(unsigned int*	,gpu_pHelperData,		sizeof(p->pHelperData));// DBSCAN_NUM_SEEDS + x
	#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
	ALLOCATE_GPU_BUFFER(unsigned int*	,gpu_pClusterIDLookup,	sizeof(unsigned int)*N);
	#else
	unsigned int*	gpu_pClusterIDLookup = 0;
	#endif
		
	
	// init vram data to zero
	CUDA_SAFE_CALL( cudaMemset(gpu_pPointState,			0, sizeof(p->pClusterIDs))); 
	CUDA_SAFE_CALL( cudaMemset(gpu_pCandidateLists,		0, sizeof(p->pCandidateLists))); 
	CUDA_SAFE_CALL( cudaMemset(gpu_pConnectionMatrix,	0, sizeof(p->pConnectionMatrix))); 
	CUDA_SAFE_CALL( cudaMemset(gpu_pHelperData,			0, sizeof(p->pHelperData))); 
	#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
	CUDA_SAFE_CALL( cudaMemset(gpu_pClusterIDLookup, 0xFF, sizeof(unsigned int)*N)); 
	printf("gpu_pClusterIDLookup[0]=0x%08x\n",(int)VRAM_READ_UINT(&gpu_pClusterIDLookup[0]));
	#endif
	
	#ifdef VRAM_READWRITE_UNIT_TEST
		printf("N=%d\n",(int)N);
		#define VRAM_READWRITE_UNIT_TEST_ONE(addr,v) VRAM_WRITE_UINT(addr,v); if (VRAM_READ_UINT(addr) != v) { printf("writefail v=%d\n",(int)v); exit(0); } else { printf("vramwriteunit ok v=%d\n",(int)v);}
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[0],0);
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[0],1);
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[0],2);
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[0],0);
		
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[5],0);
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[5],1);
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[5],2);
		VRAM_READWRITE_UNIT_TEST_ONE(&gpu_pPointState[5],0);
	#endif
	
	// choose initial seeds
	printf("start choose initial\n");
	unsigned int gNextFinalChainID = DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST;
	int iSeedPoints[DBSCAN_NUM_SEEDS];
	int iSeedID;
	for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
		// pick one n randomly
		bool bAlreadyUsed;
		do {
			n = rand() % (N-1);
			bAlreadyUsed = false;
			for (j=0;j<iSeedID;++j) if (iSeedPoints[j] == n) bAlreadyUsed = true;
		} while (bAlreadyUsed) ;
		iSeedPoints[iSeedID] = n;
		unsigned int iFinalChainID = gNextFinalChainID++;
		unsigned int iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
		//~ printf("chooseinit i=%d n=%d finalchainid=%d\n",(int)i,(int)n,(int)iFinalChainID);
		DBScanAssignFinalChainID(iSeedID,iFinalChainID,gpu_pClusterIDLookup);
		
		pSeedStates[SEEDSTATEDIM*iSeedID + 0] = 0;
		pSeedStates[SEEDSTATEDIM*iSeedID + 1] = 0;
		pSeedStates[SEEDSTATEDIM*iSeedID + 2] = 0;
		pSeedStates[SEEDSTATEDIM*iSeedID + 3] = n;
		VRAM_WRITE_UINT(&gpu_pPointState[n],iCandidateID);
		//~ printf("dbscan init : iSeedID=%d n=%d iCandidateID=%d\n",iSeedID,(int)n,iCandidateID);
	}
	
	// copy data from ram to vram
	CUDA_SAFE_CALL( cudaMemcpy(gpu_pPointCoords, 	p->pPoints, 		sizeof(p->pPoints), 		cudaMemcpyHostToDevice ));  HANDLE_ERROR("cudaMemcpy pPoints") 
	CUDA_SAFE_CALL( cudaMemcpy(gpu_pIndex,		 	p->pIndex,  		sizeof(p->pIndex),  		cudaMemcpyHostToDevice ));  HANDLE_ERROR("cudaMemcpy pIndex")
	CUDA_SAFE_CALL( cudaMemcpy(gpu_pSeedStates,		p->pSeedStates,  	sizeof(p->pSeedStates),  	cudaMemcpyHostToDevice ));  HANDLE_ERROR("cudaMemcpy pSeedStates") 
	
	printf("start copy to const vram\n");
	// copy data from ram to constant vram
	CUDA_SAFE_CALL( cudaMemcpyToSymbol(gConst_pFinalChainIDs, gpFinalChainIDs, sizeof(p->pFinalChainIDs))); HANDLE_ERROR("cudaMemcpy pFinalChainIDs") // const mem
	
	printf("start size\n");
	//~ DBSCAN_VERIFY_CANDIDATES("prepare1")
	
	// kernel setup : grid_size, block_size, mem_shared
	dim3  grid_size_many_threads_per_seed;
	dim3  grid_size_one_thread_per_seed;
	dim3  grid_size_4;
	dim3  block_size;
	unsigned int mem_shared = 0; // this is for dynamic alloc of shared mem, we alloc statically
	grid_size_many_threads_per_seed.x		= DBSCAN_NUM_SEEDS / GRIDHEIGHT;  // TODO : make sure  N is a multiple of kThreadBlockSize
	grid_size_many_threads_per_seed.y		= GRIDHEIGHT;
	grid_size_many_threads_per_seed.z		= 1;
	grid_size_one_thread_per_seed.x			= DBSCAN_NUM_SEEDS / GRIDHEIGHT / kThreadBlockSize;
	grid_size_one_thread_per_seed.y			= GRIDHEIGHT;
	grid_size_one_thread_per_seed.z			= 1;
	grid_size_4.x	= 4;
	grid_size_4.y	= 1;
	grid_size_4.z	= 1;
	block_size.x	= kThreadBlockSize;
	block_size.y	= 1; 
	block_size.z	= 1;
	#define MB(a) ((int)(a)/1024/1024)
	printf("alloc %d %d %d %d  gridsize_x=%d\n",MB(p->pPoints),MB(p->pCandidateLists),MB(p->pClusterIDs),MB(p->pConnectionMatrix),grid_size_many_threads_per_seed.x);
	
	// **** TEST NEWSEED
	
	if (TEST_KERNEL_NEWSEED) {
		printf("TEST_KERNEL_NEWSEED start\n");
		
		int iPointsLeft = N;
		for (int iTestI=0;iTestI<10000000;++iTestI) {
			unsigned int iNewSeedsNeeded = DBSCAN_NUM_SEEDS;
			
			// helper kernel : search a few new seeds (why kernel : candidate ids are in vram)
			// new seeds : sum_time="one iteration" : save index last checked and increment until next free point is found
			VRAM_WRITE_UINT(&gpu_pHelperData[DBSCAN_NUM_SEEDS+0],0); // counter
			dbscan_kernel_newseeds<<< grid_size_4, block_size, mem_shared >>>(
				gpu_pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
				gpu_pHelperData,
				iNewSeedsNeeded
			);
				
			// download gpu_pHelperData from vram
			CUDA_SAFE_CALL( cudaMemcpy(pHelperData,gpu_pHelperData,sizeof(unsigned int) * (DBSCAN_NUM_SEEDS+NEWSEEDSCAN_NUM_PARAMS),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy pHelperData readback")
			unsigned int iNewSeedsFound = min(iNewSeedsNeeded,pHelperData[DBSCAN_NUM_SEEDS+0]);
			
			// assign as noise
			iPointsLeft -= iNewSeedsFound;
			for (i=0;i<iNewSeedsFound;++i) {
				n = pHelperData[i];
				if (n < 0 || n >= N) printf("bad n:%d\n",n);
				VRAM_WRITE_UINT(&gpu_pPointState[n],DBSCAN_WITHIDX_CLUSTER_ID_NOISE);
			}
			
			// download pointstates from vram and count states
			CUDA_SAFE_CALL( cudaMemcpy(pPointState,gpu_pPointState,sizeof(p->pClusterIDs),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pPointState final download")
			int cinit = 0;
			int cnoise = 0;
			int crest = 0;
			for (n=0;n<N;++n) {
				if (pPointState[n] == DBSCAN_WITHIDX_CLUSTER_ID_INIT	) { ++cinit; continue; }
				if (pPointState[n] == DBSCAN_WITHIDX_CLUSTER_ID_NOISE	) { ++cnoise; continue; }
				++crest;
			}
			printf("iNewSeedsFound=%3d pleft=%6d cinit=%6d,cnoise=%6d,crest=%d over=%d\n",iNewSeedsFound,iPointsLeft,cinit,cnoise,crest,iPointsLeft-cinit);
			
			if (iNewSeedsFound == 0) break;
		}
		
		printf("TEST_KERNEL_NEWSEED end\n");
		return;
	}
	
	// **** TEST REFILL
	
	if (TEST_KERNEL_REFILL) {
		printf("TEST_KERNEL_REFILL start\n");
		
		// download pointstates
		CUDA_SAFE_CALL( cudaMemcpy(pPointState,gpu_pPointState,sizeof(p->pClusterIDs),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pPointState final download")
		
		// prepare test environment
		for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
			unsigned int iSetNonList	= 10;
			unsigned int iSetList		= 0;
			
			unsigned int iFinalChainID = gpFinalChainIDs[iSeedID];
			unsigned int iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
			
			VRAM_WRITE_UINT(&gpu_pSeedStates[iSeedID*SEEDSTATEDIM + 0],iSetNonList);
			VRAM_WRITE_UINT(&gpu_pSeedStates[iSeedID*SEEDSTATEDIM + 1],iSetList);
			
			// pick random points with "init" state as new unmarked
			for (i=0;i<iSetNonList;++i) {
				// pick one n randomly
				do {
					n = rand() % (N-1);
					if (pPointState[n] != DBSCAN_WITHIDX_CLUSTER_ID_INIT) continue;
					pPointState[n] = iCandidateID;
					VRAM_WRITE_UINT(&gpu_pPointState[n],iCandidateID);
					break;
				} while (true) ;
			}
		}
		
		printf("TEST_KERNEL_REFILL kernel?\n");
		// launch refill kernel
		dbscan_kernel_refill<<< grid_size_many_threads_per_seed, block_size, mem_shared >>>(
			gpu_pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
			gpu_pSeedStates,		// DBSCAN_NUM_SEEDS * 3  (real,listlen,iNeighBorCount  : atomicops)
			gpu_pCandidateLists		// DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN (fresh-seeds)
		);
		
		// init counter
		int iSeedDataCounterF[DBSCAN_NUM_SEEDS];
		int iSeedDataCounterC[DBSCAN_NUM_SEEDS];
		for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) iSeedDataCounterF[iSeedID] = 0;
		for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) iSeedDataCounterC[iSeedID] = 0;
			
		// download pointstates from vram and count states
		CUDA_SAFE_CALL( cudaMemcpy(pPointState,gpu_pPointState,sizeof(p->pClusterIDs),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pPointState final download")
		int cinit = 0;
		int cnoise = 0;
		int crest = 0;
		for (n=0;n<N;++n) {
			unsigned int iPointState = pPointState[n];
			if (iPointState == DBSCAN_WITHIDX_CLUSTER_ID_INIT	) { ++cinit; continue; }
			if (iPointState == DBSCAN_WITHIDX_CLUSTER_ID_NOISE	) { ++cnoise; continue; }
			for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
				unsigned int iFinalChainID = gpFinalChainIDs[iSeedID];
				unsigned int iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
				if (iPointState == iFinalChainID) ++iSeedDataCounterF[iSeedID];
				if (iPointState == iCandidateID ) ++iSeedDataCounterC[iSeedID];
			}
			++crest;
		}
		printf("cinit=%6d,cnoise=%6d,crest=%d\n",cinit,cnoise,crest);
				
		// download seedstate from vram
		CUDA_SAFE_CALL( cudaMemcpy(pSeedStates,gpu_pSeedStates,sizeof(p->pSeedStates),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pSeedStates readback")
		
		// analyse seeds
		for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
			unsigned int iFinalChainID = gpFinalChainIDs[iSeedID];
			unsigned int iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
			unsigned int*	piSeedState_NotListedLen	= &pSeedStates[iSeedID*SEEDSTATEDIM + 0];
			unsigned int*	piSeedState_ListLen			= &pSeedStates[iSeedID*SEEDSTATEDIM + 1];
			printf("seed:%3d nonl=%d list=%d final=%d cand=%d ",iSeedID,
				*piSeedState_NotListedLen,
				*piSeedState_ListLen,
				iSeedDataCounterF[iSeedID],
				iSeedDataCounterC[iSeedID]);
			
			for (i=0;i<*piSeedState_ListLen;++i) {
				unsigned int n = VRAM_READ_UINT(&gpu_pCandidateLists[iSeedID*CANDIDATE_LIST_MAXLEN+i]);
				unsigned int iState = pPointState[n];
				printf("%d%s,",n,(iState != iCandidateID)?"ERROR":"");
			}
			printf("\n");
		}
		
		printf("TEST_KERNEL_REFILL end\n");
		return;
	}
	
	// **** MAIN
	float	t_kernel_main			= 0.0;
	float	t_download_states		= 0.0;
	float	t_check_seedstates		= 0.0;
	float	t_finished_seeds		= 0.0;
	float	t_kernel_refill			= 0.0;
	float	t_cleanup				= 0.0;
	float	t_debug					= 0.0;
	
	printf("prepare check\n"); DBSCAN_VERIFY_CANDIDATES("prepare")
	printf("start loop\n");
	PROFILE_TIME_SECTION_START();
	int iMainRoughPointsLeft = N;
	int iOutout = 0;
	do {
		dbscan_kernel_main<<< grid_size_many_threads_per_seed, block_size, mem_shared >>>(
			gpu_pPointCoords,		// N*D
			gpu_pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
			gpu_pIndex,				// INDEX_NUM_FLOATS
			gpu_pSeedStates,		// DBSCAN_NUM_SEEDS * x  (notlised,listlen,iNeighBorCount  : atomicops)
			//~ unsigned int*	pFinalChainIDs,		// DBSCAN_NUM_SEEDS (constant memory, values >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST)
			gpu_pCandidateLists,	// DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN (fresh-seeds)
			gpu_pConnectionMatrix	// DBSCAN_NUM_SEEDS^2
			#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
			,gpu_pClusterIDLookup // [N]
			#endif
												
		);
		
		CUDA_SAFE_CALL( cudaThreadSynchronize());HANDLE_ERROR("cudaThreadSynchronize")
		PROFILE_TIME_SECTION_SUM(t_kernel_main);
		DBSCAN_VERIFY_CANDIDATES("kernel_main") PROFILE_TIME_SECTION_SUM(t_debug); 
		
		// download seedstate from vram
		CUDA_SAFE_CALL( cudaMemcpy(pSeedStates,gpu_pSeedStates,sizeof(p->pSeedStates),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pSeedStates readback")
		
		PROFILE_TIME_SECTION_SUM(t_download_states);
		
		// check seedstates
		bool	bListRefillNeeded = false;
		int		iNewSeedsNeeded = 0;
		bool	bSeedFinished[DBSCAN_NUM_SEEDS];
		for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
			bSeedFinished[iSeedID] = false;
			unsigned int*	piSeedState_NotListedLen	= &pSeedStates[iSeedID*SEEDSTATEDIM + 0];
			unsigned int*	piSeedState_ListLen			= &pSeedStates[iSeedID*SEEDSTATEDIM + 1];
			unsigned int*	piSeedState_NeighborCount	= &pSeedStates[iSeedID*SEEDSTATEDIM + 2];
			if (*piSeedState_ListLen > 0) continue;
			
			if (*piSeedState_NotListedLen > 0) {
				// refill needed
				bListRefillNeeded = true;
			} else {
				// seed finished
				bSeedFinished[iSeedID] = true;
				iNewSeedsNeeded++;
				
				// if this is the first finished seed found this round : download connection matrix and spread
				if (iNewSeedsNeeded == 1) {
					CUDA_SAFE_CALL( cudaMemcpy(gpConnectionMatrix,gpu_pConnectionMatrix,sizeof(p->pConnectionMatrix),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy pConnectionMatrix readback")
					DBScan_SpreadConnection_DebugDump("seedcheck,first");
					DBScan_SpreadConnection();
				}
				
				// cleanup
				//~ if (*piSeedState_NeighborCount >= DBSCAN_PARAM_MINPTS) { // TODO (beim da_text schreiben gesehen und entfernt) : what was this if for ? (bad if last point is noise?)
				if (GetFinalClusterIDBySeedID(iSeedID) == INF_32) { // no final id assigned yet, need to generate a new one
					unsigned int iFinalClusterID = ++giLastFinalClusterID;
					//~ printf("assign seed=%3d cid=%5d\n",iSeedID,iFinalClusterID);
					SetFinalClusterIDBySeedID(iSeedID,iFinalClusterID); // generate new cluster id and assign
					DBScan_SpreadConnection_DebugDump("seedcheck,cleanup");
					DBScan_SpreadConnection_Aux(iSeedID); // spread
				}
				//~ }
				// clear connection matrix entries for this seed
				DBScan_ClearConnection(iSeedID);
				
				// generate and assign new final chain id (upload to constant vram later)
				unsigned int iFinalChainID = gNextFinalChainID++;
				DBScanAssignFinalChainID(iSeedID,iFinalChainID,gpu_pClusterIDLookup);
			}
			//~ printf("seed %4d : %NotListedLen=%6d ListLen=%6d neighbor=%6d\n",(int)iSeedID,
				//~ (int)*piSeedState_NotListedLen,(int)*piSeedState_ListLen,(int)*piSeedState_NeighborCount);
		}
		
		
		PROFILE_TIME_SECTION_SUM(t_check_seedstates);
		DBSCAN_VERIFY_CANDIDATES("check_seedstates") PROFILE_TIME_SECTION_SUM(t_debug);
		
		
			
		#ifdef DBSCAN_ID_LOOKUP_IN_VRAM
		#ifdef DBSCAN_ID_LOOKUP_IN_VRAM_CHECK
		// check
		if (1) {
			unsigned int* temp = (unsigned int*)malloc(N*sizeof(unsigned int));
			
			// download pointstates from vram
			CUDA_SAFE_CALL( cudaMemcpy(temp,gpu_pClusterIDLookup,N*sizeof(unsigned int),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pClusterIDLookup debug download")
	
			int c = 0;
			for (int i=0;i<N;++i) {
				if (temp[i] < INF_32) {
					++c;
				}
			}
			
			if (c > DBSCAN_NUM_SEEDS) {
				printf("lookup debug : too many set %d,%d\n",c,(int)DBSCAN_NUM_SEEDS);
				exit(0);
			} 
			
			free(temp);
			PROFILE_TIME_SECTION_SUM(t_debug); 
		}
		#endif
		#endif
		
		// process finished seeds
		int iNumberOfNonRevivableSeeds = 0;
		if (iNewSeedsNeeded > 0) {
			// upload changed final ids (new chains started)
			CUDA_SAFE_CALL( cudaMemcpyToSymbol(gConst_pFinalChainIDs, gpFinalChainIDs, sizeof(p->pFinalChainIDs))); HANDLE_ERROR("cudaMemcpy gpFinalChainIDs upload2") // const mem
			
			// search new seeds in vram by iterating over gpu_pPointState
			unsigned int iNewSeedsFound = 0;
			if (bFreshSeedsLeft) {
				// helper kernel : search a few new seeds (why kernel : candidate ids are in vram)
				// new seeds : sum_time="one iteration" : save index last checked and increment until next free point is found
				VRAM_WRITE_UINT(&gpu_pHelperData[DBSCAN_NUM_SEEDS+0],0); // counter
				MY_ASSERT(NUM_THREADS_NEWSEEDSCAN == grid_size_4.x * block_size.x,"newseeds check");
				dbscan_kernel_newseeds<<< grid_size_4, block_size, mem_shared >>>(
					gpu_pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
					gpu_pHelperData,
					iNewSeedsNeeded
				);
					
				// download gpu_pHelperData from vram
				CUDA_SAFE_CALL( cudaMemcpy(pHelperData,gpu_pHelperData,sizeof(unsigned int) * (DBSCAN_NUM_SEEDS+NEWSEEDSCAN_NUM_PARAMS),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy pHelperData readback")
				iNewSeedsFound = min(iNewSeedsNeeded,pHelperData[DBSCAN_NUM_SEEDS+0]);
				
				// remember when no fresh seeds can be found anymore
				if (iNewSeedsFound < iNewSeedsNeeded) bFreshSeedsLeft = false;
			}
			
			
			// process seeds : assign new seeds or split existing ones
			for (iSeedID=0;iSeedID<DBSCAN_NUM_SEEDS;++iSeedID) {
				
				// skip seeds that still have work to do
				if (!bSeedFinished[iSeedID]) continue;
				
				// calc common helper vars, and reset seed state
				unsigned int iFinalChainID = gpFinalChainIDs[iSeedID];
				unsigned int iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
				pSeedStates[SEEDSTATEDIM*iSeedID + 0] = 0; //NotListedLen	
				pSeedStates[SEEDSTATEDIM*iSeedID + 1] = 0; //ListLen		
				pSeedStates[SEEDSTATEDIM*iSeedID + 2] = 0; //NeighborCount	
				pSeedStates[SEEDSTATEDIM*iSeedID + 3] = INF_32;
				
				// did we find enough free seeds or do we have to split existing chains ?
				if (iNewSeedsFound > 0) {
					iNewSeedsFound -= 1;
					// assign new seed
					n = pHelperData[iNewSeedsFound]; // iNewSeedN
					pSeedStates[SEEDSTATEDIM*iSeedID + 3] = n; // mark for instant use
					// pPointState : write iCandidateID, otherwise it might be marked as candidat by another seedchain
					VRAM_WRITE_UINT(&gpu_pPointState[n],iCandidateID);
				} else {
					//~ printf("split!\n");
					// split
					// choose largest existing
					unsigned int iFoundOtherSeedID = INF_32;
					unsigned int iFoundOtherListLen = 0;
					for (unsigned int iOtherSeedID=0;iOtherSeedID<DBSCAN_NUM_SEEDS;++iOtherSeedID) {	
						unsigned int iOtherListLen = pSeedStates[iOtherSeedID*SEEDSTATEDIM + 1];
						if (iFoundOtherSeedID == INF_32 || iOtherListLen > iFoundOtherListLen) {
							iFoundOtherSeedID = iOtherSeedID;
							iFoundOtherListLen = iOtherListLen;
						}
					}
					
					// split chosen
					if (iFoundOtherListLen > 1) {
						// split only the last candidate from otherseed
						unsigned int*	iSplitOriginLen = &pSeedStates[iFoundOtherSeedID*SEEDSTATEDIM + 1];
						unsigned int	iLastIndex = *iSplitOriginLen - 1;
						*iSplitOriginLen -= 1;
						unsigned int n = VRAM_READ_UINT(&gpu_pCandidateLists[iFoundOtherSeedID*CANDIDATE_LIST_MAXLEN+iLastIndex]);
						pSeedStates[SEEDSTATEDIM*iSeedID + 3] = n;
						
						// change candidate seed-assignment to avoid refill-confusion for otherseed
						VRAM_WRITE_UINT(&gpu_pPointState[n],iCandidateID);
						
						// mark split-connection 
						DBScan_SetConnection(iSeedID,iFoundOtherSeedID);
					} else {
						++iNumberOfNonRevivableSeeds;
						// split not possible algorithm nearing it's end
						//~ printf("iSeedID:%03d split not possible anymore, algorithm nearing it's end\n",(int)iSeedID);
						// listlen=0,nonlistlen=0,nextid=INF_32  signals end
					}
				}
			}
			
			// upload changed connection matrix (split)
			CUDA_SAFE_CALL( cudaMemcpy(gpu_pConnectionMatrix,gpConnectionMatrix,sizeof(p->pConnectionMatrix),cudaMemcpyHostToDevice ));  HANDLE_ERROR("cudaMemcpy pConnectionMatrix upload2") 
		
			// upload updated states to vram (changed by new cluster started, not by refill)
			CUDA_SAFE_CALL( cudaMemcpy(gpu_pSeedStates,		p->pSeedStates,  	sizeof(p->pSeedStates),  	cudaMemcpyHostToDevice ));  HANDLE_ERROR("cudaMemcpy pSeedStates upload2") 
		}
		
		PROFILE_TIME_SECTION_SUM(t_finished_seeds);
		DBSCAN_VERIFY_CANDIDATES("finished_seeds") PROFILE_TIME_SECTION_SUM(t_debug);
		
		// helper kernel : refill lists (why kernel : candidate ids are in vram)
		if (bListRefillNeeded) {
			dbscan_kernel_refill<<< grid_size_many_threads_per_seed, block_size, mem_shared >>>(
				gpu_pPointState,		// N  (outlier,candidate:chain-id,finished:chain-id)
				gpu_pSeedStates,		// DBSCAN_NUM_SEEDS * 3  (real,listlen,iNeighBorCount  : atomicops)
				gpu_pCandidateLists		// DBSCAN_NUM_SEEDS * CANDIDATE_LIST_MAXLEN (fresh-seeds)
			);
			CUDA_SAFE_CALL( cudaThreadSynchronize());HANDLE_ERROR("cudaThreadSynchronize")
		}
		
		PROFILE_TIME_SECTION_SUM(t_kernel_refill);
		DBSCAN_VERIFY_CANDIDATES("kernel_refill") PROFILE_TIME_SECTION_SUM(t_debug);
		
		// DETECT algorithm termination
		if (iNumberOfNonRevivableSeeds > 0) printf("iNumberOfNonRevivableSeeds=%d\n",(int)iNumberOfNonRevivableSeeds); 
		if (iNumberOfNonRevivableSeeds >= DBSCAN_NUM_SEEDS) {
			printf("algorithm finished\n");
			break;
		}
		//printf("DEBUG:BREAK\n"); break;
		iMainRoughPointsLeft -= DBSCAN_NUM_SEEDS - iNumberOfNonRevivableSeeds;
		if ((iOutout++ % 16) == 0 || iOutout < 16) printf("iMainRoughPointsLeft=%7d iNewSeedsNeeded=%3d\n",iMainRoughPointsLeft,iNewSeedsNeeded);
	} while (1) ;
	
	
	
	
	// cleanup
	// download pointstates from vram
	CUDA_SAFE_CALL( cudaMemcpy(pPointState,gpu_pPointState,sizeof(p->pClusterIDs),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy gpu_pPointState final download")
	
	// assign final ids, and count groups
	int counter_Init		= 0;
	int counter_Noise		= 0;
	int counter_Candidate	= 0;
	int counter_Final		= 0;
	for (n=0;n<N;++n) {
		unsigned int iState = pPointState[n]; // iCandidateID = DBSCAN_WITHIDX_CLUSTER_ID_FINAL_2_CANDIDATE(iFinalChainID);
		unsigned int iNewState = INF_32;
		if (       iState >= DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST) {
			unsigned int iFinalChainID = iState;
			unsigned int iMyIndex = iFinalChainID - DBSCAN_WITHIDX_CLUSTER_ID_FINALCHAINID_FIRST;
			MY_ASSERT(iMyIndex < N,"FinalClusterIDIndex From pPointState too high");
			iNewState = giFinalClusterIDs[iMyIndex];
			++counter_Final;
		} else if (iState >= DBSCAN_WITHIDX_CLUSTER_ID_CANDIDATE_FIRST) {
			++counter_Candidate;
		} else if (iState == DBSCAN_WITHIDX_CLUSTER_ID_INIT) {
			++counter_Init;
		} else if (iState == DBSCAN_WITHIDX_CLUSTER_ID_NOISE) {
			++counter_Noise;
		}
		pPointState[n] = iNewState;
	}
	PROFILE_TIME_SECTION_SUM(t_cleanup);
	
	printf("giDBScanClusterDoubleAssignmentCounter = %d\n",giDBScanClusterDoubleAssignmentCounter);
	
	printf("time profile:\n");
	printf("t_kernel_main			= %f\n",t_kernel_main		);
	printf("t_download_states		= %f\n",t_download_states	);
	printf("t_check_seedstates		= %f\n",t_check_seedstates	);
	printf("t_finished_seeds		= %f\n",t_finished_seeds	);
	printf("t_kernel_refill			= %f\n",t_kernel_refill		);
	printf("t_cleanup				= %f\n",t_cleanup		);
	printf("t_debug					= %f\n",t_debug		);
	
	printf("dbscan final count : Init=%d,Noise=%d,Candidate=%d,Final=%d\n",(int)counter_Init,(int)counter_Noise,(int)counter_Candidate,(int)counter_Final);
	
	sprintf(gsInfoGPUaux,"|double=%d,Init=%d,Noise=%d(%0.1f%%),Candidate=%d,Final=%d",
				(int)giDBScanClusterDoubleAssignmentCounter,(int)counter_Init,(int)counter_Noise,(float)(float(counter_Noise)/float(N)),(int)counter_Candidate,(int)counter_Final);
	
	if (counter_Init		> 0) printf("warning, count(init)>0, algorithm not finished\n");
	if (counter_Candidate	> 0) printf("warning, count(Candidate)>0, algorithm not finished\n");
	
}



