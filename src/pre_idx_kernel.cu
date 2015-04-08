

/// kernel code : this gets executed on the GPU
/// e = fEpsilon
/// f = fEpsilon*fEpsilon
__global__ void mykernel (
	float*			pDataInD_Raw,
	float*			pDataInD_Index,
	uint4*			pDataInD_State,
	unsigned int*	pDataOutD_Atom,
	ulong2*			pDataOutD_List,
	const float e,
	const float f,
	const int	iElementOffsetInPoints,
	const int	iViewDataOffsetInPoints) {
	//~ extern __shared__ float3 shared[]; 
	__shared__ float3	shared[kThreadBlockSize];
	int*	pfSharedEndMark 	= (int*)shared;
	float*	pfSharedReadCache	= (float*)(&pfSharedEndMark[1]);

    // thread runtime environment, 1D parametrization
	unsigned int	iMyResultIndex;
	unsigned int	iMyLocalDataIndex;
	#if GRIDHEIGHT > 1
	#define BLOCKIDX  (blockIdx.x * GRIDHEIGHT + blockIdx.y)
	#else
	#define BLOCKIDX  (blockIdx.x)
	#endif
    const int		tid_global = (BLOCKIDX * blockDim.x) + threadIdx.x;   //  gridDim.x   (global refers to thread-grid here...)
	
	// read coordinates relevant for index calc (so thread-wrap bounds can be calculated)
	int iMyLocalID = iElementOffsetInPoints + tid_global; // id local within this datablock, add iViewDataOffsetInPoints to get global
	shared[threadIdx.x].x = pDataInD_Raw[iMyLocalID*D + 0];
	shared[threadIdx.x].y = pDataInD_Raw[iMyLocalID*D + 1];
	shared[threadIdx.x].z = pDataInD_Raw[iMyLocalID*D + 2];
		
	// init blocker to zero
	if (threadIdx.x == 0) pfSharedEndMark[0] = 0;
	
	// sync all threads in warp
	__syncthreads();
	
	// calculate the bounds for the wrap
	float3 vMin = shared[0];
	float3 vMax = vMin;
	float3 vCur;
	{for (int d=1;d<kThreadBlockSize;++d) { // compiler should loop-unroll
		vCur = shared[d];
		vMin.x = min(vMin.x,vCur.x);
		vMin.y = min(vMin.y,vCur.y);
		vMin.z = min(vMin.z,vCur.z);
		vMax.x = max(vMax.x,vCur.x);
		vMax.y = max(vMax.y,vCur.y);
		vMax.z = max(vMax.z,vCur.z);
	}}
	
	// add epsilon to the edges of the bounds
	vMin.x = vMin.x - e;
	vMin.y = vMin.y - e;
	vMin.z = vMin.z - e;
	vMax.x = vMax.x + e;
	vMax.y = vMax.y + e;
	vMax.z = vMax.z + e;
	
	// read remaining own coordinates
	float element[D];
	
	element[0] = shared[threadIdx.x].x;
	element[1] = shared[threadIdx.x].y;
	element[2] = shared[threadIdx.x].z;
	{for (int d=3;d<D;++d) element[d] = pDataInD_Raw[iMyLocalID*D + d];} // compiler should loop-unroll

	// read state (todo : state is equal for the whole wrap)
	uint4 	mystate = pDataInD_State[BLOCKIDX];
	int x = mystate.x;
	int y = mystate.y;
	int z = mystate.z;
	int w = mystate.w;
	float fSqDist,a;
	
	
	#define K_I_0(a) (pDataInD_Index[INDEXPOS_0(				((int)x)+(a))])
	#define K_I_1(a) (pDataInD_Index[INDEXPOS_1((int)x,			((int)y)+(a))])
	#define K_I_2(a) (pDataInD_Index[INDEXPOS_2((int)x,	(int)y,	((int)z)+(a))])
	#define K_INIT_INDEX iMyLocalDataIndex = ((int)x)*SX + ((int)y)*SY + ((int)z)*SZ - iViewDataOffsetInPoints;
	
	#define GPU_IDX1		(K_I_0(1) >= vMin.x && K_I_0(0) <= vMax.x) 
	#define GPU_IDX2		(K_I_1(1) >= vMin.y && K_I_1(0) <= vMax.y) 
	#define GPU_IDX3		(K_I_2(1) >= vMin.z && K_I_2(0) <= vMax.z) 
	#ifndef ENABLE_GPU_IDX3
	#undef	GPU_IDX3	
	#define	GPU_IDX3		(1)
	#endif
	
	//~ x = y = z = I0; // debug hack to terminate thread quickly
	
	// detect if started with finished state
	for (;x<I0;++x) if (GPU_IDX1) { 
	for (;y<I0;++y) if (GPU_IDX2) { 
	for (;z<I0;++z) if (GPU_IDX3) { K_INIT_INDEX
	for (;w<SZ;++w,++iMyLocalDataIndex) { // compiler should loop-unroll ?
		
		__syncthreads();
		if (threadIdx.x < D) pfSharedReadCache[threadIdx.x] = pDataInD_Raw[iMyLocalDataIndex*D + threadIdx.x];
		__syncthreads();
		
		// calc square distance
		fSqDist			= element[0]	- pfSharedReadCache[0];
		fSqDist			*= fSqDist;
		#define MYADD(d) a = element[d]	- pfSharedReadCache[d];	fSqDist += a*a;
		MYADD(1)
		MYADD(2)
		MYADD(3)
		MYADD(4)
		MYADD(5)
		MYADD(6)
		MYADD(7)
		MYADD(8)
		#undef MYADD
		// warning, a loop was not unrolled by the compiler here, so i had to do it manually...
			
		// RESULT LIST
		if (fSqDist < f && iMyLocalDataIndex > iMyLocalID) {
			iMyResultIndex = atomicInc(pDataOutD_Atom,0xffffffff);
			if (iMyResultIndex <= kLastValidResultIndex) {
				pDataOutD_List[iMyResultIndex] = make_ulong2(iMyLocalID,iMyLocalDataIndex); // result is in local ids, and must be transformed outside
			} else {
				// mark overflow in resultlist
				pfSharedEndMark[0] = 1;
			}
		}
		
		// wait for others
		//__syncthreads(); // todo : try without  this !
		
		//~ // abort if overlow in resultlist
		if (pfSharedEndMark[0]) {
			if (threadIdx.x == 0) pDataInD_State[BLOCKIDX] = make_uint4(x,y,z,w);
			x=kStateEndValue;
			y=kStateEndValue;
			z=kStateEndValue;
			w=kStateEndValue;
		}
	} w = 0;
	} z = 0;
	} y = 0;
	}
	
	// if we finished normally, save state as finished
	if (threadIdx.x == 0 && x < kStateEndValue) 
		pDataInD_State[BLOCKIDX] = make_uint4(kStateEndValue,kStateEndValue,kStateEndValue,kStateEndValue);
	
	#undef K_I_0
	#undef K_I_1
	#undef K_I_2
	#undef K_INIT_INDEX
	#undef GPU_IDX1
	#undef GPU_IDX2
	#undef GPU_IDX3
	#undef BLOCKIDX
}


