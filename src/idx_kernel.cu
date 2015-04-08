
/// kernel code : this gets executed on the GPU
/// e = fEpsilon
/// f = fEpsilon*fEpsilon
__global__ static void mykernel (
	float*			pDataInD_Raw,
	float*			pDataInD_Index,
	uint4*			pDataInD_State,
	float3*			pDataInD_Bounds,
	unsigned int*	pDataOutD_Atom,
	ulong2*			pDataOutD_List,
	const float e,
	const float f,
	const int	iElementOffsetInPoints,
	const int	iViewDataOffsetInPoints) {
	
	#ifdef ENABLE_GPU_WRITEOUTLIST_BIG_RESULT_LIST
	__shared__ int		iSharedEndMark;
	#endif
	__shared__ float	pfSharedReadCache[D];

    // thread runtime environment, 1D parametrization
	unsigned int	iMyResultIndex;
	int				iMyLocalDataIndex;
	#define BLOCKIDX  (blockIdx.x * GRIDHEIGHT + blockIdx.y)
	
	// read state
	uint4 	mystate = pDataInD_State[BLOCKIDX];
	int x = mystate.x;
	int y = mystate.y;
	int z = mystate.z;
	int w = mystate.w;
	float fSqDist,a;
		
	//~ if (x < kStateEndValue) {
		// id local within this datablock, add iViewDataOffsetInPoints to get global
		const int iMyLocalID			= iElementOffsetInPoints + (BLOCKIDX * blockDim.x) + threadIdx.x; 
		const int iBlockFirstLocalID	= iElementOffsetInPoints + (BLOCKIDX * blockDim.x); 
		
		// read thread-block bounds
		#ifndef GPU_NESTED_LOOP
		float3 vMin = pDataInD_Bounds[BLOCKIDX*2 + 0];
		float3 vMax = pDataInD_Bounds[BLOCKIDX*2 + 1];
		#endif
			
		// read own coordinates
		float element[D];
		{for (int d=0;d<D;++d) element[d] = pDataInD_Raw[iMyLocalID*D + d];} // compiler should loop-unroll

		// init blocker to zero
		#ifdef ENABLE_GPU_WRITEOUTLIST_BIG_RESULT_LIST
		if (threadIdx.x == 0) iSharedEndMark = 0; 
		#endif
		
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
		
		//~ int iRunCount = 2;
		
		//~ x = y = z = I0; // debug hack to terminate thread quickly
		
		// the conditions used for looping must be independant from threadidx, so they are the same inside a threadblock (branching)
		
		#ifdef GPU_NESTED_LOOP
		if (x <= iBlockFirstLocalID) x = iBlockFirstLocalID+1; // only scan my "half" (same branching inside threadblock)
		for (iMyLocalDataIndex=x;iMyLocalDataIndex<N;++iMyLocalDataIndex) { /// WARNING ! only works on single GPU (needs complete data), and if iViewDataOffsetInPoints == 0
		#else
		for (;x<I0;++x) if (GPU_IDX1) {
		for (;y<I0;++y) if (GPU_IDX2) { 
		for (;z<I0;++z) if (GPU_IDX3) {
							/*
							if (--iRunCount == 0) {
								if (threadIdx.x == 0) { 
									pDataInD_State[BLOCKIDX] = make_uint4(x,y,z,w); 
									atomicInc(pDataOutD_Atom+1,0x3fffffff);
								}
								x=kStateEndValue;
								y=kStateEndValue;
								z=kStateEndValue;
								w=kStateEndValue;
							}
							*/
							K_INIT_INDEX
		for (;w<SZ;++w,++iMyLocalDataIndex) { // compiler should loop-unroll ?
		#endif
			//~ __syncthreads();
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
			
			#if (D == 9)
			MYADD(8)
			#endif
			#if (D == 16)
			MYADD(8)
			MYADD(9)
			MYADD(10)
			MYADD(11)
			MYADD(12)
			MYADD(13)
			MYADD(14)
			MYADD(15)
			#endif
			#undef MYADD
			// warning, a loop was not unrolled by the compiler here, so i had to do it manually...
			
			// RESULT LIST
			if (fSqDist < f && iMyLocalDataIndex > iMyLocalID) {
				iMyResultIndex = atomicInc(pDataOutD_Atom,0x3fffffff);
				#ifdef ENABLE_GPU_WRITEOUTLIST
				if (iMyResultIndex <= kLastValidResultIndex) {
					pDataOutD_List[iMyResultIndex] = make_ulong2(iMyLocalID,iMyLocalDataIndex); // result is in local ids, and must be transformed outside
				} else {
					// mark overflow in resultlist
					#ifdef ENABLE_GPU_WRITEOUTLIST_BIG_RESULT_LIST
					iSharedEndMark = 1;
					#endif
				}
				#endif
			}
			
			// wait for others
			__syncthreads(); // todo : try without  this !
			
			// abort if overlow in resultlist
			#ifdef ENABLE_GPU_WRITEOUTLIST_BIG_RESULT_LIST
			if (iSharedEndMark) {
				if (threadIdx.x == 0) pDataInD_State[BLOCKIDX] = make_uint4(x,y,z,w);
				x=kStateEndValue;
				y=kStateEndValue;
				z=kStateEndValue;
				w=kStateEndValue;
			}
			//~ __syncthreads(); // todo : try without  this !
			#endif
		#ifndef GPU_NESTED_LOOP
		} w = 0;
		} z = 0;
		} y = 0;
		} 
		#else
		} x = iMyLocalDataIndex;
		#endif
		
		
		if (threadIdx.x == 0 && x < kStateEndValue) { 
			pDataInD_State[BLOCKIDX] = make_uint4(kStateEndValue,kStateEndValue,kStateEndValue,kStateEndValue); 
		}
								
		//~ if (threadIdx.x == 0 && iRunCount > 0) pDataOutD_Atom[1] = 1;
		
		// if we finished normally, save state as finished
		//~ if (threadIdx.x == 0 && x < kStateEndValue) 
			//~ pDataInD_State[BLOCKIDX] = make_uint4(kStateEndValue,kStateEndValue,kStateEndValue,kStateEndValue);
		
		#undef K_I_0
		#undef K_I_1
		#undef K_I_2
		#undef K_INIT_INDEX
		#undef GPU_IDX1
		#undef GPU_IDX2
		#undef GPU_IDX3
		#undef BLOCKIDX
	//~ }
}


