
#define BLOCKIDX			(blockIdx.x * GRIDHEIGHT + blockIdx.y)
#define MYID				(BLOCKIDX * blockDim.x + threadIdx.x)  
#define POSITIVE_INFINITY 	(1.0E+14) // todo : system constant to explicitly set infinity flag ? gpu/cpu dependent infinity model (wrap instead of +-) ?
#define GRIDHEIGHT 			32 // TODO 
#define	kThreadBlockSize	(64) // x*64:,64,128,192,256 (see cuda-guide page 65)
#define	KMEANS_MAXITER		(200)
#define	kCostDiffEpsilon	(0.00001) // TODO ?


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


/// kernel code : this gets executed on the GPU
__global__ static void kmeans_kernel_findclosest (
	float*			pIn_Points,
	float*			pIn_Medoids,
	float*			pOut_ClosestMedoidSqDist,
	unsigned int*	pOut_ClosestMedoidIndex) {
	
	// read own coordinates
	float element[D];
	{for (int d=0;d<D;++d) element[d] = pIn_Points[MYID*D + d];} // compiler should loop-unroll
	//~ atomicInc(pOut_MyDebugCounter,0x7fffFFFF);
	
	// test which medoid is closest
	float a;
	float fSqDist;
	float fClosestMedoidIndexSqDist = POSITIVE_INFINITY;
	int iClosestMedoidIndex = 0;
	for (int i=0;i<K;++i) {
		fSqDist = 0.0;
		// pIn_Medoids : all thread access the same data at the same time
		{for (int d=0;d<D;++d) { a = pIn_Medoids[i*D + d] - element[d]; fSqDist += a*a; }} // compiler should loop-unroll
		if (fSqDist < fClosestMedoidIndexSqDist) { fClosestMedoidIndexSqDist = fSqDist; iClosestMedoidIndex = i; }
	}
	
	// writeout result
	pOut_ClosestMedoidSqDist[MYID] = fClosestMedoidIndexSqDist; // for cost-calc
	pOut_ClosestMedoidIndex[ MYID] = iClosestMedoidIndex;
}


typedef struct {
	float			pPoints				[N*D];
	float			pInitialMedoids		[K*D]; /// same for cpu and gpu, so that time compare is equal
	float			pMedoids			[K*D];
	float			pMedoidCounts		[K];
	float			pClosestMedoidSqDist[N];
	unsigned int	pClosestMedoidIndex	[N];
} KMeansData;


char gsInfoGPU[512] = "";
char gsInfoCPU[512] = "";

void	kmeans_gpu	(KMeansData* p) {
	unsigned int	i,k,d,n;
	float	fLastCost = POSITIVE_INFINITY;
	float	fCost;
	float	fCostDiff;
	cudaError_t myLastErr; 
	
	
	#ifndef __DEVICE_EMULATION__
		//~ CUDA_SAFE_CALL(cudaSetDevice(0)); /// GT 8500
		CUDA_SAFE_CALL(cudaSetDevice(1)); /// GTX 280
	#endif
	
	float*			pPoints					= p->pPoints;
	float*			pMedoids				= p->pMedoids;
	float*			pMedoidCounts			= p->pMedoidCounts;
	float*			pClosestMedoidSqDist	= p->pClosestMedoidSqDist;
	unsigned int*	pClosestMedoidIndex		= p->pClosestMedoidIndex;
	
	// initial medoid assignment (same for cpu and gpu)
	for (k=0;k<K;++k) {
		for (d=0;d<D;++d) pMedoids[k*D+d] = p->pInitialMedoids[k*D+d];
	}
	
	// allocate and init gpu buffers
	float*			pGPUIn_Data;
	float*			pGPUIn_Medoids;
	float*			pGPUOut_ClosestMedoidSqDist;
	unsigned int*	pGPUOut_ClosestMedoidIndex;
	unsigned int*	pGPUOut_MyDebugCounter;
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_Data,					sizeof(p->pPoints)		));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUIn_Medoids,				sizeof(p->pMedoids)		));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUOut_ClosestMedoidSqDist,	sizeof(p->pClosestMedoidSqDist)	));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUOut_ClosestMedoidIndex,	sizeof(p->pClosestMedoidIndex)	));
	CUDA_SAFE_CALL( cudaMalloc( (void**) &pGPUOut_MyDebugCounter,		sizeof(unsigned int)	));
	
	
	// copy data from ram to vram
	CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_Data,		p->pPoints,		sizeof(p->pPoints),		cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy init points")
	CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_Medoids,	p->pMedoids,	sizeof(p->pMedoids),	cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy init medoids")
	
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
	
	
	float t_kernel	= 0;
	float t_download	= 0;
	float t_med_init	= 0;
	float t_med_calc	= 0;
	float t_med_div	= 0;
	float t_upload	= 0;
	PROFILE_TIME_SECTION_START();
	
	
	for (i=1;i<KMEANS_MAXITER;++i) { 
		CUDA_SAFE_CALL( cudaMemset(pGPUOut_MyDebugCounter,			0, sizeof(unsigned int))); 
		kmeans_kernel_findclosest<<< grid_size, block_size, mem_shared >>>(
			pGPUIn_Data,
			pGPUIn_Medoids,
			pGPUOut_ClosestMedoidSqDist,
			pGPUOut_ClosestMedoidIndex);	
		
		CUDA_SAFE_CALL( cudaThreadSynchronize());HANDLE_ERROR("cudaThreadSynchronize")
		
		PROFILE_TIME_SECTION_SUM(t_kernel);
		
		// read back results
		CUDA_SAFE_CALL( cudaMemcpy(pClosestMedoidSqDist,pGPUOut_ClosestMedoidSqDist,sizeof(p->pClosestMedoidSqDist),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy ClosestMedoidSqDist")
		CUDA_SAFE_CALL( cudaMemcpy(pClosestMedoidIndex, pGPUOut_ClosestMedoidIndex, sizeof(p->pClosestMedoidIndex), cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy ClosestMedoidIndex")

		PROFILE_TIME_SECTION_SUM(t_download);
		
		// calc avg for new medoids, and cost
		fCost = 0.0;
		for (k=0;k<K;++k) pMedoidCounts[k] = 0.0;
		for (k=0;k<K;++k) for (d=0;d<D;++d) pMedoids[k*D+d] = 0.0;
			
		PROFILE_TIME_SECTION_SUM(t_med_init);
		
		for (n=0;n<N;++n) {
			k = pClosestMedoidIndex[n];
			pMedoidCounts[k]++;
			//~ printf("sqdist %5d %i %f\n",n,k,pClosestMedoidSqDist[n]);
			fCost += pClosestMedoidSqDist[n]; // calc cost
			for (d=0;d<D;++d) pMedoids[k*D+d] += pPoints[n*D+d];
		}
		
		PROFILE_TIME_SECTION_SUM(t_med_calc);
		
		//~ for (k=0;k<K;++k) printf("pMedoidCounts[%d] %d\n",k,(int)pMedoidCounts[k]);
		for (k=0;k<K;++k) for (d=0;d<D;++d) pMedoids[k*D+d] /= pMedoidCounts[k];
		
		PROFILE_TIME_SECTION_SUM(t_med_div);
		
		// copy data from ram to vram
		CUDA_SAFE_CALL( cudaMemcpy(pGPUIn_Medoids,	p->pMedoids,	sizeof(p->pMedoids),	cudaMemcpyHostToDevice	));  HANDLE_ERROR("cudaMemcpy init medoids")
			
		PROFILE_TIME_SECTION_SUM(t_upload);
		
		// read back results
		unsigned int iMyDebugCounter = 0;
		//~ CUDA_SAFE_CALL( cudaMemcpy(&iMyDebugCounter,pGPUOut_MyDebugCounter,sizeof(unsigned int),cudaMemcpyDeviceToHost)); HANDLE_ERROR("cudaMemcpy ClosestMedoidSqDist")
			
		// calc cost difference
		fCostDiff = fabs(fLastCost - fCost);
		fLastCost = fCost;
		printf("iteration %d, cost=%f diff=%f iMyDebugCounter=%d\n",i,fCost,fCostDiff,iMyDebugCounter);
		
		// detect convergence
		if (i > 1 && fCostDiff < kCostDiffEpsilon) break;  
	}
	sprintf(gsInfoGPU,"iter=%d",i);
	
	printf("time profile:\n");
	printf("t_kernel	= %f\n",t_kernel		);
	printf("t_download	= %f\n",t_download		);
	printf("t_med_init	= %f\n",t_med_init		);
	printf("t_med_calc	= %f\n",t_med_calc		);
	printf("t_med_div	= %f\n",t_med_div		);
	printf("t_upload	= %f\n",t_upload		);
	
	// release GPU-memory
	CUDA_SAFE_CALL(cudaFree(pGPUIn_Data));
	CUDA_SAFE_CALL(cudaFree(pGPUIn_Medoids));
	CUDA_SAFE_CALL(cudaFree(pGPUOut_ClosestMedoidSqDist));
	CUDA_SAFE_CALL(cudaFree(pGPUOut_ClosestMedoidIndex));
	CUDA_SAFE_CALL(cudaFree(pGPUOut_MyDebugCounter));
}


void	kmeans_cpu	(KMeansData* p) {
	unsigned int	i,k,d,n;
	float	fLastCost = POSITIVE_INFINITY;
	float	fCost;
	float	fCostDiff;
	
	float*			pCurPoint;
	float*			pPoints					= p->pPoints;
	float*			pMedoids				= p->pMedoids;
	float*			pMedoidCounts			= p->pMedoidCounts;
	float*			pClosestMedoidSqDist	= p->pClosestMedoidSqDist;
	unsigned int*	pClosestMedoidIndex		= p->pClosestMedoidIndex;
	
	// initial medoid assignment (same for cpu and gpu)
	for (k=0;k<K;++k) {
		for (d=0;d<D;++d) pMedoids[k*D+d] = p->pInitialMedoids[k*D+d];
	}
	
	float t_pointloop	= 0;
	float t_med_init	= 0;
	float t_med_calc	= 0;
	float t_med_div		= 0;
	PROFILE_TIME_SECTION_START();
	
	
	for (i=1;i<KMEANS_MAXITER;++i) {
		unsigned int iMyDebugCounter2 = 0;
		// assign points
		for (n=0;n<N;++n) {
			pCurPoint = &pPoints[n*D];
			//~ ++iMyDebugCounter2;
			
			// test which medoid is closest
			float a;
			float fSqDist;
			float fClosestMedoidIndexSqDist = POSITIVE_INFINITY;
			int iClosestMedoidIndex = 0;
			for (int j=0;j<K;++j) {
				fSqDist = 0.0;
				{for (int d=0;d<D;++d) { a = pMedoids[j*D + d] - pCurPoint[d]; fSqDist += a*a; }} // compiler should loop-unroll
				if (fSqDist < fClosestMedoidIndexSqDist) { fClosestMedoidIndexSqDist = fSqDist; iClosestMedoidIndex = j; }
			}
			
			pClosestMedoidSqDist[n] = fClosestMedoidIndexSqDist; // for cost-calc
			pClosestMedoidIndex[n] = iClosestMedoidIndex;
		}
		
		PROFILE_TIME_SECTION_SUM(t_pointloop);
		
		// calc avg for new medoids, and cost
		fCost = 0.0;
		for (k=0;k<K;++k) pMedoidCounts[k] = 0.0;
		for (k=0;k<K;++k) for (d=0;d<D;++d) pMedoids[k*D+d] = 0.0;
			
		PROFILE_TIME_SECTION_SUM(t_med_init);
		
		for (n=0;n<N;++n) {
			k = pClosestMedoidIndex[n];
			pMedoidCounts[k]++;
			//~ printf("sqdist %5d %i %f\n",n,k,pClosestMedoidSqDist[n]);
			fCost += pClosestMedoidSqDist[n]; // calc cost
			for (d=0;d<D;++d) pMedoids[k*D+d] += pPoints[n*D+d];
		}
		
		PROFILE_TIME_SECTION_SUM(t_med_calc);
		
		//~ for (k=0;k<K;++k) printf("pMedoidCounts[%d] %d\n",k,(int)pMedoidCounts[k]);
		for (k=0;k<K;++k) for (d=0;d<D;++d) pMedoids[k*D+d] /= pMedoidCounts[k];
		
		PROFILE_TIME_SECTION_SUM(t_med_div);
		
		// calc cost difference
		fCostDiff = fabs(fLastCost - fCost);
		fLastCost = fCost;
		printf("iteration %d, cost=%f diff=%f iMyDebugCounter2=%d\n",i,fCost,fCostDiff,iMyDebugCounter2);
		
		// detect convergence
		if (i > 1 && fCostDiff < kCostDiffEpsilon) break;  
	};
	sprintf(gsInfoCPU,"iter=%d",i);
	
	printf("time profile:\n");
	printf("t_pointloop		= %f\n",t_pointloop		);
	printf("t_med_init		= %f\n",t_med_init		);
	printf("t_med_calc		= %f\n",t_med_calc		);
	printf("t_med_div		= %f\n",t_med_div		);
}

// idea : problem : n atomic ops at the same time = bad
// idea : solution ? : half items ?  use threadblock shared mem ? difficult to coordinate
// idea : problem : can't just add datapoints together, need to be datapoints of the same medoid
// idea : TODO : calc number of points per medoid in medoid step ?
// idea : problem : AtomicAdd geht nur mit integers, nicht mit floats
// idea : problem : memoryaccess ist teuer -> erst alle speichern dann alle wieder laden schlecht ?
// idea : es macht sinn anzunehmen dass n >> k	

// idea : a) von den datenpunkten aus per semaphore auf die medoiden, problem : semaphore,atomic-int
// idea : b) von den medoiden aus, problem : k mal alle durchgehen, speicherzugriffe
// idea : c) liste für jeden medoid, problem : speicherplatz nicht abschätzbar im vorraus
// idea : d) summieren auf cpu
// idea : e) 

/*
__global__ static void kmeans_kernel_sum (
	float*			pIn_Points,
	unsigned int*	pOut_ClosestMedoidIndex,
	unsigned int*	pOut_MedoidCounter,
	float*			pOut_MedoidPosSum) {
	
	// read own coordinates
	float element[D];
	{for (int d=0;d<D;++d) element[d] = pIn_Points[MYID*D + d];} // compiler should loop-unroll
	int iClosestMedoidIndex = 0;
	
}
*/
