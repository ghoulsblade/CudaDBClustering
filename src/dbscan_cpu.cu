

char gsInfoCPU[512] = "";

// ***** ***** ***** ***** ***** CPU utils

/// returns false if point was outlier, recurses to fill cluster
bool 	dbscan_cpu_aux	(int n,int iClusterID,DBScanData* p,bool bRootCall);

RobIntList	oList_DBScan_CPUStack;

/// dbscan cpu main 
void	dbscan_cpu	(DBScanData* p) {
	int	n;
	int iClusterID = 0;
	
	int*		pClusterIDs		= p->pClusterIDs;
	
	RobIntList_Init(&oList_DBScan_CPUStack);
	
	for (n=0;n<N;++n) pClusterIDs[n] = DBSCAN_INDEX_INIT;
	for (n=0;n<N;++n) if (pClusterIDs[n] == DBSCAN_INDEX_INIT) { // for each unvisited point P in dataset
		if (dbscan_cpu_aux(n,iClusterID,p,true)) {
			while (oList_DBScan_CPUStack.iSize > 0) dbscan_cpu_aux(RobIntList_Pop(&oList_DBScan_CPUStack),iClusterID,p,true); // recursion using custom stack
			//~ printf("clusterid=%d\n",(int)iClusterID);
			printf(".");
			++iClusterID;
		}
	}
	sprintf(gsInfoCPU,"clust=%d",(int)iClusterID);
	printf("dbscan_cpu : next cluster id : %d\n",iClusterID);
}	

inline float 	dbscan_cpu_sqdist	(float* element,float* p2) {
	float a,res = 0.0;
	#ifdef DATAORG_STRUCT_OF_STREAMS
	for (int d=0;d<D;++d) { a = p2[d*N] - element[d]; res += a*a; } /// struct of streams
	#else
	for (int d=0;d<D;++d) { a = p2[d] - element[d]; res += a*a; } // the default/intuitive dataorg
	#endif
	return res;
}

bool 	dbscan_cpu_aux	(int n,int iClusterID,DBScanData* p,bool bRootCall) {
	float*		pIndex			= p->pIndex;
	float*		pPoints			= p->pPoints;
	int*		pClusterIDs		= p->pClusterIDs;
	
	if (!bRootCall) {
		if (pClusterIDs[n] != DBSCAN_INDEX_INIT) return false;
			
		pClusterIDs[n] = DBSCAN_INDEX_MARKED;
		// old, direct recursion, overflows call-stack for large data
		RobIntList_Push(&oList_DBScan_CPUStack,n); 
		return false; 
	}
	
	
	if (pClusterIDs[n] != DBSCAN_INDEX_INIT && 
		pClusterIDs[n] != DBSCAN_INDEX_MARKED) return false;
	
	int		iNeighBors[DBSCAN_PARAM_MINPTS-1];
	int		iNeighBorCount = 0;
	float	element[D]; for (int d=0;d<D;++d) element[d] = DATAPOINT(pPoints,n,d);
	bool	bRes 		= false;
	int 	m,i;
	const float e = DBSCAN_PARAM_EPSILON;
	
	#define K_I_0(a) (pIndex[INDEXPOS_0(				((int)x)+(a))])
	#define K_I_1(a) (pIndex[INDEXPOS_1((int)x,			((int)y)+(a))])
	#define K_I_2(a) (pIndex[INDEXPOS_2((int)x,	(int)y,	((int)z)+(a))])
	#define K_INIT_INDEX m = ((int)x)*SX + ((int)y)*SY + ((int)z)*SZ;

	#define CPU_IDX1 if (K_I_0(1) < element[0]-e) continue; if (K_I_0(0) > element[0]+e) break;
	#define CPU_IDX2 if (K_I_1(1) < element[1]-e) continue; if (K_I_1(0) > element[1]+e) break;
	#define CPU_IDX3 if (K_I_2(1) < element[2]-e) continue; if (K_I_2(0) > element[2]+e) break;
	
	#ifdef MYTEST_DISABLE_INDEX 
	for (int x=0;x<I0;++x) { 
	for (int y=0;y<I0;++y) { 
	for (int z=0;z<I0;++z) { K_INIT_INDEX
	for (int w=0;w<SZ;++w,++m) {
	#else
	for (int x=0;x<I0;++x) { CPU_IDX1
	for (int y=0;y<I0;++y) { CPU_IDX2
	for (int z=0;z<I0;++z) { CPU_IDX3 K_INIT_INDEX
	for (int w=0;w<SZ;++w,++m) {
	#endif
	if (dbscan_cpu_sqdist(element,&DATAPOINT(pPoints,m,0)) <= DBSCAN_PARAM_SQEPSILON) {
		++iNeighBorCount;
		if (iNeighBorCount < DBSCAN_PARAM_MINPTS) {
			// not yet sure if n is a corepoints, so save the neighboors for now
			iNeighBors[iNeighBorCount-1] = m;
		} else {
			// n is a corepoint
			if (iNeighBorCount == DBSCAN_PARAM_MINPTS) {
				// freshly determined as corepoint, init and recurse to neighbors
				bRes = true;
				pClusterIDs[n] = iClusterID;
				// recurse to the first few marked neighbors
				for (i=0;i<DBSCAN_PARAM_MINPTS-1;++i) dbscan_cpu_aux(iNeighBors[i],iClusterID,p,false);
			}
			// recurse to M
			dbscan_cpu_aux(m,iClusterID,p,false);
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
	#undef CPU_IDX1
	#undef CPU_IDX2
	#undef CPU_IDX3
	
	if (!bRes) pClusterIDs[n] = DBSCAN_INDEX_OUTLIER; // not enough, mark as NOISE/outlier
	return bRes;
}



