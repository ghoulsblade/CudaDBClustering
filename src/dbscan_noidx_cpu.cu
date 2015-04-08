

// ***** ***** ***** ***** ***** CPU utils

/// returns false if point was outlier, recurses to fill cluster
bool 	dbscan_cpu_aux	(int n,int iClusterID,float* pPoints,int* pClusterIDs,bool bRootCall);

RobIntList	oList_DBScan_CPUStack;

/// dbscan cpu main 
void	dbscan_cpu	(DBScanData* p) {
	int	n;
	int iClusterID = 0;
	
	float*		pPoints			= p->pPoints;
	int*		pClusterIDs		= p->pClusterIDs;
	
	RobIntList_Init(&oList_DBScan_CPUStack);
	
	for (n=0;n<N;++n) pClusterIDs[n] = DBSCAN_INDEX_INIT;
	for (n=0;n<N;++n) if (pClusterIDs[n] == DBSCAN_INDEX_INIT) { // for each unvisited point P in dataset
		if (dbscan_cpu_aux(n,iClusterID,pPoints,pClusterIDs,true)) {
			while (oList_DBScan_CPUStack.iSize > 0) dbscan_cpu_aux(RobIntList_Pop(&oList_DBScan_CPUStack),iClusterID,pPoints,pClusterIDs,true); // recursion using custom stack
			//~ printf("clusterid=%d\n",(int)iClusterID);
			printf(".");
			++iClusterID;
		}
	}
	printf("dbscan_cpu : next cluster id : %d\n",iClusterID);
}	

inline float 	dbscan_cpu_sqdist	(float* p1,float* p2) {
	float a,res = 0.0;
	for (int d=0;d<D;++d) { a = p2[d*N] - p1[d*N]; res += a*a; }
	return res;
}

bool 	dbscan_cpu_aux	(int n,int iClusterID,float* pPoints,int* pClusterIDs,bool bRootCall) {
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
	float*	pMyPoint	= &pPoints[n];
	bool	bRes 		= false;
	int 	m,i;
	for (m=0;m<N;++m) if (dbscan_cpu_sqdist(pMyPoint,&pPoints[m]) <= DBSCAN_PARAM_SQEPSILON) {
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
				for (i=0;i<DBSCAN_PARAM_MINPTS-1;++i) dbscan_cpu_aux(iNeighBors[i],iClusterID,pPoints,pClusterIDs,false);
			}
			// recurse to M
			dbscan_cpu_aux(m,iClusterID,pPoints,pClusterIDs,false);
		}
	}
	if (!bRes) pClusterIDs[n] = DBSCAN_INDEX_OUTLIER; // not enough, mark as NOISE/outlier
	return bRes;
}



