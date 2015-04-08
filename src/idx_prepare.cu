
// ##### ##### ##### ##### ##### index structure generation

int cmp_myindex_0	(const void* a,const void* b) { float d = ((float*)a)[0] - ((float*)b)[0]; return (d<0) ? -1 : ((d==0)?0:1); }
int cmp_myindex_1	(const void* a,const void* b) { float d = ((float*)a)[1] - ((float*)b)[1]; return (d<0) ? -1 : ((d==0)?0:1); }
int cmp_myindex_2	(const void* a,const void* b) { float d = ((float*)a)[2] - ((float*)b)[2]; return (d<0) ? -1 : ((d==0)?0:1); }
int cmp_myindex_3	(const void* a,const void* b) { float d = ((float*)a)[3] - ((float*)b)[3]; return (d<0) ? -1 : ((d==0)?0:1); }

int mymin (const int a,const int b) { return (a>b)?b:a; }
int mymax (const int a,const int b) { return (a<b)?b:a; }

/// offset and len are in elements
inline void	MySortData	(float* pData,int iNumLines,int iOffset,int iLen,int (*compare)(const void *, const void *)) {
	iLen = mymin(iLen,iNumLines - iOffset);
	if (iLen > 1) qsort(pData + iOffset*D,iLen,sizeof(float)*D,compare);
}


/// the method sorts pData and writes to pIndex
/// pData points to a buffer of size n*sizeof(float)*D
/// pIndex points to a buffer of size SIZE_INDEX
void	IndexStructure_Generate	(float* pData,float* pIndex) {
	// see also http://www.cppreference.com/stdother/qsort.html
	// see also http://en.wikipedia.org/wiki/Mergesort : c++ implementation
	// see also http://en.wikipedia.org/wiki/Introsort : quicksort + heapsort combo
	
	// 1st sort on 1st dimension
	// TODO : only the first coords ? 
	
	// sort 2nd dimension chunks by 2nd coordinate..
	// 2nd sort breaks the first sorting within chunk...
	
	int sx = SX;
	int sy = SY;
	int sz = SZ;
	int offset_x,x;
	int offset_y,y;
	int offset_z,z;
	int iLastValidIndex = N-1;
	
	// checking if the position calc is correct
	assert(INDEXPOS_0(I0)					== INDEXSTART_1-1);
	assert(INDEXPOS_1(I0-1,I0)				== INDEXSTART_2-1);
	assert(INDEXPOS_2(I0-1,I0-1,I0)			== INDEX_END-1);
	assert(sz < kStateEndValue); // avoid overflow for kernel stack (uint4)
	
	// local macros (code formatted for view with tabsize=4)
	#define DATA_ELEMENT(d,i)	((pData)[(d)+mymin(i,iLastValidIndex)*D]) // secure read with bounds-checking
	
	#define MAX_0	pIndex[INDEXPOS_0(I0		)] = DATA_ELEMENT(0,N-1);	
	#define MAX_1	pIndex[INDEXPOS_1(x,I0		)] = DATA_ELEMENT(1,offset_x+sx-1);	
	#define MAX_2	pIndex[INDEXPOS_2(x,y,I0	)] = DATA_ELEMENT(2,offset_y+sy-1);	
	
	// store k-medians after sorting the level
	#define MIN_0	pIndex[INDEXPOS_0(x			)] = DATA_ELEMENT(0,offset_x);
	#define MIN_1	pIndex[INDEXPOS_1(x,y		)] = DATA_ELEMENT(1,offset_y);
	#define MIN_2	pIndex[INDEXPOS_2(x,y,z		)] = DATA_ELEMENT(2,offset_z);
	
	#define SORT_0	MySortData(pData,N,0,N,	cmp_myindex_0); 
	#define SORT_1	MySortData(pData,N,offset_x,sx,	cmp_myindex_1);
	#define SORT_2	MySortData(pData,N,offset_y,sy,	cmp_myindex_2);
	#define SORT_3	MySortData(pData,N,offset_z,sz,	cmp_myindex_3);
	
	SORT_0 MAX_0
	printf("%f,%f,...,%f\n",DATA_ELEMENT(0,0),DATA_ELEMENT(0,1),DATA_ELEMENT(0,N-1));
	for (x=0,offset_x=0;		x<I0;++x,offset_x+=sx) { MIN_0 SORT_1 MAX_1
	for (y=0,offset_y=offset_x;	y<I0;++y,offset_y+=sy) { MIN_1 SORT_2 MAX_2
	for (z=0,offset_z=offset_y;	z<I0;++z,offset_z+=sz) { MIN_2 // SORT_3 
	}
	}
	}
}

