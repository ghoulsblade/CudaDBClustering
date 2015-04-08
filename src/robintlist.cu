
typedef struct {
	int		iSize;
	int		iAllocatedSize;
	int*	pData;
} RobIntList;
void	RobIntList_Init		(RobIntList* p) {
	p->iSize = 0;
	p->iAllocatedSize = 0;
	p->pData = 0;
}
void	RobIntList_Destroy	(RobIntList* p) {
	p->iSize = 0;
	p->iAllocatedSize = 0;
	if (p->pData) free(p->pData); p->pData = 0;
}
void	RobIntList_Push		(RobIntList* p,int iValue) {
	p->iSize += 1;
	if (p->iAllocatedSize < p->iSize) {
		if (p->iAllocatedSize < 32) 
				p->iAllocatedSize = 32;
		else	p->iAllocatedSize *= 2;
		printf("RobIntList_Push, allocsize=%d MB\n",(int)p->iAllocatedSize*4/1024/1024);
		p->pData = (int*)realloc(p->pData,p->iAllocatedSize*sizeof(int));
	}
	p->pData[p->iSize-1] = iValue;
}
int		RobIntList_Pop		(RobIntList* p) {
	if (p->iSize <= 0) return 0;
	p->iSize -= 1;
	return p->pData[p->iSize];
}
