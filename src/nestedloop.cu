
int		NestedLoop	(float*	pDataIn_Raw,float fSqEpsilon)  {
	int iNumResults = 0;
	//float e = fEpsilon;
	float f = fSqEpsilon;
	float a,fSqDist;
	int i,j;
	int maxi=0;
	int maxc=0;
	float element[D];
	//~ int checkmap[N];
	//~ for (i=0;i<N;++i) checkmap[i] = 0;
	for (i=0;i<N;++i) {
		{ for (int d=0;d<D;++d) element[d] = pDataIn_Raw[i*D + d]; }
		for (j=i+1;j<N;++j) {
			// calc square distance
			fSqDist = 0.0f;
			{ for (int d=0;d<D;++d) { a = element[d] - pDataIn_Raw[j*D + d]; fSqDist += a*a; } }

			// RESULT LIST
			if (fSqDist < f) {
				++iNumResults;
				//~ checkmap[i]++;
				//~ checkmap[j]++;
			}
		}
	}
	//~ for (i=0;i<N;++i) if (checkmap[i] > maxc) { maxi=i; maxc=checkmap[i]; }
	printf("check:iNumResults=%d maxc=%d for i=%d\n",iNumResults,maxc,maxi);
	return iNumResults;
}

