

int		Idx_CPU		(float*	pDataIn_Raw,float* pDataIn_Index,float fEpsilon,float fSqEpsilon)  {
	ulong2* pDataOut_List_Check	= (ulong2*)malloc(DATASIZE_OUT_LIST);
	
	int iMyDataIndex;
	int c = 0;
	int iResultsCPU = 0;
	float a;
	float e = fEpsilon;
	float f = fSqEpsilon;
	float fSqDist;
	float element[D];
	for (int tid_global=0;tid_global<N;++tid_global) {
		{ for (int d=0;d<D;++d) element[d] = pDataIn_Raw[tid_global*D + d]; }
		
		#define K_I_0(a) (pDataIn_Index[INDEXPOS_0(					((int)x)+(a))])
		#define K_I_1(a) (pDataIn_Index[INDEXPOS_1((int)x,			((int)y)+(a))])
		#define K_I_2(a) (pDataIn_Index[INDEXPOS_2((int)x,	(int)y,	((int)z)+(a))])
		#define K_INIT_INDEX iMyDataIndex = ((int)x)*SX + ((int)y)*SY + ((int)z)*SZ

		int x = 0;
		int y = 0;
		int z = 0;
		int w = 0;
		#define CPU_IDX1 if (K_I_0(1) < element[0]-e) continue; if (K_I_0(0) > element[0]+e) break;
		#define CPU_IDX2 if (K_I_1(1) < element[1]-e) continue; if (K_I_1(0) > element[1]+e) break;
		#define CPU_IDX3 if (K_I_2(1) < element[2]-e) continue; if (K_I_2(0) > element[2]+e) break;
		#ifndef ENABLE_CPU_IDX3
		#undef	CPU_IDX3
		#define	CPU_IDX3
		#endif
		for (;x<I0;++x) { CPU_IDX1
		for (;y<I0;++y) { CPU_IDX2
		for (;z<I0;++z) { CPU_IDX3 K_INIT_INDEX;
		for (;w<SZ;++w,++iMyDataIndex) {
			if (iMyDataIndex <= tid_global) continue;
			
			// calc square distance
			fSqDist = 0.0f;
			{ for (int d=0;d<D;++d) { a = element[d] - pDataIn_Raw[iMyDataIndex*D + d]; fSqDist += a*a; } }

			// RESULT LIST
			if (fSqDist < f) {
				++iResultsCPU;
				pDataOut_List_Check[c].x = tid_global;
				pDataOut_List_Check[c].y = iMyDataIndex;
				if (++c >= kLastValidResultIndex) c = 0;
			}
		}w = 0;
		}z = 0;
		}y = 0;
		}
		
		#undef K_I_0
		#undef K_I_1
		#undef K_I_2
		#undef K_INIT_INDEX
		#undef CPU_IDX1
		#undef CPU_IDX2
		#undef CPU_IDX3
	}
	printf("check : with index on cpu: iNumResults=%d\n",iResultsCPU);
	return iResultsCPU;
}

