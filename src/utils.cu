
unsigned int timer = 0;
unsigned int timer_initialised = 0;
void	RobStartTimer	() {
	if (timer_initialised) {
		CUT_SAFE_CALL( cutResetTimer( timer ) );
	} else {
		timer = 0;
		CUT_SAFE_CALL( cutCreateTimer( &timer ) );
		timer_initialised = 1;
	}
	CUT_SAFE_CALL( cutStartTimer( timer));
}
// returns time since last RobStartTimer() in seconds as float
float	RobStopTimer	() {
    CUT_SAFE_CALL( cutStopTimer( timer));
	float res = cutGetTimerValue(timer)/1000.0f;
	return (res > 0) ? res : 0;
}

unsigned int gMyGlobalTimer = 0;
void	InitGlobalTimer () {
	gMyGlobalTimer = 0;
	CUT_SAFE_CALL( cutCreateTimer( &gMyGlobalTimer ) );
	cutStartTimer(gMyGlobalTimer);
}

inline float	GetGlobalTimerValue () { return cutGetTimerValue(gMyGlobalTimer); }

float gfSectionStart = 0.0;
float	ProfileTimerStartSection	() { // returns time since last ProfileTimerStartSection() call
	float t = GetGlobalTimerValue();
	float res = t - gfSectionStart;
	gfSectionStart = t;
	return res;
}

#define BUFFERSIZE_FILEREAD (1024*64)

/*
#define DATA_CLUST_RANDOM
#define DATA_CLUST_RANDOM_NUM_CLUST 6
#define DATA_CLUST_RANDOM_RAD_MIN 1
#define DATA_CLUST_RANDOM_RAD_MAX 2
#define DATA_CLUST_RANDOM_CENTER_MIN 2
#define DATA_CLUST_RANDOM_CENTER_MAX 6
*/


float frand	(float fmymin,float fmymax) { return fmymin + (fmymax-fmymin)*float(rand())/float(RAND_MAX); }

#ifdef DATA_CLUST_RANDOM
int		GenerateData_ClusterRandom	(float* pData) {
	int k,d;
	float* myClustCenter = 0;
	myClustCenter = (float*)malloc(sizeof(float)*(DATA_CLUST_RANDOM_NUM_CLUST)*(D));
	float myClustRad[DATA_CLUST_RANDOM_NUM_CLUST];
	for (k=0;k<DATA_CLUST_RANDOM_NUM_CLUST;++k) {
		for (d=0;d<D;++d) myClustCenter[k*D+d] = frand(	DATA_CLUST_RANDOM_CENTER_MIN,
														DATA_CLUST_RANDOM_CENTER_MAX);
		myClustRad[k] = frand(	DATA_CLUST_RANDOM_RAD_MIN,
								DATA_CLUST_RANDOM_RAD_MAX);
	}
	
	for (k=0;k<N;++k) {
		int iClust = rand() % (DATA_CLUST_RANDOM_NUM_CLUST);
		float r = frand(0,1);
		r = r * r; // square so the distribution is not linear.. todo : gauss here ?
		for (d=0;d<D;++d) {
			float x = r * frand(-1,1);
			DATAPOINT(pData,k,d) = myClustCenter[iClust*D+d] + myClustRad[iClust]*x;
		}
	}
	free(myClustCenter);
	return 0;
}
#endif

/// writes iNumLines*D*sizeof(float) bytes of data to pData
int		ReadTextData	(const char* szFilePath,float* pData) {
	#ifdef DATA_CLUST_RANDOM
	printf("ReadTextData: redirecting to GenerateData_ClusterRandom\n");
	return GenerateData_ClusterRandom(pData);
	#else
	#ifdef DATA_PURE_RANDOM
	printf("ReadTextData: DATA_PURE_RANDOM set\n");
	#else
	printf("ReadTextData: reading file %s\n",szFilePath);
	#endif
	
	float* pDataBase = pData;
	FILE* fp = 0;	
    //~ szFilePath = cutFindFilePath(szDataFileName, argv[0]);
	//~ printf("filepath=%s\n",szFilePath);
    //~ cutFree(szDataFilePath); 
	#ifndef DATA_PURE_RANDOM
	fp = fopen(szFilePath,"r");
	if (!fp) { printf("couldn't open datafile %s\n",szFilePath); exit(1); }
	char mybuf[BUFFERSIZE_FILEREAD+1];
	#endif
	int iRealNumLines = 0;
	
	//~ if (1) { // just test out file : print first line and count remaining lines
		//~ while (fgets(mybuf,BUFFERSIZE_FILEREAD,fp)) {
			//~ if (iRealNumLines < 5) printf("line %d:%s\n",iRealNumLines+1,mybuf);
			//~ ++iRealNumLines;
		//~ }
		//~ printf("total lines : %d\n",iRealNumLines);
		//~ exit(0);
	//~ }
	
	#ifndef DATA_PURE_RANDOM
	bool bFileEnd = false;
	float fMin[D];
	float fMax[D];
	float f;
	#endif
	int iLinesFilledWithRandomData = 0;
	for (int k=0;k<N;++k) {
		#ifdef DATA_PURE_RANDOM
			++iLinesFilledWithRandomData;
			for (int i=0;i<D;++i) pData[i] = DATA_PURE_RANDOM_MIN + (DATA_PURE_RANDOM_MAX-DATA_PURE_RANDOM_MIN)*float(rand())/float(RAND_MAX);
		#else
			if (!bFileEnd && fgets(mybuf,BUFFERSIZE_FILEREAD,fp)) {
				++iRealNumLines;
				const char* a = mybuf;
				for (int i=0;i<D;++i) {
					if (sscanf(a,"%f",&f) < 1) { 
						f = 0;
						printf("warning:line %d failed to read coordinate %d/%d\n",k+1,i,D);
					}
					pData[i] = f;
					if (iRealNumLines == 1 || fMin[i] > f) fMin[i] = f;
					if (iRealNumLines == 1 || fMax[i] < f) fMax[i] = f;
					a += strcspn(a," \t");
					a += strspn(a," \t");
				}
			} else {
				++iLinesFilledWithRandomData;
				bFileEnd = true;
				for (int i=0;i<D;++i) pData[i] = fMin[i] + (fMax[i]-fMin[i])*float(rand())/float(RAND_MAX);
				// avoid this if possible, by adjusting the number of lines in the file, or by playing with I0 and N
			}
		#endif
		pData += D;
	}
	fclose(fp);
	
	#ifdef NORMALIZE_DATA_TO_MINMAX
	pData = pDataBase;
	int i,k;
	if (1) {
		float fDiff[D];
		float fNormScale = DATA_PURE_RANDOM_MAX-DATA_PURE_RANDOM_MIN;
		for (i=0;i<D;++i) fDiff[i] = fMax[i]-fMin[i];
		printf("normalizing and scaling to range [%f;%f]\n",(float)DATA_PURE_RANDOM_MIN,(float)DATA_PURE_RANDOM_MAX);
		for (i=0;i<D;++i) printf("dim=%d min=%f,max=%f,diff=%f\n",i,fMin[i],fMax[i],fDiff[i]);
		for (k=0;k<N;++k) {
			for (i=0;i<D;++i) {
				float oldval = pData[i];
				pData[i] = (fDiff[i] > 0.0) ? (DATA_PURE_RANDOM_MIN + fNormScale * (pData[i]-fMin[i]) / fDiff[i]) : 0.0;
				//~ printf("dataread line=%d i=%d value=%f oldval=%f\n",k,i,(float)pData[i],oldval);
			}
			pData += D;
		}
	}
	#endif
	
	printf("ReadTextData %s : %d lines of real data, added %d lines of random data\n",szFilePath,iRealNumLines,iLinesFilledWithRandomData);
	
	
	if (iRealNumLines != N) { printf("WARNING ! iRealNumLines=%d does not match the hardcoded N=%d\n",iRealNumLines,N); }
	return iRealNumLines;
	//for (i=0;i<10;++i) printf("file[%d,0]=%f\n",i,pfData[i*D+0]);
	#endif
}


void	PrintDeviceInfos	() {
	int count = 0;
	CUDA_SAFE_CALL( cudaGetDeviceCount(&count));
	CUDA_SAFE_CALL( cudaSetDevice(count-1) ); // take the last device (e.g. NOT the main display device)
	for (int i=0;i<count;++i) {
		struct cudaDeviceProp prop;
		CUDA_SAFE_CALL( cudaGetDeviceProperties(&prop,i));
		printf("##### ##### ##### ##### #####");
		printf("device %d\n",i);
		printf("name : %s\n",prop.name);
		#define PRINTDEVINFO_I(field) if (prop.field >= 1024) printf("%10dk %s\n",(int)prop.field/1024,#field); else printf("%10d %s\n",(int)prop.field,#field);
		PRINTDEVINFO_I(totalGlobalMem)
		PRINTDEVINFO_I(sharedMemPerBlock)
		PRINTDEVINFO_I(regsPerBlock)
		PRINTDEVINFO_I(warpSize)
		PRINTDEVINFO_I(memPitch)
		PRINTDEVINFO_I(maxThreadsPerBlock)
		PRINTDEVINFO_I(maxThreadsDim[0])
		PRINTDEVINFO_I(maxThreadsDim[1])
		PRINTDEVINFO_I(maxThreadsDim[2])
		PRINTDEVINFO_I(maxGridSize[0])
		PRINTDEVINFO_I(maxGridSize[1])
		PRINTDEVINFO_I(maxGridSize[2])
		PRINTDEVINFO_I(totalConstMem)
		PRINTDEVINFO_I(major)
		PRINTDEVINFO_I(minor)
		PRINTDEVINFO_I(clockRate)
		PRINTDEVINFO_I(textureAlignment)
		/*
		struct cudaDeviceProp
		{
		  char   name[256];
		  size_t totalGlobalMem;
		  size_t sharedMemPerBlock;
		  int    regsPerBlock;
		  int    warpSize;
		  size_t memPitch;
		  int    maxThreadsPerBlock;
		  int    maxThreadsDim[3];
		  int    maxGridSize[3]; 
		  size_t totalConstMem; 
		  int    major;
		  int    minor;
		  int    clockRate;
		  size_t textureAlignment;
		};
		*/
	}
}


const char* GetDataSourceName () {
	#ifdef DATA_PURE_RANDOM
	return "unirand";
	#else
		#ifdef DATA_CLUST_RANDOM
		return "clustrand";
		#else
		return "file";
		#endif
	#endif
}


void	RobWriteReportLine 	(const char* szReportFilePath,const char* szReport) {
	FILE* fp = fopen(szReportFilePath,"a");
	if (!fp) { printf("ERROR:RobWriteReportLine, couldn't write to report file\n"); return; }
		
	// time-text
	char myTimeText[256] = "";
	time_t mytime;
	time(&mytime);
	strftime(myTimeText,255,"%Y.%m.%d_%H:%M:%S",localtime(&mytime));
	
	
	char myline[512];
	sprintf(myline,"%s N=%d D=%d DataSource=%s %s\n",myTimeText,(int)N,(int)D,GetDataSourceName(),szReport);
	 
	printf("%s",myline);
	fprintf(fp,"%s",myline);
	fclose(fp);  
}

inline float3 min3 (const float3 a,const float3 b) { return make_float3(min(a.x,b.x),min(a.y,b.y),min(a.z,b.z)); }
inline float3 max3 (const float3 a,const float3 b) { return make_float3(max(a.x,b.x),max(a.y,b.y),max(a.z,b.z)); }







