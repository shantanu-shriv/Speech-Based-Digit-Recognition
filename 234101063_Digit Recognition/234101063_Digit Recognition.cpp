// 234101063_Digit Recognition.cpp : Defines the entry point for the console application.
//


#include "stdafx.h"
#include<stdio.h>
#include<string.h>
#include<limits.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<float.h>
#include<Windows.h>

#define K 32					//LBG Codebook Size
#define DELTA 0.00001			//K-Means Parameter
#define EPSILON 0.03			 //LBG Splitting Parameter
#define UNIVERSE_SIZE 50000		//Universe Size
#define CLIP 5000				//Max value after normalizing
#define FS 320					//Frame Size
#define Q 12					//No. of cepstral coefficient
#define P 12					//No. of LPC
#define pie (22.0/7)
#define N 5						//no. of states in HMM Model
#define M 32					//Codebook Size
#define T_ 400					//Max possible no. of frames
#define TRAIN_SIZE 20			//Training Files for each utterance
#define TEST_SIZE 50			//Total Test Files if Train Size is 25

//HMM Model Variables
long double A[N + 1][N + 1];
long double	B[N + 1][M + 1]; 
long double	pi[N + 1];
long double	alpha[T_ + 1][N + 1]; 
long double	beta[T_ + 1][N + 1]; 
long double	gamma[T_ + 1][N + 1]; 
long double	delta[T_+1][N+1];
long double	xi[T_+1][N+1][N+1];
	
long double	A_bar[N + 1][N + 1];
long double	B_bar[N + 1][M + 1]; 
long double	pi_bar[N + 1];
int O[T_+1], q[T_+1], psi[T_+1][N+1], q_star[T_+1];
long double P_star=-1, P_star_dash=-1;

//Store 1 file values
int samples[50000];
//No. of frames in file
int T=160;
//Index of start frame where actual speech activity happens
int start_frame;
//Index of end frame where actual speech activity ends
int end_frame;

//Durbin's Algo variables
long double R[P+1];
long double a[P+1];
//Cepstral Coefficient
long double C[Q+1];
//Store codebook
long double reference[M+1][Q+1];
//Tokhura Weights
long double tokhuraWeight[Q+1]={0.0, 1.0, 3.0, 7.0, 13.0, 19.0, 22.0, 25.0, 33.0, 42.0, 50.0, 56.0, 61.0};
//Store energry per frame
long double energy[T_]={0};
//Universe vector
long double X[UNIVERSE_SIZE][Q];
//Universe Vector size
int LBG_M=0;
//Codebook
long double codebook[K][Q];
//Store mapping of universe with cluster
int cluster[UNIVERSE_SIZE];

/*
=============================================================SPEECH REPRESENTATTION MODULE=====================================================================================================
*/

//Normalize the data
void normalize_data(char file[100]){
	//open inputfile
	FILE* fp=fopen(file,"r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return;
	}
	int amp=0,avg=0;
	int i=0;
	int n=0;
	int min_amp=INT_MAX;
	int max_amp=INT_MIN;
	//calculate average, minimum & maximum amplitude
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		avg+=amp;
		min_amp=(amp<min_amp)?amp:min_amp;
		max_amp=(amp>max_amp)?amp:max_amp;
		n++;
	}
	avg/=n;
	T=(n-FS)/80 + 1;
	if(T>T_) T=T_;
	//update minimum & maximum amplitude after DC Shift
	min_amp-=avg;
	max_amp-=avg;
	fseek(fp,0,SEEK_SET);
	while(!feof(fp)){
		fscanf(fp,"%d",&amp);
		if(min_amp==max_amp){
			amp=0;
		}
		else{
			//handle DC Shift
			amp-=avg;
			//normalize the data
			amp=(amp*CLIP)/((max_amp>min_amp)?max_amp:(-1)*min_amp);
			//store normalized data
			samples[i++]=amp;
		}
	}
	fclose(fp);
}

//calculate energy of frame
void calculate_energy_of_frame(int frame_no){
	int sample_start_index=frame_no*80;
	energy[frame_no]=0;
	for(int i=0;i<FS;i++){
		energy[frame_no]+=samples[i+sample_start_index]*samples[i+sample_start_index];
		energy[frame_no]/=FS;
	}
}

//Calculate Max Energy of file
long double calculate_max_energy(){
	int nf=T;
	long double max_energy=DBL_MIN;
	for(int f=0;f<nf;f++){
		if(energy[f]>max_energy){
			max_energy=energy[f];
		}
	}
	return max_energy;
}

//calculate average energy of file
long double calculate_avg_energy(){
	int nf=T;
	long double avg_energy=0.0;
	for(int f=0;f<nf;f++){
		avg_energy+=energy[f];
	}
	return avg_energy/nf;
}

//mark starting and ending of speech activity
void mark_checkpoints(){
	int nf=T;
	//Calculate energy of each frame
	for(int f=0;f<nf;f++){
		calculate_energy_of_frame(f);
	}
	//Make 10% of average energy as threshold
	long double threshold_energy=calculate_avg_energy()/10;
	//long double threshold_energy=calculate_max_energy()/10;
	int isAboveThresholdStart=1;
	int isAboveThresholdEnd=1;
	start_frame=0;
	end_frame=nf-1;
	//Find start frame where speech activity starts
	for(int f=0;f<nf-5;f++){
		for(int i=0;i<5;i++){
			isAboveThresholdStart*=(energy[f+i]>threshold_energy);
		}
		if(isAboveThresholdStart){
			start_frame=((f-5) >0)?(f-5):(0);
			break;
		}
		isAboveThresholdStart=1;
	}
	//Find end frame where speech activity ends
	for(int f=nf-1;f>4;f--){
		for(int i=0;i<5;i++){
			isAboveThresholdEnd*=(energy[f-i]>threshold_energy);
		}
		if(isAboveThresholdEnd){
			end_frame=((f+5) < nf)?(f+5):(nf-1);
			break;
		}
		isAboveThresholdEnd=1;
	}
}


//Calculate ai's using Durbin's Algo
void durbinAlgo(){
	//step-0:initialize energy
	long double E=R[0];
	long double alpha[13][13];
	for(int i=1;i<=P;i++){
		double k;
		long double numerator=R[i];
		long double alphaR=0.0;
		for(int j=1;j<=(i-1);j++){
			alphaR+=alpha[j][i-1]*R[i-j];
		}
		numerator-=alphaR;
		//step-1: calculate k
		k=numerator/E;
		//step-2: calculate alpha[i][i]
		alpha[i][i]=k;
		//step-3: calculate alpha[j][i]
		for(int j=1;j<=(i-1);j++){
			alpha[j][i]=alpha[j][i-1]-(k*alpha[i-j][i-1]);
			if(i==P){
				a[j]=alpha[j][i];
			}
		}
		//step-4: update energy
		E=(1-k*k)*E;
		if(i==P){
			a[i]=alpha[i][i];
		}
	}
}

//Calculate minimun LPC Coefficients using AutoCorrelation
void autoCorrelation(int frame_no){
	long double s[FS];
	int sample_start_index=frame_no*80;
	
	//Hamming Window Function
	for(int i=0;i<FS;i++){
		long double wn=0.54-0.46*cos((2*(22.0/7.0)*i)/(FS-1));
		s[i]=wn*samples[i+sample_start_index];
	}
	
	//Calculate R0 to R12
	for(int i=0;i<=P;i++){
		long double sum=0.0;
		for(int y=0;y<=FS-1-i;y++){
			sum+=((s[y])*(s[y+i]));
		}
		R[i]=sum;
	}

	//Apply Durbin's Algorithm to calculate ai's
	durbinAlgo();
}


//Apply Cepstral Transformation to LPC to get Cepstral Coefficient
void cepstralTransformation(){
	C[0]=2.0*(log(R[0])/log(2.0));
	for(int m=1;m<=P;m++){
		C[m]=a[m];
		for(int k=1;k<m;k++){
			C[m]+=((k*C[k]*a[m-k])/m);
		}
	}
}

//Apply raised Sine window on Cepstral Coefficients
void raisedSineWindow(){
	for(int m=1;m<=P;m++){
		//raised sine window
		long double wm=(1+(Q/2)*sin(pie*m/Q));
		C[m]*=wm;
	}
}

//Store Cepstral coefficients of each frame of file
void process_universe_file(FILE* fp, char file[]){
	//normalize data
	normalize_data(file);
	int m=0;
	int nf=T;
	//repeat procedure for frames
	for(int f=0;f<nf;f++){
		//Apply autocorrelation
		autoCorrelation(f);
		//Apply cepstral Transformation
		cepstralTransformation();
		//apply raised sine window "or" liftering
		raisedSineWindow();
		for(int i=1;i<=Q;i++){
			fprintf(fp,"%Lf,",C[i]);
		}
		fprintf(fp,"\n");
		//printf(".");
	}
}


//calculate minimium Tokhura Distance
int minTokhuraDistance(long double testC[]){
	long double minD=DBL_MAX;
	int minDi=0;
	for(int i=1;i<=M;i++){
		long double distance=0.0;
		for(int j=1;j<=Q;j++){
			distance+=(tokhuraWeight[j]*(testC[j]-reference[i][j])*(testC[j]-reference[i][j]));
		}
		if(distance<minD){
			minD=distance;
			minDi=i;
		}
	}
	return minDi;
}

//Generate Observation Sequence
void generate_observation_sequence(char file[]){
	FILE* fp=fopen("o.txt","w");
	//normalize data
	normalize_data(file);
	int m=0;
	//mark starting and ending index
	mark_checkpoints();
	T=(end_frame-start_frame+1);
	int nf=T;
	//long double avg_energy=calculate_avg_energy();
	//repeat procedure for each frames
	for(int f=start_frame;f<=end_frame;f++){
		//Apply autocorrelation
		autoCorrelation(f);
		//Apply cepstral Transformation
		cepstralTransformation();
		//apply raised sine window "or" liftering
		raisedSineWindow();
		fprintf(fp,"%d ",minTokhuraDistance(C));
	}
	fprintf(fp,"\n");
	fclose(fp);
}


/*
================================================================LBG MODULE========================================================================================================
*/

//Initialize codebook with centroid of the Universe

void initialize_with_centroid(){
	long double centroid[12]={0.0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[j]+=X[i][j];
		}
	}
	for(int i=0;i<12;i++){
		centroid[i]/=LBG_M;
		codebook[0][i]=centroid[i];
	}
}



//	Calculate distance between input and codevector

long double calculate_distance(long double x[12], long double y[12]){
	long double distance=0.0;
	for(int i=0;i<12;i++){
		distance+=(tokhuraWeight[i+1]*(x[i]-y[i])*(x[i]-y[i]));
	}
	return distance;
}



//	Classification of Universe into k clusters

void nearest_neighbour(int k){
	for(int i=0;i<LBG_M;i++){
		//store minimum distance between input and codebook
		long double nn=DBL_MAX;
		//store index of codevector with which input has minimum distance
		int cluster_index;
		for(int j=0;j<k;j++){
			//compute distance between input and codevector
			long double dxy=calculate_distance(X[i],codebook[j]);
			if(dxy<=nn){
				cluster_index=j;
				nn=dxy;
			}
		}
		//classification of ith input to cluster_index cluster
		cluster[i]=cluster_index;
	}
}



//codevector updation

void codevector_update(int k){
	long double centroid[K][12]={0.0};
	//Store number of vectors in each cluster
	int n[K]={0};
	for(int i=0;i<LBG_M;i++){
		for(int j=0;j<12;j++){
			centroid[cluster[i]][j]+=X[i][j];
		}
		n[cluster[i]]++;
	}
	//Codevector Updation as Centroid of each cluster
	for(int i=0;i<k;i++){
		for(int j=0;j<12;j++){
			codebook[i][j]=centroid[i][j]/n[i];
		}
	}
}



//	Calculate overall average Distortion
//	D=(1/M)*sigma_1toM(d(x(n),y(n)))

long double calculate_distortion(){
	long double distortion=0.0;
	for(int i=0;i<LBG_M;i++){
		distortion+=calculate_distance(X[i],codebook[cluster[i]]);
	}
	distortion/=LBG_M;
	return distortion;
}



// Applying	K-Means Algorithm

void KMeans(int k){
	FILE* fp=fopen("distortion.txt","a");
	if(fp==NULL){
		printf("Error pening file!\n");
		return;
	}
	//iterative index
	int m=0;
	//store previous and current D
	long double prev_D=DBL_MAX, cur_D=DBL_MAX;
	//repeat until convergence
	do{
		//Classification
		nearest_neighbour(k);
		//Iterative index update
		m++;
		//Codevector Updation
		codevector_update(k);
		prev_D=cur_D;
		//Calculate overall avg Distortion / D
		cur_D=calculate_distortion();
		printf("Epoch %d\t:\t",m);
		printf("Distortion:%Lf\n",cur_D);
		fprintf(fp,"%Lf\n",cur_D);
	}while((prev_D-cur_D)>DELTA);//repeat until distortion difference is >delta
	//Print Updated Codebook
	printf("\nCode book of size: %d generated\n",k);
	fclose(fp);
}



//	LBG Algorithm
void LBG(){
	printf("\nLBG Algorithm:\n");
	//Start from single codebook
	int k=1;
	//Compute codevector as centroid of universe
	initialize_with_centroid();
	//repeat until desired size codebook is reached
	while(k!=K){
		//Split each codebook entry Yi to Yi(1+epsilon) & Yi(1-epsilon)
		for(int i=0;i<k;i++){
			for(int j=0;j<12;j++){
				long double Yi=codebook[i][j];
				//Yi(1+epsilon)
				codebook[i][j]=Yi-EPSILON;
				//Yi(1-epsilon)
				codebook[i+k][j]=Yi+EPSILON;
			}
		}
		//Double size of codebook
		k=k*2;
		//Call K-means with split codebook
		KMeans(k);
	}
}




/*
================================================================HMM Module START==========================================================================================================================
*/
//Initialize every variable of HMM module to zero
void initialization()
{
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			A[i][j] = 0;
		}
		for (int j = 1; j <= M; j++)
		{
			B[i][O[j]] = 0;
		}
		pi[i] = 0;
	}
	for (int i = 1; i <= T; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			alpha[i][j] = 0;
			beta[i][j] = 0;
			gamma[i][j] = 0;
		}
	}
}

//Calculate Alpha
//Forward Procedure
void calculate_alpha()
{
	//Initialization
	for (int i = 1; i <= N; i++)
	{
		alpha[1][i] = pi[i] * B[i][O[1]];
	}
	//Induction
	for (int t = 1; t < T; t++)
	{
		for (int j = 1; j <= N; j++)
		{
			long double sum = 0;
			for (int i = 1; i <= N; i++)
			{
				sum += alpha[t][i] * A[i][j];
			}
			alpha[t + 1][j] = sum * B[j][O[t + 1]];
		}
	}

}

//Solution to problem1; Evaluate model
//P(O|lambda)=sigma_i=1toN(alpha[T][i])
long double calculate_score()
{
	long double probability = 0;
	for (int i = 1; i <= N; i++)
	{
		probability += alpha[T][i];
	}
	return probability;
}

//Calculate Beta
//Backward Procedure
void calculate_beta()
{
	//Initailization
	for (int i = 1; i <= N; i++)
	{
		beta[T][i] = 1;
	}
	//Induction
	for (int t = T - 1; t >= 1; t--)
	{
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				beta[t][i] += A[i][j] * B[j][O[t + 1]] * beta[t + 1][j];
			}
		}
	}
}

//Predict most individually likely states using gamma
//One of the solution to problem 2 of HMM
void predict_state_sequence(){
	for (int t = 1; t <= T; t++)
	{
		long double max = 0;
		int index = 0;
		for (int j = 1; j <= N; j++)
		{
			if (gamma[t][j] > max)
			{
				max = gamma[t][j];
				index = j;
			}
		}
		q[t] = index;
	}
	FILE* fp=fopen("predicted_seq_gamma.txt","w");
	//printf("\nState Sequence\n");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for (int t = 1; t <= T; t++)
	{
		//printf("%d ", q[t]);
		fprintf(fp,"%4d\t",q[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);
	//printf("\n");
}

//Calculate Gamma
void calculate_gamma()
{
	for (int t = 1; t <= T; t++)
	{
		long double sum = 0;
		for (int i = 1; i <= N; i++)
		{
			sum += alpha[t][i] * beta[t][i];
		}
		for (int i = 1; i <= N; i++)
		{
			gamma[t][i] = alpha[t][i] * beta[t][i] / sum;
		}
	}
	predict_state_sequence();
}

//Solution to Problem2 Of HMM
void viterbi_algo(){
	//Initialization
	for(int i=1;i<=N;i++){
		delta[1][i]=pi[i]*B[i][O[1]];
		psi[1][i]=0;
	}
	//Recursion
	for(int t=2;t<=T;t++){
		for(int j=1;j<=N;j++){
			long double max=DBL_MIN;
			int index=0;
			for(int i=1;i<=N;i++){
				if(delta[t-1][i]*A[i][j]>max){
					max=delta[t-1][i]*A[i][j];
					index=i;
				}
			}
			delta[t][j]=max*B[j][O[t]];
			psi[t][j]=index;
		}
	}
	//Termination
	P_star=DBL_MIN;
	for(int i=1;i<=N;i++){
		if(delta[T][i]>P_star){
			P_star=delta[T][i];
			q_star[T]=i;
		}
	}
	//State Sequence (Path) Backtracking
	for(int t=T-1;t>=1;t--){
		q_star[t]=psi[t+1][q_star[t+1]];
	}

	FILE* fp=fopen("predicted_seq_viterbi.txt","w");
	for (int t = 1; t <= T; t++)
	{
		fprintf(fp,"%4d\t",O[t]);
	}
	fprintf(fp,"\n");
	for(int t=1;t<=T;t++){
		fprintf(fp,"%4d\t",q_star[t]);
	}
	fprintf(fp,"\n");
	fclose(fp);

}

//Calculate XI
void calculate_xi(){
	for(int t=1;t<T;t++){
		long double denominator=0.0;
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				denominator+=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j]);
			}
		}
		for(int i=1;i<=N;i++){
			for(int j=1;j<=N;j++){
				xi[t][i][j]=(alpha[t][i]*A[i][j]*B[j][O[t+1]]*beta[t+1][j])/denominator;
			}
		}
	}
}

//Reestimation; Solution to problem3 of HMM
void re_estimation(){
	//calculate Pi_bar
	for(int i=1;i<=N;i++){
		pi_bar[i]=gamma[1][i];
	}
	//calculate aij_bar
	for(int i=1;i<=N;i++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int j=1;j<=N;j++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T-1;t++){
				numerator+=xi[t][i][j];
				denominator+=gamma[t][i];
			}
			A_bar[i][j]=(numerator/denominator);
			if(A_bar[i][j]>max_value){
				max_value=A_bar[i][j];
				mi=j;
			}
			adjust_sum+=A_bar[i][j];
		}
		A_bar[i][mi]+=(1-adjust_sum);
	}
	//calculate bjk_bar
	for(int j=1;j<=N;j++){
		int mi=0;
		long double max_value=DBL_MIN;
		long double adjust_sum=0;
		for(int k=1;k<=M;k++){
			long double numerator=0.0, denominator=0.0;
			for(int t=1;t<=T;t++){
				//if(q_star[t]==j){
					if(O[t]==k){
						numerator+=gamma[t][j];
					}
					denominator+=gamma[t][j];
				//}
			}
			B_bar[j][k]=(numerator/denominator);
			if(B_bar[j][k]>max_value){
				max_value=B_bar[j][k];
				mi=k;
			}
			if(B_bar[j][k]<1.00e-030){
				B_bar[j][k]=1.00e-030;
				//adjust_sum+=B_bar[j][k];
			}
			adjust_sum+=B_bar[j][k];
		}
		//B_bar[j][mi]-=adjust_sum;
		B_bar[j][mi]+=(1-adjust_sum);
		//printf("maxB index:%d\nadjust_sum=%.16e\nB_bar[j][mi]=%.16e\n",mi,adjust_sum,B_bar[j][mi]);
	}
	
	//update Pi_bar
	for(int i=1;i<=N;i++){
		pi[i]=pi_bar[i];
	}
	//upadte aij_bar
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_bar[i][j];
		}
	}
	//update bjk_bar
	for(int j=1;j<=N;j++){
		for(int k=1;k<=M;k++){
			B[j][k]=B_bar[j][k];
		}
	}
}



//Store initial values of HMM model parameter into arrays
void initial_model(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"initial_model/A_%d.txt",d);
	fp = fopen(filenameA, "r");
	if (fp == NULL)
	{
		printf("Error\n");
	}

	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fscanf(fp, "%Lf ", &A[i][j]);

		}

	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"initial_model/B_%d.txt",d);
	fp = fopen(filenameB, "r");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fscanf(fp, "%Lf ", &B[i][j]);
		}

	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}

//Train HMM Model for given digit and given utterance
void train_model(int digit, int utterance){
	int m=0;

	do{
		calculate_alpha();
		calculate_beta();
		calculate_gamma();
		P_star_dash=P_star;
		viterbi_algo();
		calculate_xi();
		re_estimation();
		m++;
		printf("Digit:%d\tEpoch:%d\t=>\tP*=%e\n",digit,m,P_star);
	}while(m<60 && P_star > P_star_dash);
	
	//Store A in file
	FILE *fp;
	char filenameA[40];
	_snprintf(filenameA,40,"234101063_lambda/A_%d_%d.txt",digit,utterance);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);

	//Store B in file
	char filenameB[40];
	_snprintf(filenameB,40,"234101063_lambda/B_%d_%d.txt",digit,utterance);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}

//Calculate average model parameter for given digit
void calculate_avg_model_param(int d){
	long double A_sum[N+1][N+1]={0};
	long double B_sum[N+1][M+1]={0};
	long double temp;
	FILE* fp;
	for(int u=1;u<=25;u++){
		char filenameA[40];
		_snprintf(filenameA,40,"234101063_lambda/A_%d_%d.txt",d,u);
		fp=fopen(filenameA,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= N; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				A_sum[i][j]+=temp;
			}
		}
		fclose(fp);
		char filenameB[40];
		_snprintf(filenameB,40,"234101063_lambda/B_%d_%d.txt",d,u);
		fp=fopen(filenameB,"r");
		for (int i = 1; i <= N; i++)
		{
			for (int j = 1; j <= M; j++)
			{
				fscanf(fp, "%Lf ", &temp);
				B_sum[i][j]+=temp;
				//B[i][j]=B_sum[i][j]/25;
			}
		}
		fclose(fp);
	}
	FILE* avgfp;
	char fnameA[40];
	_snprintf(fnameA,40,"initial_model/A_%d.txt",d);
	avgfp=fopen(fnameA,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=N;j++){
			A[i][j]=A_sum[i][j]/25;
			fprintf(avgfp,"%e ", A[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
	char fnameB[40];
	_snprintf(fnameB,40,"initial_model/B_%d.txt",d);
	avgfp=fopen(fnameB,"w");
	for(int i=1;i<=N;i++){
		for(int j=1;j<=M;j++){
			B[i][j]=B_sum[i][j]/25;
			fprintf(avgfp,"%e ", B[i][j]);
		}
		fprintf(avgfp,"\n");
	}
	fclose(avgfp);
}

//Store converged Model Parameter
void store_final_lambda(int digit){
	FILE *fp;
	char filenameA[40];
	_snprintf(filenameA,40,"234101063_lambda/A_%d.txt",digit);
	fp=fopen(filenameA,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= N; j++)
		{
			fprintf(fp, "%e ", A[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
	char filenameB[40];
	_snprintf(filenameB,40,"234101063_lambda/B_%d.txt",digit);
	fp=fopen(filenameB,"w");
	for (int i = 1; i <= N; i++)
	{
		for (int j = 1; j <= M; j++)
		{
			fprintf(fp, "%e ", B[i][j]);
		}
		fprintf(fp,"\n");
	}
	fclose(fp);
}


//Store model parameters of given digit in array for test input
void processTestFile(int d){
	FILE *fp;
	initialization();
	char filenameA[40];
	_snprintf(filenameA,40,"234101063_lambda/A_%d.txt",d);
	fp=fopen(filenameA,"r");
	if (fp == NULL){
		printf("Error\n");
	}
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= N; j++){
			fscanf(fp, "%Lf ", &A[i][j]);
		}
	}
	fclose(fp);

	char filenameB[40];
	_snprintf(filenameB,40,"234101063_lambda/B_%d.txt",d);
	fp=fopen(filenameB,"r");
	for (int i = 1; i <= N; i++){
		for (int j = 1; j <= M; j++){
			fscanf(fp, "%Lf ", &B[i][j]);
		}
	}
	fclose(fp);

	fp = fopen("initial_model/pi.txt", "r");
	for (int i = 1; i <= N; i++)
	{
		fscanf(fp, "%Lf ", &pi[i]);
	}
	fclose(fp);

	fp=fopen("o.txt","r");
	for (int i = 1; i <= T; i++)
	{
		fscanf(fp, "%d\t", &O[i]);
	}
	fclose(fp);
}

//recognize digit as max probability of all digit models
int recognize_digit(){
	int rec_digit=0;
	long double max_prob=DBL_MIN;
	for(int d=0;d<=9;d++){
		processTestFile(d);
		calculate_alpha();
		long double prob=calculate_score();
		printf("P(O|lambda%d)=%e\n",d,prob);
		if(prob>max_prob){
			max_prob=prob;
			rec_digit=d;
		}
	}
	return rec_digit;
}

//Train HMM for given dataset
void training(){
	//Initialize A,B,PI as Intertia Model

	for(int d=0;d<=9;d++){
		char srcfnameA[40];
		_snprintf(srcfnameA,40,"initial/A_%d.txt",d);
		char srcfnameB[40];
		_snprintf(srcfnameB,40,"initial/B_%d.txt",d);
		char destfnameA[40];
		_snprintf(destfnameA,40,"initial_model/A_%d.txt",d);
		char destfnameB[40];
		_snprintf(destfnameB,40,"initial_model/B_%d.txt",d);
		char copyA[100];
		_snprintf(copyA,100,"copy /Y %s %s",srcfnameA,destfnameA);
		char copyB[100];
		_snprintf(copyB,100,"copy /Y %s %s",srcfnameB,destfnameB);
	}


	for(int d=0;d<=9;d++){
		for(int u=1;u<=TRAIN_SIZE;u++){
			char filename[40];
			_snprintf(filename,40,"234101063_dataset/234101063_E_%d_%d.txt",d,u);
			generate_observation_sequence(filename);
			initial_model(d);
			train_model(d,u);
		}
		calculate_avg_model_param(d);
	store_final_lambda(d);
	}
}

//Test HMM for given dataset
void testing(){
	double accuracy=0.0;
	for(int d=0;d<=9;d++){
		for(int u=21;u<=30;u++){
			char filename[40];
			_snprintf(filename,40,"234101063_dataset/234101063_E_%d_%d.txt",d,u);
			generate_observation_sequence(filename);
			printf("Digit=%d\n",d);
			int rd=recognize_digit();
			printf("Recognized Digit:%d\n",rd);
			if(rd==d){
				accuracy+=1.0;
			}
		}
	}
	printf("Accuracy:%f\n",accuracy);
}

//Live testing of Digit Recognition HMM Model
void live_test_HMM(){
	Sleep(2000);
	system("Recording_Module.exe 3 live_input/test.wav live_input/test.txt");
	generate_observation_sequence("live_input/test.txt");
	int rd=recognize_digit();
	printf("Recognized Digit:%d\n",rd);		
}

int _tmain(int argc, _TCHAR* argv[])
{
	// Generate universe
	FILE* universefp;
	universefp=fopen("universe.csv","w");
	for(int d=0;d<=9;d++){
		for(int u=1;u<=TRAIN_SIZE;u++){
			char filename[40];
			_snprintf(filename,40,"234101063_dataset/234101063_E_%d_%d.txt",d,u);
			process_universe_file(universefp,filename);
		}
	}

	//open inputfile
	FILE* fp=fopen("universe.csv","r");
	if(fp==NULL){
		printf("Error in Opening File!\n");
		return 0;
	}
	
	int i=0;
	long double c;
	while(!feof(fp)){
		fscanf(fp,"%Lf,",&c);
		X[LBG_M][i]=c;
		//Ceptral coeffecient index
		i=(i+1)%12;
		//Compute Universe vector size
		if(i==0) LBG_M++;
	}
	fclose(fp);

	//Apply LGB algorithm
	LBG();

	FILE* fp2=fopen("codebook.csv","w");
	if(fp2==NULL){
		printf("Error opening file!\n");
		return 0;
	}
	for(int i=0;i<K;i++){
		for(int j=0;j<12;j++){
			fprintf(fp2,"%Lf,",codebook[i][j]);
		}
		fprintf(fp2,"\n");
	}
	fclose(fp2);

	// Read codebook
	FILE* fp3;
	fp3=fopen("codebook.csv","r");
	if(fp3==NULL){
		printf("Error in Opening File!\n");
		return 0;
	}
	for(int i=1;i<=M;i++){
		for(int j=1;j<=Q;j++){
			fscanf(fp3,"%Lf,",&reference[i][j]);
		}
	}
	fclose(fp3);

	//Train
	training();

	int choice;

	printf("***********************************************MENU***********************************************");
	printf("\nEnter 1 to start manual testing\n");
	printf("\nEnter 2 to start live testing\n");

	printf("\nEnter Choice ");
	scanf("%d",&choice);

	if(choice == 1)
	{
	//Test
	testing();
	getchar();
	}
	else if(choice == 2)
	{
	// Live testing
	live_test_HMM();
	getchar();
	}

	getchar();
	return 0;
}


