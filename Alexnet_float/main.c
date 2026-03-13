#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "parameter.h"
#include "functions.h"
#include "imgnet_10.h"
#include "class_labels.h"

double SW_TIME = 0;

float *ifmap1 = 0;
float *ofmap1 = 0, *ofmap1l = 0, *ofmap1p = 0, *ofmap1pp = 0;
float *ofmap2 = 0, *ofmap2l = 0, *ofmap2p = 0, *ofmap2pp = 0;
float *ofmap3 = 0, *ofmap3p = 0;
float *ofmap4 = 0, *ofmap4p = 0;
float *ofmap5 = 0, *ofmap5p = 0;
float *ofmap6 = 0;
float *ofmap7 = 0;
float *ofmap8 = 0;

float *fmap1, *bias1, *fmap2, *bias2, *fmap3, *bias3, *fmap4, *bias4, *fmap5, *bias5;
float *fmap6, *bias6, *fmap7, *bias7, *fmap8, *bias8;

// --- 추가된 부분: 바이너리 가중치 파일을 읽어오는 헬퍼 함수 ---
void load_binary_weights(const char* filepath, float* buffer, size_t num_elements) {
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL) {
        printf("Error: %s 파일을 찾을 수 없습니다. 경로를 확인하세요.\n", filepath);
        exit(1); // 파일이 없으면 강제 종료
    }
    size_t read_count = fread(buffer, sizeof(float), num_elements, fp);
    if (read_count != num_elements) {
        printf("Warning: %s 읽기 오류 (읽은 개수: %zu / 기대 개수: %zu)\n", filepath, read_count, num_elements);
    }
    fclose(fp);
}
// ----------------------------------------------------------------

void conv_ref(){
	// Layer #1
	convolution_S(ofmap1, ifmap1, fmap1, M_C1, C_C1, F_C1, E_C1, R_C1, S_C1, H_C1, W_C1, U_C1, P_C1);
	bias(ofmap1, bias1, M_C1, E_C1, F_C1);
	relu(ofmap1, M_C1, E_C1, F_C1);
	// LRN(ofmap1l, ofmap1, 5, 0.0001, 0.75, 1.0, M_C1, E_C1, F_C1); // PyTorch 구조에 맞춰 주석 처리
	pooling(ofmap1p, ofmap1, E_C1, F_C1, M_C1, E_P1, F_P1, K_P1, S_P1); // ofmap1l 대신 ofmap1을 직접 받음
	paddata(ofmap1pp, ofmap1p, M_C1, E_P1, F_P1, P_C2); 

	// Layer #2
	convolution_G(ofmap2, ofmap1pp, fmap2, M_C2, C_C2, F_C2, E_C2, R_C2, S_C2, H_C2, W_C2, 0, G_C2);
	bias(ofmap2, bias2, M_C2, E_C2, F_C2);
	relu(ofmap2, M_C2, E_C2, F_C2);
	// LRN(ofmap2l, ofmap2, 5, 0.0001, 0.75, 1.0, M_C2, E_C2, F_C2); // PyTorch 구조에 맞춰 주석 처리
	pooling(ofmap2p, ofmap2, E_C2, F_C2, M_C2, E_P2, F_P2, K_P2, S_P2); // ofmap2l 대신 ofmap2를 직접 받음
	paddata(ofmap2pp, ofmap2p, M_C2, E_P2, F_P2, P_C3);

	// Layer #3
	convolution_B(ofmap3, ofmap2pp, fmap3, M_C3, C_C3, F_C3, E_C3, R_C3, S_C3, H_C3, W_C3, 0);
	bias(ofmap3, bias3, M_C3, E_C3, F_C3);
	relu(ofmap3, M_C3, E_C3, F_C3);
	paddata(ofmap3p, ofmap3, M_C3, E_P3, E_P3, P_C4);

	// Layer #4
	convolution_G(ofmap4, ofmap3p, fmap4, M_C4, C_C4, F_C4, E_C4, R_C4, S_C4, H_C4, W_C4, 0, G_C4);
	bias(ofmap4, bias4, M_C4, E_C4, F_C4);
	relu(ofmap4, M_C4, E_C4, F_C4);
	paddata(ofmap4p, ofmap4, M_C4, E_P4, E_P4, P_C5);

	// Layer #5
	convolution_G(ofmap5, ofmap4p, fmap5, M_C5, C_C5, F_C5, E_C5, R_C5, S_C5, H_C5, W_C5, 0, G_C5);
	bias(ofmap5, bias5, M_C5, E_C5, F_C5);
	relu(ofmap5, M_C5, E_C5, F_C5);
	pooling(ofmap5p, ofmap5, E_C5, F_C5, M_C5, H_C6, W_C6, K_P3, S_P3);
}

void fc(){
	// Layer #6
	convolution_B(ofmap6, ofmap5p, fmap6, M_C6, C_C6, F_C6, E_C6, R_C6, S_C6, H_C6, W_C6, P_C6);
	bias(ofmap6, bias6, M_C6, E_C6, F_C6);
	relu(ofmap6, M_C6, E_C6, F_C6);

	// Layer #7
	convolution_B(ofmap7, ofmap6, fmap7, M_C7, C_C7, F_C7, E_C7, R_C7, S_C7, H_C7, W_C7, P_C7);
	bias(ofmap7, bias7, M_C7, E_C7, F_C7);
	relu(ofmap7, M_C7, E_C7, F_C7);

	// Layer #8
	convolution_B(ofmap8, ofmap7, fmap8, M_C8, C_C8, F_C8, E_C8, R_C8, S_C8, H_C8, W_C8, P_C8);
	bias(ofmap8, bias8, M_C8, E_C8, F_C8);
}

int main()
{
	int data_set = 0;
	int i = 0;
	float lval; int lidx;

	printf("----- AlexNet Pure C SW Emulation Start -----\n\n");

	// 1. 메모리 동적 할당
	fmap1 = (float*)calloc(M_C1 * C_C1 * R_C1 * S_C1, sizeof(float));
	fmap2 = (float*)calloc(M_C2 * C_C2 * R_C2 * S_C2, sizeof(float));
	fmap3 = (float*)calloc(M_C3 * C_C3 * R_C3 * S_C3, sizeof(float));
	fmap4 = (float*)calloc(M_C4 * C_C4 * R_C4 * S_C4, sizeof(float));
	fmap5 = (float*)calloc(M_C5 * C_C5 * R_C5 * S_C5, sizeof(float));
	fmap6 = (float*)calloc(M_C6 * C_C6 * R_C6 * S_C6, sizeof(float));
	fmap7 = (float*)calloc(M_C7 * C_C7 * R_C7 * S_C7, sizeof(float));
	fmap8 = (float*)calloc(M_C8 * C_C8 * R_C8 * S_C8, sizeof(float));

	bias1 = (float*)calloc(M_C1, sizeof(float));
	bias2 = (float*)calloc(M_C2, sizeof(float));
	bias3 = (float*)calloc(M_C3, sizeof(float));
	bias4 = (float*)calloc(M_C4, sizeof(float));
	bias5 = (float*)calloc(M_C5, sizeof(float));
	bias6 = (float*)calloc(M_C6, sizeof(float));
	bias7 = (float*)calloc(M_C7, sizeof(float));
	bias8 = (float*)calloc(M_C8, sizeof(float));

	// --- 추가된 부분: 가중치 데이터 로드 ---
	printf("가중치(Weight) 및 편향(Bias) 데이터를 불러오는 중...\n");

	load_binary_weights("alexnet_weights_bin/fmap1.bin", fmap1, M_C1 * C_C1 * R_C1 * S_C1);
	load_binary_weights("alexnet_weights_bin/bias1.bin", bias1, M_C1);
	load_binary_weights("alexnet_weights_bin/fmap2.bin", fmap2, M_C2 * C_C2 * R_C2 * S_C2);
	load_binary_weights("alexnet_weights_bin/bias2.bin", bias2, M_C2);
	load_binary_weights("alexnet_weights_bin/fmap3.bin", fmap3, M_C3 * C_C3 * R_C3 * S_C3);
	load_binary_weights("alexnet_weights_bin/bias3.bin", bias3, M_C3);
	load_binary_weights("alexnet_weights_bin/fmap4.bin", fmap4, M_C4 * C_C4 * R_C4 * S_C4);
	load_binary_weights("alexnet_weights_bin/bias4.bin", bias4, M_C4);
	load_binary_weights("alexnet_weights_bin/fmap5.bin", fmap5, M_C5 * C_C5 * R_C5 * S_C5);
	load_binary_weights("alexnet_weights_bin/bias5.bin", bias5, M_C5);
	load_binary_weights("alexnet_weights_bin/fmap6.bin", fmap6, M_C6 * C_C6 * R_C6 * S_C6);
	load_binary_weights("alexnet_weights_bin/bias6.bin", bias6, M_C6);
	load_binary_weights("alexnet_weights_bin/fmap7.bin", fmap7, M_C7 * C_C7 * R_C7 * S_C7);
	load_binary_weights("alexnet_weights_bin/bias7.bin", bias7, M_C7);
	load_binary_weights("alexnet_weights_bin/fmap8.bin", fmap8, M_C8 * C_C8 * R_C8 * S_C8);
	load_binary_weights("alexnet_weights_bin/bias8.bin", bias8, M_C8);

	printf("가중치 데이터 로드 완료!\n\n");
	// ----------------------------------------------------------------

	for (data_set = 0; data_set < _data; data_set++){ 

		printf("Processing Image %d...\n", data_set);

		ifmap1 = (float *)calloc(H_C1*W_C1*C_C1,sizeof(float));
		ofmap1 = (float *)calloc(E_C1*F_C1*M_C1,sizeof(float));
		ofmap1l= (float *)calloc(E_C1*F_C1*M_C1,sizeof(float));
		ofmap1p= (float *)calloc(E_P1*F_P1*M_C1,sizeof(float));
		ofmap1pp=(float *)calloc(H_C2*W_C2*M_C1,sizeof(float));
		ofmap2 = (float *)calloc(E_C2*F_C2*M_C2,sizeof(float));
		ofmap2l= (float *)calloc(E_C2*F_C2*M_C2,sizeof(float));
		ofmap2p= (float *)calloc(E_P2*F_P2*M_C2,sizeof(float));
		ofmap2pp=(float *)calloc(H_C3*W_C3*M_C2,sizeof(float));
		ofmap3 = (float *)calloc(E_C3*F_C3*M_C3,sizeof(float));
		ofmap3p= (float *)calloc(H_C3*W_C3*M_C3,sizeof(float));
		ofmap4 = (float *)calloc(E_C4*F_C4*M_C4,sizeof(float));
		ofmap4p= (float *)calloc(H_C5*W_C5*M_C4,sizeof(float));
		ofmap5 = (float *)calloc(E_C5*F_C5*M_C5,sizeof(float));
		ofmap5p= (float *)calloc(H_C6*W_C6*M_C5,sizeof(float));
		ofmap6 = (float *)calloc(E_C6*F_C6*M_C6,sizeof(float));
		ofmap7 = (float *)calloc(E_C7*F_C7*M_C7,sizeof(float));
		ofmap8 = (float *)calloc(E_C8*F_C8*M_C8,sizeof(float));

		// Input mapping
		for (i = 0; i<H_C1*W_C1*C_C1; i++){
			ifmap1[i] = input_image[data_set][i];
		}

		clock_t start_time = clock();

		conv_ref();
		fc();

		clock_t end_time = clock();
		double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

		lval = -100; lidx = 0;
		for (int idx = 0; idx < _class; idx++){
			if (ofmap8[idx] > lval){
				lval = ofmap8[idx];
				lidx = idx;
			}
		}

		printf("Image %1d: %3d (%2.5f) %s \n", data_set, lidx, lval, label[lidx]);
		printf("SW Execution Time: %.4f seconds\n\n", elapsed_time);

		free(ifmap1);
		free(ofmap1); free(ofmap1l); free(ofmap1p); free(ofmap1pp);
		free(ofmap2); free(ofmap2l); free(ofmap2p); free(ofmap2pp);
		free(ofmap3); free(ofmap3p);
		free(ofmap4); free(ofmap4p);
		free(ofmap5); free(ofmap5p);
		free(ofmap6);
		free(ofmap7);
		free(ofmap8);
	}

	free(fmap1); free(fmap2); free(fmap3); free(fmap4);
	free(fmap5); free(fmap6); free(fmap7); free(fmap8);
	free(bias1); free(bias2); free(bias3); free(bias4);
	free(bias5); free(bias6); free(bias7); free(bias8);

	printf("----- Benchmarking Complete -----\n");

	return 0;
}