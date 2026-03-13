#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>  // 추가됨!

#include "parameter.h"
#include "functions.h"
#include "imgnet_10.h"
#include "class_labels.h"

double SW_TIME = 0;

// 모든 float 포인터를 8비트 꼬마 상자(int8_t)로 변환
int8_t *ifmap1 = 0;
int8_t *ofmap1 = 0, *ofmap1l = 0, *ofmap1p = 0, *ofmap1pp = 0;
int8_t *ofmap2 = 0, *ofmap2l = 0, *ofmap2p = 0, *ofmap2pp = 0;
int8_t *ofmap3 = 0, *ofmap3p = 0;
int8_t *ofmap4 = 0, *ofmap4p = 0;
int8_t *ofmap5 = 0, *ofmap5p = 0;
int8_t *ofmap6 = 0;
int8_t *ofmap7 = 0;
int8_t *ofmap8 = 0;

int8_t *fmap1, *bias1, *fmap2, *bias2, *fmap3, *bias3, *fmap4, *bias4, *fmap5, *bias5;
int8_t *fmap6, *bias6, *fmap7, *bias7, *fmap8, *bias8;

// 파이썬에서 뽑은 32비트 float을 읽어오자마자 8비트 정수로 변환하여 꼬마 상자에 담는 함수
void load_quantized_weights(const char* filepath, int8_t* buffer, size_t num_elements) {
    FILE *fp = fopen(filepath, "rb");
    if (fp == NULL) {
        printf("Error: %s 파일을 찾을 수 없습니다.\n", filepath);
        exit(1);
    }
    
    // 1. 일단 원래 크기(float)대로 임시로 쫙 읽어옴
    float *temp_float = (float*)malloc(num_elements * sizeof(float));
    fread(temp_float, sizeof(float), num_elements, fp);
    fclose(fp);

    // 2. 16(2^4)을 곱해서 소수점을 정수부로 끌어올린 후, int8_t에 욱여넣음
    for(size_t i = 0; i < num_elements; i++) {
        int32_t q_val = (int32_t)roundf(temp_float[i]);
        if (q_val > 127) q_val = 127;
        if (q_val < -128) q_val = -128;
        buffer[i] = (int8_t)q_val;
    }
    free(temp_float);
}

void conv_ref(){
	// Layer #1
	convolution_S(ofmap1, ifmap1, fmap1, M_C1, C_C1, F_C1, E_C1, R_C1, S_C1, H_C1, W_C1, U_C1, P_C1);
	bias(ofmap1, bias1, M_C1, E_C1, F_C1);
	relu(ofmap1, M_C1, E_C1, F_C1);
	pooling(ofmap1p, ofmap1, E_C1, F_C1, M_C1, E_P1, F_P1, K_P1, S_P1);
	paddata(ofmap1pp, ofmap1p, M_C1, E_P1, F_P1, P_C2); 

	// Layer #2
	convolution_G(ofmap2, ofmap1pp, fmap2, M_C2, C_C2, F_C2, E_C2, R_C2, S_C2, H_C2, W_C2, 0, G_C2);
	bias(ofmap2, bias2, M_C2, E_C2, F_C2);
	relu(ofmap2, M_C2, E_C2, F_C2);
	pooling(ofmap2p, ofmap2, E_C2, F_C2, M_C2, E_P2, F_P2, K_P2, S_P2);
	paddata(ofmap2pp, ofmap2p, M_C2, E_P2, F_P2, P_C3);

	// Layer #3
	convolution_B(ofmap3, ofmap2pp, fmap3, M_C3, C_C3, F_C3, E_C3, R_C3, S_C3, H_C3, W_C3, 0);
	bias(ofmap3, bias3, M_C3, E_C3, F_C3);
	relu(ofmap3, M_C3, E_C3, F_C3);
	paddata(ofmap3p, ofmap3, M_C3, E_P3, E_P3, P_C4);

	// Layer #4 (수정했던 M_C4, P_C5 반영됨!)
	convolution_G(ofmap4, ofmap3p, fmap4, M_C4, C_C4, F_C4, E_C4, R_C4, S_C4, H_C4, W_C4, 0, G_C4);
	bias(ofmap4, bias4, M_C4, E_C4, F_C4);
	relu(ofmap4, M_C4, E_C4, F_C4);
	paddata(ofmap4p, ofmap4, M_C4, E_C4, F_C4, P_C5);

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
	int8_t lval; int lidx; // 이것도 int8_t 로 변경

	printf("----- AlexNet 8-BIT INT EMULATION Start -----\n\n");

	// float (4바이트) -> int8_t (1바이트)로 사이즈 확 줄어듦!
	fmap1 = (int8_t*)calloc(M_C1 * C_C1 * R_C1 * S_C1, sizeof(int8_t));
	fmap2 = (int8_t*)calloc(M_C2 * C_C2 * R_C2 * S_C2, sizeof(int8_t));
	fmap3 = (int8_t*)calloc(M_C3 * C_C3 * R_C3 * S_C3, sizeof(int8_t));
	fmap4 = (int8_t*)calloc(M_C4 * C_C4 * R_C4 * S_C4, sizeof(int8_t));
	fmap5 = (int8_t*)calloc(M_C5 * C_C5 * R_C5 * S_C5, sizeof(int8_t));
	fmap6 = (int8_t*)calloc(M_C6 * C_C6 * R_C6 * S_C6, sizeof(int8_t));
	fmap7 = (int8_t*)calloc(M_C7 * C_C7 * R_C7 * S_C7, sizeof(int8_t));
	fmap8 = (int8_t*)calloc(M_C8 * C_C8 * R_C8 * S_C8, sizeof(int8_t));

	bias1 = (int8_t*)calloc(M_C1, sizeof(int8_t));
	bias2 = (int8_t*)calloc(M_C2, sizeof(int8_t));
	bias3 = (int8_t*)calloc(M_C3, sizeof(int8_t));
	bias4 = (int8_t*)calloc(M_C4, sizeof(int8_t));
	bias5 = (int8_t*)calloc(M_C5, sizeof(int8_t));
	bias6 = (int8_t*)calloc(M_C6, sizeof(int8_t));
	bias7 = (int8_t*)calloc(M_C7, sizeof(int8_t));
	bias8 = (int8_t*)calloc(M_C8, sizeof(int8_t));

	printf("가중치를 8비트 정수(Q4)로 깎아서 불러오는 중...\n");
	load_quantized_weights("alexnet_weights_bin/fmap1.bin", fmap1, M_C1 * C_C1 * R_C1 * S_C1);
	load_quantized_weights("alexnet_weights_bin/bias1.bin", bias1, M_C1);
	load_quantized_weights("alexnet_weights_bin/fmap2.bin", fmap2, M_C2 * C_C2 * R_C2 * S_C2);
	load_quantized_weights("alexnet_weights_bin/bias2.bin", bias2, M_C2);
	load_quantized_weights("alexnet_weights_bin/fmap3.bin", fmap3, M_C3 * C_C3 * R_C3 * S_C3);
	load_quantized_weights("alexnet_weights_bin/bias3.bin", bias3, M_C3);
	load_quantized_weights("alexnet_weights_bin/fmap4.bin", fmap4, M_C4 * C_C4 * R_C4 * S_C4);
	load_quantized_weights("alexnet_weights_bin/bias4.bin", bias4, M_C4);
	load_quantized_weights("alexnet_weights_bin/fmap5.bin", fmap5, M_C5 * C_C5 * R_C5 * S_C5);
	load_quantized_weights("alexnet_weights_bin/bias5.bin", bias5, M_C5);
	load_quantized_weights("alexnet_weights_bin/fmap6.bin", fmap6, M_C6 * C_C6 * R_C6 * S_C6);
	load_quantized_weights("alexnet_weights_bin/bias6.bin", bias6, M_C6);
	load_quantized_weights("alexnet_weights_bin/fmap7.bin", fmap7, M_C7 * C_C7 * R_C7 * S_C7);
	load_quantized_weights("alexnet_weights_bin/bias7.bin", bias7, M_C7);
	load_quantized_weights("alexnet_weights_bin/fmap8.bin", fmap8, M_C8 * C_C8 * R_C8 * S_C8);
	load_quantized_weights("alexnet_weights_bin/bias8.bin", bias8, M_C8);
	printf("가중치 데이터 로드 완료!\n\n");

	for (data_set = 0; data_set < _data; data_set++){ 

		printf("Processing Image %d...\n", data_set);

		ifmap1 = (int8_t *)calloc(H_C1*W_C1*C_C1, sizeof(int8_t));
		ofmap1 = (int8_t *)calloc(E_C1*F_C1*M_C1, sizeof(int8_t));
		ofmap1p= (int8_t *)calloc(E_P1*F_P1*M_C1, sizeof(int8_t));
		ofmap1pp=(int8_t *)calloc(H_C2*W_C2*M_C1, sizeof(int8_t));
		ofmap2 = (int8_t *)calloc(E_C2*F_C2*M_C2, sizeof(int8_t));
		ofmap2p= (int8_t *)calloc(E_P2*F_P2*M_C2, sizeof(int8_t));
		ofmap2pp=(int8_t *)calloc(H_C3*W_C3*M_C2, sizeof(int8_t));
		ofmap3 = (int8_t *)calloc(E_C3*F_C3*M_C3, sizeof(int8_t));
		ofmap3p= (int8_t *)calloc(H_C3*W_C3*M_C3, sizeof(int8_t));
		ofmap4 = (int8_t *)calloc(E_C4*F_C4*M_C4, sizeof(int8_t));
		ofmap4p= (int8_t *)calloc(H_C5*W_C5*M_C4, sizeof(int8_t));
		ofmap5 = (int8_t *)calloc(E_C5*F_C5*M_C5, sizeof(int8_t));
		ofmap5p= (int8_t *)calloc(H_C6*W_C6*M_C5, sizeof(int8_t));
		ofmap6 = (int8_t *)calloc(E_C6*F_C6*M_C6, sizeof(int8_t));
		ofmap7 = (int8_t *)calloc(E_C7*F_C7*M_C7, sizeof(int8_t));
		ofmap8 = (int8_t *)calloc(E_C8*F_C8*M_C8, sizeof(int8_t));

		// 입력 이미지도 8비트 정수로 변환해서 넣음
		for (i = 0; i<H_C1*W_C1*C_C1; i++){
			int32_t q_val = (int32_t)roundf(input_image[data_set][i]);
			if (q_val > 127) q_val = 127;
			if (q_val < -128) q_val = -128;
			ifmap1[i] = (int8_t)q_val;
		}

		clock_t start_time = clock();
		conv_ref();
		fc();
		clock_t end_time = clock();
		double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

		lval = -128; lidx = 0;
		for (int idx = 0; idx < _class; idx++){
			if (ofmap8[idx] > lval){
				lval = ofmap8[idx];
				lidx = idx;
			}
		}

		printf("Image %1d: %3d (Score: %d) %s \n", data_set, lidx, lval, label[lidx]);
		printf("8-BIT Execution Time: %.4f seconds\n\n", elapsed_time);

		free(ifmap1); free(ofmap1); free(ofmap1p); free(ofmap1pp);
		free(ofmap2); free(ofmap2p); free(ofmap2pp);
		free(ofmap3); free(ofmap3p); free(ofmap4); free(ofmap4p);
		free(ofmap5); free(ofmap5p); free(ofmap6); free(ofmap7); free(ofmap8);
	}

	free(fmap1); free(fmap2); free(fmap3); free(fmap4);
	free(fmap5); free(fmap6); free(fmap7); free(fmap8);
	free(bias1); free(bias2); free(bias3); free(bias4);
	free(bias5); free(bias6); free(bias7); free(bias8);

	printf("----- Benchmarking Complete -----\n");
	return 0;
}