#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdint.h>  // 8비트 정수(int8_t)를 사용하기 위한 필수 헤더
#include <math.h>

#define MY_MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MY_MAX(X,Y) ((X) > (Y) ? (X) : (Y))

// 우리가 사용할 Q-Format (소수점 4자리 = 2^4 = 16)
#define Q_FRAC 4

// paddata (패딩은 0을 덧대는 거라 연산 없이 그대로 8비트 복사)
void paddata(int8_t *ofmap, int8_t *ifmap, int C, int H, int W, int padding) {
	unsigned int c = 0, e = 0, f = 0;
	unsigned int E = H+2*padding, F = W+2*padding;
	for (c = 0; c < C; c++) {
		for (e = 0; e < E; e++)		{
			for (f = 0; f < F; f++)			{
				if (f< padding || f >= W+padding)
					ofmap[(c*E + e)*F + f] = 0;
				else if (e< padding || e >= H+padding)
					ofmap[(c*E + e)*F + f] = 0;
				else
					ofmap[(c*E + e)*F + f] = ifmap[(c*H + e-padding)*W + f-padding];
			}
		}
	}
}

// 일반 컨볼루션 연산 (Layer 3, 6, 7, 8)
void convolution_B(int8_t *ofmap, int8_t *ifmap, int8_t *fmap, int M, int C, int F, int E, int R, int S, int H, int W, int padding) {
	int m, c, r, s, e, f;
	for (m = 0; m < M; m++) {
		for (e = 0; e < E; e++) {
			for (f = 0; f < F; f++) {
				int32_t sum = 0; // [핵심 1] 터지지 않게 32비트 임시 창고 준비
				for (c = 0; c < C; c++) {
					for (r = 0; r < R; r++) {
						for (s = 0; s < S; s++) {
							int fmap_idx = m*C*R*S + c*R*S + r*S + s;
							int ifmap_idx = c*H*W + (e+r)*W + (f+s);
							// 8비트 * 8비트 = 16비트를 32비트 창고에 누적
							sum += (int32_t)fmap[fmap_idx] * (int32_t)ifmap[ifmap_idx];
						}
					}
				}
				// [핵심 2] 소수점 복구(>> 4) 후 오버플로우 잘라내기(Saturation)
				sum >>= Q_FRAC; 
				if (sum > 127) sum = 127;
				else if (sum < -128) sum = -128;
				
				ofmap[m*E*F + e*F + f] = (int8_t)sum; // 다시 8비트 상자에 쏙!
			}
		}
	}
}

// 스트라이드(U)가 있는 컨볼루션 (Layer 1 전용)
void convolution_S(int8_t *ofmap, int8_t *ifmap, int8_t *fmap, int M, int C, int F, int E, int R, int S, int H, int W, int U, int padding) {
	int m, c, r, s, e, f;
	for (m = 0; m < M; m++) {
		for (e = 0; e < E; e++) {
			for (f = 0; f < F; f++) {
				int32_t sum = 0;
				for (c = 0; c < C; c++) {
					for (r = 0; r < R; r++) {
						for (s = 0; s < S; s++) {
							int fmap_idx = m*C*R*S + c*R*S + r*S + s;
							int ifmap_idx = c*H*W + (e*U+r)*W + (f*U+s);
							sum += (int32_t)fmap[fmap_idx] * (int32_t)ifmap[ifmap_idx];
						}
					}
				}
				sum >>= Q_FRAC;
				if (sum > 127) sum = 127;
				else if (sum < -128) sum = -128;
				ofmap[m*E*F + e*F + f] = (int8_t)sum;
			}
		}
	}
}

// 그룹 컨볼루션 (Layer 2, 4, 5 / G=1로 동작)
void convolution_G(int8_t *ofmap, int8_t *ifmap, int8_t *fmap, int M, int C, int F, int E, int R, int S, int H, int W, int padding, int G) {
	int m_per_g = M / G;
	int c_per_g = C / G;
	for (int g = 0; g < G; g++) {
		for (int m = 0; m < m_per_g; m++) {
			int out_c = g * m_per_g + m;
			for (int e = 0; e < E; e++) {
				for (int f = 0; f < F; f++) {
					int32_t sum = 0;
					for (int c = 0; c < c_per_g; c++) {
						int in_c = g * c_per_g + c;
						for (int r = 0; r < R; r++) {
							for (int s = 0; s < S; s++) {
								int fmap_idx = out_c*c_per_g*R*S + c*R*S + r*S + s;
								int ifmap_idx = in_c*H*W + (e+r)*W + (f+s);
								sum += (int32_t)fmap[fmap_idx] * (int32_t)ifmap[ifmap_idx];
							}
						}
					}
					sum >>= Q_FRAC;
					if (sum > 127) sum = 127;
					else if (sum < -128) sum = -128;
					ofmap[out_c*E*F + e*F + f] = (int8_t)sum;
				}
			}
		}
	}
}

// 편향(Bias) 더하기
void bias(int8_t *ofmap, int8_t *bias_arr, int M, int E, int F) {
	for (int m = 0; m < M; m++) {
		for (int e = 0; e < E; e++) {
			for (int f = 0; f < F; f++) {
				int32_t val = (int32_t)ofmap[m*E*F + e*F + f] + (int32_t)bias_arr[m];
				if (val > 127) val = 127;
				else if (val < -128) val = -128;
				ofmap[m*E*F + e*F + f] = (int8_t)val;
			}
		}
	}
}

// 활성화 함수 (ReLU)
void relu(int8_t *ofmap, int C, int E, int F) {
	for (int c = 0; c<C; c++) {
		for (int e = 0; e<E; e++) {
			for (int f = 0; f<F; f++) {
				if (ofmap[c*E*F + e*F + f] < 0) {
					ofmap[c*E*F + e*F + f] = 0;
				}
			}
		}
	}
}

// 풀링 (Max Pooling - 정수형에서도 완벽히 동일하게 동작)
void pooling(int8_t *ofmap, int8_t *ifmap, int E_in, int F_in, int C, int E_out, int F_out, int kernel, int stride) {
	for (int c = 0; c < C; c++) {
		for (int e = 0; e < E_out; e++) {
			for (int f = 0; f < F_out; f++) {
				int8_t max_val = -128; // 8비트 최소값으로 초기화
				for (int r = 0; r < kernel; r++) {
					for (int s = 0; s < kernel; s++) {
						int h_idx = e * stride + r;
						int w_idx = f * stride + s;
						if (h_idx < E_in && w_idx < F_in) {
							int8_t val = ifmap[c*E_in*F_in + h_idx*F_in + w_idx];
							if (val > max_val) max_val = val;
						}
					}
				}
				ofmap[c*E_out*F_out + e*F_out + f] = max_val;
			}
		}
	}
}

#endif