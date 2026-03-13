#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stdint.h>
#include <math.h>

#define MY_MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MY_MAX(X,Y) ((X) > (Y) ? (X) : (Y))

// 16비트를 위한 새로운 Q-Format (소수점 11자리 = 2^11 = 2048)
// 이제 미세한 소수점들도 뭉개지지 않고 살아남습니다!
#define Q_FRAC 7

void paddata(int16_t *ofmap, int16_t *ifmap, int C, int H, int W, int padding) {
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

void convolution_B(int16_t *ofmap, int16_t *ifmap, int16_t *fmap, int M, int C, int F, int E, int R, int S, int H, int W, int padding) {
	int m, c, r, s, e, f;
	for (m = 0; m < M; m++) {
		for (e = 0; e < E; e++) {
			for (f = 0; f < F; f++) {
				// [핵심] 곱셈 누적 시 32비트도 터집니다! 64비트(int64_t) 슈퍼 창고 사용!
				int64_t sum = 0; 
				for (c = 0; c < C; c++) {
					for (r = 0; r < R; r++) {
						for (s = 0; s < S; s++) {
							int fmap_idx = m*C*R*S + c*R*S + r*S + s;
							int ifmap_idx = c*H*W + (e+r)*W + (f+s);
							sum += (int64_t)fmap[fmap_idx] * (int64_t)ifmap[ifmap_idx];
						}
					}
				}
				sum >>= Q_FRAC; // 소수점 위치 원상복구
				
				// 16비트 한계치(32767 ~ -32768) 클리핑
				if (sum > 32767) sum = 32767;
				else if (sum < -32768) sum = -32768;
				
				ofmap[m*E*F + e*F + f] = (int16_t)sum;
			}
		}
	}
}

void convolution_S(int16_t *ofmap, int16_t *ifmap, int16_t *fmap, int M, int C, int F, int E, int R, int S, int H, int W, int U, int padding) {
	int m, c, r, s, e, f;
	for (m = 0; m < M; m++) {
		for (e = 0; e < E; e++) {
			for (f = 0; f < F; f++) {
				int64_t sum = 0;
				for (c = 0; c < C; c++) {
					for (r = 0; r < R; r++) {
						for (s = 0; s < S; s++) {
							int fmap_idx = m*C*R*S + c*R*S + r*S + s;
							int ifmap_idx = c*H*W + (e*U+r)*W + (f*U+s);
							sum += (int64_t)fmap[fmap_idx] * (int64_t)ifmap[ifmap_idx];
						}
					}
				}
				sum >>= Q_FRAC;
				if (sum > 32767) sum = 32767;
				else if (sum < -32768) sum = -32768;
				ofmap[m*E*F + e*F + f] = (int16_t)sum;
			}
		}
	}
}

void convolution_G(int16_t *ofmap, int16_t *ifmap, int16_t *fmap, int M, int C, int F, int E, int R, int S, int H, int W, int padding, int G) {
	int m_per_g = M / G;
	int c_per_g = C / G;
	for (int g = 0; g < G; g++) {
		for (int m = 0; m < m_per_g; m++) {
			int out_c = g * m_per_g + m;
			for (int e = 0; e < E; e++) {
				for (int f = 0; f < F; f++) {
					int64_t sum = 0;
					for (int c = 0; c < c_per_g; c++) {
						int in_c = g * c_per_g + c;
						for (int r = 0; r < R; r++) {
							for (int s = 0; s < S; s++) {
								int fmap_idx = out_c*c_per_g*R*S + c*R*S + r*S + s;
								int ifmap_idx = in_c*H*W + (e+r)*W + (f+s);
								sum += (int64_t)fmap[fmap_idx] * (int64_t)ifmap[ifmap_idx];
							}
						}
					}
					sum >>= Q_FRAC;
					if (sum > 32767) sum = 32767;
					else if (sum < -32768) sum = -32768;
					ofmap[out_c*E*F + e*F + f] = (int16_t)sum;
				}
			}
		}
	}
}

void bias(int16_t *ofmap, int16_t *bias_arr, int M, int E, int F) {
	for (int m = 0; m < M; m++) {
		for (int e = 0; e < E; e++) {
			for (int f = 0; f < F; f++) {
				int64_t val = (int64_t)ofmap[m*E*F + e*F + f] + (int64_t)bias_arr[m];
				if (val > 32767) val = 32767;
				else if (val < -32768) val = -32768;
				ofmap[m*E*F + e*F + f] = (int16_t)val;
			}
		}
	}
}

void relu(int16_t *ofmap, int C, int E, int F) {
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

void pooling(int16_t *ofmap, int16_t *ifmap, int E_in, int F_in, int C, int E_out, int F_out, int kernel, int stride) {
	for (int c = 0; c < C; c++) {
		for (int e = 0; e < E_out; e++) {
			for (int f = 0; f < F_out; f++) {
				int16_t max_val = -32768; // 16비트 최솟값으로 초기화
				for (int r = 0; r < kernel; r++) {
					for (int s = 0; s < kernel; s++) {
						int h_idx = e * stride + r;
						int w_idx = f * stride + s;
						if (h_idx < E_in && w_idx < F_in) {
							int16_t val = ifmap[c*E_in*F_in + h_idx*F_in + w_idx];
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