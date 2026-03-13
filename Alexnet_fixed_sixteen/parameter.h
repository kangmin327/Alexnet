//
#define _mode  2
#define _data  10
#define _class 1000

//Convlayer #1 parameter
#define C_C1 3
#define M_C1 64
#define E_C1 55
#define F_C1 55
#define R_C1 11
#define S_C1 11
#define H_C1 227
#define W_C1 227
#define U_C1 4
#define P_C1 0

#define E_P1 27
#define F_P1 27

#define MEMPOS_C1 0
#define MEMPOS_B1 C_C1*M_C1*R_C1*S_C1

//Pool #1 parameter
#define K_P1 3
#define S_P1 2
// #define E_P1 (E_C1+S_P1-K_P1)/2
// #define F_P1 (F_C1+S_P1-K_P1)/2


//Convlayer #2 parameter
#define C_C2 64
#define M_C2 192
#define E_C2 27
#define F_C2 27
#define R_C2 5
#define S_C2 5
#define H_C2 31
#define W_C2 31
#define U_C2 1
#define P_C2 2
#define G_C2 1

#define E_P2 13
#define F_P2 13

#define MEMPOS_C2 (MEMPOS_B1+(M_C1))
#define MEMPOS_B2 (MEMPOS_C2+(C_C2*M_C2*R_C2*S_C2)/G_C2)

//Pool #2 parameter
#define K_P2 3
#define S_P2 2
// #define E_P2 (E_C2+S_P2-K_P2)/2
// #define F_P2 (F_C2+S_P2-K_P2)/2

//Convlayer #3 parameter
#define C_C3 192
#define M_C3 384
#define E_C3 13
#define F_C3 13
#define R_C3 3
#define S_C3 3
#define H_C3 15 //padded
#define W_C3 15 //padded
#define P_C3 1
#define U_C3 1

#define E_P3 13
#define F_P3 13

#define MEMPOS_C3 (MEMPOS_B2+(M_C2))
#define MEMPOS_B3 (MEMPOS_C3+(C_C3*M_C3*R_C3*S_C3))

#define L3_TileM 8
#define L3_TileC 2
#define L3_TileE 13
#define L3_TileF 13

#define TileH   L3_TileE + R_C2 - 1
#define TileW   L3_TileF + S_C2 - 1

//Convlayer #4 parameter
#define C_C4 384
#define M_C4 256
#define E_C4 13
#define F_C4 13
#define R_C4 3
#define S_C4 3
#define H_C4 15
#define W_C4 15
#define P_C4 1
#define U_C4 1
#define G_C4 1

#define E_P4 13
#define F_P4 13

#define MEMPOS_C4 (MEMPOS_B3+(M_C3))
#define MEMPOS_B4 (MEMPOS_C4+(C_C4*M_C4*R_C4*S_C4)/G_C4)

//Convlayer #5 parameter
#define C_C5 256
#define M_C5 256
#define E_C5 13
#define F_C5 13
#define R_C5 3
#define S_C5 3
#define H_C5 15
#define W_C5 15
#define P_C5 1
#define U_C5 1
#define G_C5 1
#define MEMPOS_C5 (MEMPOS_B4+(M_C4))
#define MEMPOS_B5 (MEMPOS_C5+(C_C5*M_C5*R_C5*S_C5)/G_C5)

//Pool #3 parameter
#define K_P3 3
#define S_P3 2
// #define E_P3 (E_C5+S_P3-K_P3)/2
// #define F_P3 (F_C5+S_P3-K_P3)/2


//FClayer #6 parameter
#define C_C6 256
#define M_C6 4096
#define E_C6 1
#define F_C6 1
#define R_C6 6
#define S_C6 6
#define H_C6 6
#define W_C6 6
#define P_C6 0
#define MEMPOS_C6 (MEMPOS_B5+(M_C5))
#define MEMPOS_B6 (MEMPOS_C6+(C_C6*M_C6*R_C6*S_C6))

//FClayer #7 parameter
#define C_C7 4096
#define M_C7 4096
#define E_C7 1
#define F_C7 1
#define R_C7 1
#define S_C7 1
#define H_C7 1
#define W_C7 1
#define P_C7 0
#define MEMPOS_C7 (MEMPOS_B6+(M_C6))
#define MEMPOS_B7 (MEMPOS_C7+(C_C7*M_C7*R_C7*S_C7))

//FClayer #8 parameter
#define C_C8 4096
#define M_C8 1000
#define E_C8 1
#define F_C8 1
#define R_C8 1
#define S_C8 1
#define H_C8 1
#define W_C8 1
#define P_C8 0
#define MEMPOS_C8 (MEMPOS_B7+(M_C7))
#define MEMPOS_B8 (MEMPOS_C8+(C_C8*M_C8*R_C8*S_C8))

#define MEMPOS_AF (MEMPOS_B8+(M_C8))


// //Softmax #1 parameter
// #define N_S1 1
// #define C_S1 500
// #define M_S1 10
// #define E_S1 1
// #define F_S1 1
// #define R_S1 1
// #define S_S1 1
// #define H_S1 1
// #define W_S1 1
// #define U_S1 1
