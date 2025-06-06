/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) SAM_equation_system_cost_y_0_fun_jac_ut_xt_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_densify CASADI_PREFIX(densify)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s10 CASADI_PREFIX(s10)
#define casadi_s11 CASADI_PREFIX(s11)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_s8 CASADI_PREFIX(s8)
#define casadi_s9 CASADI_PREFIX(s9)
#define casadi_sparsify CASADI_PREFIX(sparsify)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

casadi_real casadi_dot(casadi_int n, const casadi_real* x, const casadi_real* y) {
  casadi_int i;
  casadi_real r = 0;
  for (i=0; i<n; ++i) r += *x++ * *y++;
  return r;
}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

#define CASADI_CAST(x,y) ((x) y)

void casadi_densify(const casadi_real* x, const casadi_int* sp_x, casadi_real* y, casadi_int tr) {
  casadi_int nrow_x, ncol_x, i, el;
  const casadi_int *colind_x, *row_x;
  if (!y) return;
  nrow_x = sp_x[0]; ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x+ncol_x+3;
  casadi_clear(y, nrow_x*ncol_x);
  if (!x) return;
  if (tr) {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[i + row_x[el]*ncol_x] = CASADI_CAST(casadi_real, *x++);
      }
    }
  } else {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[row_x[el]] = CASADI_CAST(casadi_real, *x++);
      }
      y += nrow_x;
    }
  }
}

void casadi_sparsify(const casadi_real* x, casadi_real* y, const casadi_int* sp_y, casadi_int tr) {
  casadi_int nrow_y, ncol_y, i, el;
  const casadi_int *colind_y, *row_y;
  nrow_y = sp_y[0];
  ncol_y = sp_y[1];
  colind_y = sp_y+2; row_y = sp_y+ncol_y+3;
  if (tr) {
    for (i=0; i<ncol_y; ++i) {
      for (el=colind_y[i]; el!=colind_y[i+1]; ++el) {
        *y++ = CASADI_CAST(casadi_real, x[i + row_y[el]*ncol_y]);
      }
    }
  } else {
    for (i=0; i<ncol_y; ++i) {
      for (el=colind_y[i]; el!=colind_y[i+1]; ++el) {
        *y++ = CASADI_CAST(casadi_real, x[row_y[el]]);
      }
      x += nrow_y;
    }
  }
}

static const casadi_int casadi_s0[5] = {4, 1, 0, 1, 0};
static const casadi_int casadi_s1[25] = {0, 1, 2, 3, 7, 11, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36};
static const casadi_int casadi_s2[5] = {4, 1, 0, 1, 1};
static const casadi_int casadi_s3[5] = {4, 1, 0, 1, 2};
static const casadi_int casadi_s4[5] = {4, 1, 0, 1, 3};
static const casadi_int casadi_s5[23] = {19, 1, 0, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
static const casadi_int casadi_s6[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s7[4] = {0, 1, 0, 0};
static const casadi_int casadi_s8[3] = {0, 0, 0};
static const casadi_int casadi_s9[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
static const casadi_int casadi_s10[65] = {25, 25, 0, 1, 2, 3, 7, 11, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s11[3] = {25, 0, 0};

/* SAM_equation_system_cost_y_0_fun_jac_ut_xt:(i0[19],i1[6],i2[0],i3[],i4[25])->(o0[25],o1[25x25,37nz],o2[25x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real *rr, *ss;
  const casadi_int *cii;
  const casadi_real *cr, *cs;
  casadi_real *w0=w+0, *w1=w+19, *w2=w+22, *w3=w+47, w4, w5, w6, w7, w8, *w9=w+55, *w10=w+59, w11, w12, w13, w14, w15, w16, *w17=w+69, *w18=w+75, *w19=w+81, *w20=w+118, *w21=w+140, *w23=w+156, *w24=w+160, w25, w26, *w27=w+166;
  /* #0: @0 = input[0][0] */
  casadi_copy(arg[0], 19, w0);
  /* #1: @1 = @0[:3] */
  for (rr=w1, ss=w0+0; ss!=w0+3; ss+=1) *rr++ = *ss;
  /* #2: @2 = input[4][0] */
  casadi_copy(arg[4], 25, w2);
  /* #3: @3 = @2[:3] */
  for (rr=w3, ss=w2+0; ss!=w2+3; ss+=1) *rr++ = *ss;
  /* #4: @1 = (@1-@3) */
  for (i=0, rr=w1, cs=w3; i<3; ++i) (*rr++) -= (*cs++);
  /* #5: output[0][0] = @1 */
  casadi_copy(w1, 3, res[0]);
  /* #6: @4 = @2[3] */
  for (rr=(&w4), ss=w2+3; ss!=w2+4; ss+=1) *rr++ = *ss;
  /* #7: @5 = @0[3] */
  for (rr=(&w5), ss=w0+3; ss!=w0+4; ss+=1) *rr++ = *ss;
  /* #8: @6 = @0[4] */
  for (rr=(&w6), ss=w0+4; ss!=w0+5; ss+=1) *rr++ = *ss;
  /* #9: @6 = (-@6) */
  w6 = (- w6 );
  /* #10: @7 = @0[5] */
  for (rr=(&w7), ss=w0+5; ss!=w0+6; ss+=1) *rr++ = *ss;
  /* #11: @7 = (-@7) */
  w7 = (- w7 );
  /* #12: @8 = @0[6] */
  for (rr=(&w8), ss=w0+6; ss!=w0+7; ss+=1) *rr++ = *ss;
  /* #13: @8 = (-@8) */
  w8 = (- w8 );
  /* #14: @9 = vertcat(@5, @6, @7, @8) */
  rr=w9;
  *rr++ = w5;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w8;
  /* #15: @10 = @0[3:7] */
  for (rr=w10, ss=w0+3; ss!=w0+7; ss+=1) *rr++ = *ss;
  /* #16: @5 = ||@10||_F */
  w5 = sqrt(casadi_dot(4, w10, w10));
  /* #17: @9 = (@9/@5) */
  for (i=0, rr=w9; i<4; ++i) (*rr++) /= w5;
  /* #18: @6 = @9[0] */
  for (rr=(&w6), ss=w9+0; ss!=w9+1; ss+=1) *rr++ = *ss;
  /* #19: @7 = (@4*@6) */
  w7  = (w4*w6);
  /* #20: @8 = @2[4] */
  for (rr=(&w8), ss=w2+4; ss!=w2+5; ss+=1) *rr++ = *ss;
  /* #21: @11 = @9[1] */
  for (rr=(&w11), ss=w9+1; ss!=w9+2; ss+=1) *rr++ = *ss;
  /* #22: @12 = (@8*@11) */
  w12  = (w8*w11);
  /* #23: @7 = (@7-@12) */
  w7 -= w12;
  /* #24: @12 = @2[5] */
  for (rr=(&w12), ss=w2+5; ss!=w2+6; ss+=1) *rr++ = *ss;
  /* #25: @13 = @9[2] */
  for (rr=(&w13), ss=w9+2; ss!=w9+3; ss+=1) *rr++ = *ss;
  /* #26: @14 = (@12*@13) */
  w14  = (w12*w13);
  /* #27: @7 = (@7-@14) */
  w7 -= w14;
  /* #28: @14 = @2[6] */
  for (rr=(&w14), ss=w2+6; ss!=w2+7; ss+=1) *rr++ = *ss;
  /* #29: @15 = @9[3] */
  for (rr=(&w15), ss=w9+3; ss!=w9+4; ss+=1) *rr++ = *ss;
  /* #30: @16 = (@14*@15) */
  w16  = (w14*w15);
  /* #31: @7 = (@7-@16) */
  w7 -= w16;
  /* #32: output[0][1] = @7 */
  if (res[0]) res[0][3] = w7;
  /* #33: @7 = (@4*@11) */
  w7  = (w4*w11);
  /* #34: @16 = (@8*@6) */
  w16  = (w8*w6);
  /* #35: @7 = (@7+@16) */
  w7 += w16;
  /* #36: @16 = (@12*@15) */
  w16  = (w12*w15);
  /* #37: @7 = (@7+@16) */
  w7 += w16;
  /* #38: @16 = (@14*@13) */
  w16  = (w14*w13);
  /* #39: @7 = (@7-@16) */
  w7 -= w16;
  /* #40: output[0][2] = @7 */
  if (res[0]) res[0][4] = w7;
  /* #41: @7 = (@4*@13) */
  w7  = (w4*w13);
  /* #42: @16 = (@8*@15) */
  w16  = (w8*w15);
  /* #43: @7 = (@7-@16) */
  w7 -= w16;
  /* #44: @16 = (@12*@6) */
  w16  = (w12*w6);
  /* #45: @7 = (@7+@16) */
  w7 += w16;
  /* #46: @16 = (@14*@11) */
  w16  = (w14*w11);
  /* #47: @7 = (@7+@16) */
  w7 += w16;
  /* #48: output[0][3] = @7 */
  if (res[0]) res[0][5] = w7;
  /* #49: @15 = (@4*@15) */
  w15  = (w4*w15);
  /* #50: @13 = (@8*@13) */
  w13  = (w8*w13);
  /* #51: @15 = (@15+@13) */
  w15 += w13;
  /* #52: @11 = (@12*@11) */
  w11  = (w12*w11);
  /* #53: @15 = (@15-@11) */
  w15 -= w11;
  /* #54: @6 = (@14*@6) */
  w6  = (w14*w6);
  /* #55: @15 = (@15+@6) */
  w15 += w6;
  /* #56: output[0][4] = @15 */
  if (res[0]) res[0][6] = w15;
  /* #57: @17 = @0[7:13] */
  for (rr=w17, ss=w0+7; ss!=w0+13; ss+=1) *rr++ = *ss;
  /* #58: @18 = @2[7:13] */
  for (rr=w18, ss=w2+7; ss!=w2+13; ss+=1) *rr++ = *ss;
  /* #59: @17 = (@17-@18) */
  for (i=0, rr=w17, cs=w18; i<6; ++i) (*rr++) -= (*cs++);
  /* #60: output[0][5] = @17 */
  if (res[0]) casadi_copy(w17, 6, res[0]+7);
  /* #61: @17 = @0[13:19] */
  for (rr=w17, ss=w0+13; ss!=w0+19; ss+=1) *rr++ = *ss;
  /* #62: @18 = @2[13:19] */
  for (rr=w18, ss=w2+13; ss!=w2+19; ss+=1) *rr++ = *ss;
  /* #63: @17 = (@17-@18) */
  for (i=0, rr=w17, cs=w18; i<6; ++i) (*rr++) -= (*cs++);
  /* #64: output[0][6] = @17 */
  if (res[0]) casadi_copy(w17, 6, res[0]+13);
  /* #65: @17 = input[1][0] */
  casadi_copy(arg[1], 6, w17);
  /* #66: output[0][7] = @17 */
  if (res[0]) casadi_copy(w17, 6, res[0]+19);
  /* #67: @19 = zeros(25x25,37nz) */
  casadi_clear(w19, 37);
  /* #68: @20 = ones(25x1,22nz) */
  casadi_fill(w20, 22, 1.);
  /* #69: {@17, @21} = vertsplit(@20) */
  casadi_copy(w20, 6, w17);
  casadi_copy(w20+6, 16, w21);
  /* #70: @1 = @21[:3] */
  for (rr=w1, ss=w21+0; ss!=w21+3; ss+=1) *rr++ = *ss;
  /* #71: @15 = @21[3] */
  for (rr=(&w15), ss=w21+3; ss!=w21+4; ss+=1) *rr++ = *ss;
  /* #72: @22 = 00 */
  /* #73: @6 = vertcat(@15, @22, @22, @22) */
  rr=(&w6);
  *rr++ = w15;
  /* #74: @6 = (@6/@5) */
  w6 /= w5;
  /* #75: @23 = dense(@6) */
  casadi_densify((&w6), casadi_s0, w23, 0);
  /* #76: @9 = (@9/@5) */
  for (i=0, rr=w9; i<4; ++i) (*rr++) /= w5;
  /* #77: @6 = project(@10) */
  casadi_sparsify(w10, (&w6), casadi_s0, 0);
  /* #78: @15 = @21[3] */
  for (rr=(&w15), ss=w21+3; ss!=w21+4; ss+=1) *rr++ = *ss;
  /* #79: @11 = dot(@6, @15) */
  w11 = casadi_dot(1, (&w6), (&w15));
  /* #80: @11 = (@11/@5) */
  w11 /= w5;
  /* #81: @24 = (@9*@11) */
  for (i=0, rr=w24, cr=w9; i<4; ++i) (*rr++)  = ((*cr++)*w11);
  /* #82: @23 = (@23-@24) */
  for (i=0, rr=w23, cs=w24; i<4; ++i) (*rr++) -= (*cs++);
  /* #83: @11 = @23[0] */
  for (rr=(&w11), ss=w23+0; ss!=w23+1; ss+=1) *rr++ = *ss;
  /* #84: @6 = (@4*@11) */
  w6  = (w4*w11);
  /* #85: @15 = @23[1] */
  for (rr=(&w15), ss=w23+1; ss!=w23+2; ss+=1) *rr++ = *ss;
  /* #86: @13 = (@8*@15) */
  w13  = (w8*w15);
  /* #87: @6 = (@6-@13) */
  w6 -= w13;
  /* #88: @13 = @23[2] */
  for (rr=(&w13), ss=w23+2; ss!=w23+3; ss+=1) *rr++ = *ss;
  /* #89: @7 = (@12*@13) */
  w7  = (w12*w13);
  /* #90: @6 = (@6-@7) */
  w6 -= w7;
  /* #91: @7 = @23[3] */
  for (rr=(&w7), ss=w23+3; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #92: @16 = (@14*@7) */
  w16  = (w14*w7);
  /* #93: @6 = (@6-@16) */
  w6 -= w16;
  /* #94: @16 = (@4*@15) */
  w16  = (w4*w15);
  /* #95: @25 = (@8*@11) */
  w25  = (w8*w11);
  /* #96: @16 = (@16+@25) */
  w16 += w25;
  /* #97: @25 = (@12*@7) */
  w25  = (w12*w7);
  /* #98: @16 = (@16+@25) */
  w16 += w25;
  /* #99: @25 = (@14*@13) */
  w25  = (w14*w13);
  /* #100: @16 = (@16-@25) */
  w16 -= w25;
  /* #101: @25 = (@4*@13) */
  w25  = (w4*w13);
  /* #102: @26 = (@8*@7) */
  w26  = (w8*w7);
  /* #103: @25 = (@25-@26) */
  w25 -= w26;
  /* #104: @26 = (@12*@11) */
  w26  = (w12*w11);
  /* #105: @25 = (@25+@26) */
  w25 += w26;
  /* #106: @26 = (@14*@15) */
  w26  = (w14*w15);
  /* #107: @25 = (@25+@26) */
  w25 += w26;
  /* #108: @7 = (@4*@7) */
  w7  = (w4*w7);
  /* #109: @13 = (@8*@13) */
  w13  = (w8*w13);
  /* #110: @7 = (@7+@13) */
  w7 += w13;
  /* #111: @15 = (@12*@15) */
  w15  = (w12*w15);
  /* #112: @7 = (@7-@15) */
  w7 -= w15;
  /* #113: @11 = (@14*@11) */
  w11  = (w14*w11);
  /* #114: @7 = (@7+@11) */
  w7 += w11;
  /* #115: @18 = @21[4:10] */
  for (rr=w18, ss=w21+4; ss!=w21+10; ss+=1) *rr++ = *ss;
  /* #116: @27 = @21[10:16] */
  for (rr=w27, ss=w21+10; ss!=w21+16; ss+=1) *rr++ = *ss;
  /* #117: @2 = vertcat(@1, @6, @16, @25, @7, @18, @27, @17) */
  rr=w2;
  for (i=0, cs=w1; i<3; ++i) *rr++ = *cs++;
  *rr++ = w6;
  *rr++ = w16;
  *rr++ = w25;
  *rr++ = w7;
  for (i=0, cs=w18; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w27; i<6; ++i) *rr++ = *cs++;
  for (i=0, cs=w17; i<6; ++i) *rr++ = *cs++;
  /* #118: (@19[0, 1, 2, 3, 7, 11, 15, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36] = @2) */
  for (cii=casadi_s1, rr=w19, ss=w2; cii!=casadi_s1+25; ++cii, ++ss) rr[*cii] = *ss;
  /* #119: @28 = zeros(3x1,0nz) */
  /* #120: @6 = ones(25x1,1nz) */
  w6 = 1.;
  /* #121: {NULL, @16} = vertsplit(@6) */
  w16 = w6;
  /* #122: @6 = @16[0] */
  for (rr=(&w6), ss=(&w16)+0; ss!=(&w16)+1; ss+=1) *rr++ = *ss;
  /* #123: @6 = (-@6) */
  w6 = (- w6 );
  /* #124: @25 = vertcat(@22, @6, @22, @22) */
  rr=(&w25);
  *rr++ = w6;
  /* #125: @25 = (@25/@5) */
  w25 /= w5;
  /* #126: @23 = dense(@25) */
  casadi_densify((&w25), casadi_s2, w23, 0);
  /* #127: @25 = project(@10) */
  casadi_sparsify(w10, (&w25), casadi_s2, 0);
  /* #128: @6 = @16[0] */
  for (rr=(&w6), ss=(&w16)+0; ss!=(&w16)+1; ss+=1) *rr++ = *ss;
  /* #129: @16 = dot(@25, @6) */
  w16 = casadi_dot(1, (&w25), (&w6));
  /* #130: @16 = (@16/@5) */
  w16 /= w5;
  /* #131: @24 = (@9*@16) */
  for (i=0, rr=w24, cr=w9; i<4; ++i) (*rr++)  = ((*cr++)*w16);
  /* #132: @23 = (@23-@24) */
  for (i=0, rr=w23, cs=w24; i<4; ++i) (*rr++) -= (*cs++);
  /* #133: @16 = @23[0] */
  for (rr=(&w16), ss=w23+0; ss!=w23+1; ss+=1) *rr++ = *ss;
  /* #134: @25 = (@4*@16) */
  w25  = (w4*w16);
  /* #135: @6 = @23[1] */
  for (rr=(&w6), ss=w23+1; ss!=w23+2; ss+=1) *rr++ = *ss;
  /* #136: @7 = (@8*@6) */
  w7  = (w8*w6);
  /* #137: @25 = (@25-@7) */
  w25 -= w7;
  /* #138: @7 = @23[2] */
  for (rr=(&w7), ss=w23+2; ss!=w23+3; ss+=1) *rr++ = *ss;
  /* #139: @11 = (@12*@7) */
  w11  = (w12*w7);
  /* #140: @25 = (@25-@11) */
  w25 -= w11;
  /* #141: @11 = @23[3] */
  for (rr=(&w11), ss=w23+3; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #142: @15 = (@14*@11) */
  w15  = (w14*w11);
  /* #143: @25 = (@25-@15) */
  w25 -= w15;
  /* #144: @15 = (@4*@6) */
  w15  = (w4*w6);
  /* #145: @13 = (@8*@16) */
  w13  = (w8*w16);
  /* #146: @15 = (@15+@13) */
  w15 += w13;
  /* #147: @13 = (@12*@11) */
  w13  = (w12*w11);
  /* #148: @15 = (@15+@13) */
  w15 += w13;
  /* #149: @13 = (@14*@7) */
  w13  = (w14*w7);
  /* #150: @15 = (@15-@13) */
  w15 -= w13;
  /* #151: @13 = (@4*@7) */
  w13  = (w4*w7);
  /* #152: @26 = (@8*@11) */
  w26  = (w8*w11);
  /* #153: @13 = (@13-@26) */
  w13 -= w26;
  /* #154: @26 = (@12*@16) */
  w26  = (w12*w16);
  /* #155: @13 = (@13+@26) */
  w13 += w26;
  /* #156: @26 = (@14*@6) */
  w26  = (w14*w6);
  /* #157: @13 = (@13+@26) */
  w13 += w26;
  /* #158: @11 = (@4*@11) */
  w11  = (w4*w11);
  /* #159: @7 = (@8*@7) */
  w7  = (w8*w7);
  /* #160: @11 = (@11+@7) */
  w11 += w7;
  /* #161: @6 = (@12*@6) */
  w6  = (w12*w6);
  /* #162: @11 = (@11-@6) */
  w11 -= w6;
  /* #163: @16 = (@14*@16) */
  w16  = (w14*w16);
  /* #164: @11 = (@11+@16) */
  w11 += w16;
  /* #165: @29 = zeros(6x1,0nz) */
  /* #166: @23 = vertcat(@28, @25, @15, @13, @11, @29, @29, @29) */
  rr=w23;
  *rr++ = w25;
  *rr++ = w15;
  *rr++ = w13;
  *rr++ = w11;
  /* #167: @24 = @23[:4] */
  for (rr=w24, ss=w23+0; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #168: (@19[4:20:4] = @24) */
  for (rr=w19+4, ss=w24; rr!=w19+20; rr+=4) *rr = *ss++;
  /* #169: @25 = ones(25x1,1nz) */
  w25 = 1.;
  /* #170: {NULL, @15} = vertsplit(@25) */
  w15 = w25;
  /* #171: @25 = @15[0] */
  for (rr=(&w25), ss=(&w15)+0; ss!=(&w15)+1; ss+=1) *rr++ = *ss;
  /* #172: @25 = (-@25) */
  w25 = (- w25 );
  /* #173: @13 = vertcat(@22, @22, @25, @22) */
  rr=(&w13);
  *rr++ = w25;
  /* #174: @13 = (@13/@5) */
  w13 /= w5;
  /* #175: @24 = dense(@13) */
  casadi_densify((&w13), casadi_s3, w24, 0);
  /* #176: @13 = project(@10) */
  casadi_sparsify(w10, (&w13), casadi_s3, 0);
  /* #177: @25 = @15[0] */
  for (rr=(&w25), ss=(&w15)+0; ss!=(&w15)+1; ss+=1) *rr++ = *ss;
  /* #178: @15 = dot(@13, @25) */
  w15 = casadi_dot(1, (&w13), (&w25));
  /* #179: @15 = (@15/@5) */
  w15 /= w5;
  /* #180: @23 = (@9*@15) */
  for (i=0, rr=w23, cr=w9; i<4; ++i) (*rr++)  = ((*cr++)*w15);
  /* #181: @24 = (@24-@23) */
  for (i=0, rr=w24, cs=w23; i<4; ++i) (*rr++) -= (*cs++);
  /* #182: @15 = @24[0] */
  for (rr=(&w15), ss=w24+0; ss!=w24+1; ss+=1) *rr++ = *ss;
  /* #183: @13 = (@4*@15) */
  w13  = (w4*w15);
  /* #184: @25 = @24[1] */
  for (rr=(&w25), ss=w24+1; ss!=w24+2; ss+=1) *rr++ = *ss;
  /* #185: @11 = (@8*@25) */
  w11  = (w8*w25);
  /* #186: @13 = (@13-@11) */
  w13 -= w11;
  /* #187: @11 = @24[2] */
  for (rr=(&w11), ss=w24+2; ss!=w24+3; ss+=1) *rr++ = *ss;
  /* #188: @16 = (@12*@11) */
  w16  = (w12*w11);
  /* #189: @13 = (@13-@16) */
  w13 -= w16;
  /* #190: @16 = @24[3] */
  for (rr=(&w16), ss=w24+3; ss!=w24+4; ss+=1) *rr++ = *ss;
  /* #191: @6 = (@14*@16) */
  w6  = (w14*w16);
  /* #192: @13 = (@13-@6) */
  w13 -= w6;
  /* #193: @6 = (@4*@25) */
  w6  = (w4*w25);
  /* #194: @7 = (@8*@15) */
  w7  = (w8*w15);
  /* #195: @6 = (@6+@7) */
  w6 += w7;
  /* #196: @7 = (@12*@16) */
  w7  = (w12*w16);
  /* #197: @6 = (@6+@7) */
  w6 += w7;
  /* #198: @7 = (@14*@11) */
  w7  = (w14*w11);
  /* #199: @6 = (@6-@7) */
  w6 -= w7;
  /* #200: @7 = (@4*@11) */
  w7  = (w4*w11);
  /* #201: @26 = (@8*@16) */
  w26  = (w8*w16);
  /* #202: @7 = (@7-@26) */
  w7 -= w26;
  /* #203: @26 = (@12*@15) */
  w26  = (w12*w15);
  /* #204: @7 = (@7+@26) */
  w7 += w26;
  /* #205: @26 = (@14*@25) */
  w26  = (w14*w25);
  /* #206: @7 = (@7+@26) */
  w7 += w26;
  /* #207: @16 = (@4*@16) */
  w16  = (w4*w16);
  /* #208: @11 = (@8*@11) */
  w11  = (w8*w11);
  /* #209: @16 = (@16+@11) */
  w16 += w11;
  /* #210: @25 = (@12*@25) */
  w25  = (w12*w25);
  /* #211: @16 = (@16-@25) */
  w16 -= w25;
  /* #212: @15 = (@14*@15) */
  w15  = (w14*w15);
  /* #213: @16 = (@16+@15) */
  w16 += w15;
  /* #214: @24 = vertcat(@28, @13, @6, @7, @16, @29, @29, @29) */
  rr=w24;
  *rr++ = w13;
  *rr++ = w6;
  *rr++ = w7;
  *rr++ = w16;
  /* #215: @23 = @24[:4] */
  for (rr=w23, ss=w24+0; ss!=w24+4; ss+=1) *rr++ = *ss;
  /* #216: (@19[5:21:4] = @23) */
  for (rr=w19+5, ss=w23; rr!=w19+21; rr+=4) *rr = *ss++;
  /* #217: @13 = ones(25x1,1nz) */
  w13 = 1.;
  /* #218: {NULL, @6} = vertsplit(@13) */
  w6 = w13;
  /* #219: @13 = @6[0] */
  for (rr=(&w13), ss=(&w6)+0; ss!=(&w6)+1; ss+=1) *rr++ = *ss;
  /* #220: @13 = (-@13) */
  w13 = (- w13 );
  /* #221: @7 = vertcat(@22, @22, @22, @13) */
  rr=(&w7);
  *rr++ = w13;
  /* #222: @7 = (@7/@5) */
  w7 /= w5;
  /* #223: @23 = dense(@7) */
  casadi_densify((&w7), casadi_s4, w23, 0);
  /* #224: @7 = project(@10) */
  casadi_sparsify(w10, (&w7), casadi_s4, 0);
  /* #225: @13 = @6[0] */
  for (rr=(&w13), ss=(&w6)+0; ss!=(&w6)+1; ss+=1) *rr++ = *ss;
  /* #226: @6 = dot(@7, @13) */
  w6 = casadi_dot(1, (&w7), (&w13));
  /* #227: @6 = (@6/@5) */
  w6 /= w5;
  /* #228: @9 = (@9*@6) */
  for (i=0, rr=w9; i<4; ++i) (*rr++) *= w6;
  /* #229: @23 = (@23-@9) */
  for (i=0, rr=w23, cs=w9; i<4; ++i) (*rr++) -= (*cs++);
  /* #230: @6 = @23[0] */
  for (rr=(&w6), ss=w23+0; ss!=w23+1; ss+=1) *rr++ = *ss;
  /* #231: @5 = (@4*@6) */
  w5  = (w4*w6);
  /* #232: @7 = @23[1] */
  for (rr=(&w7), ss=w23+1; ss!=w23+2; ss+=1) *rr++ = *ss;
  /* #233: @13 = (@8*@7) */
  w13  = (w8*w7);
  /* #234: @5 = (@5-@13) */
  w5 -= w13;
  /* #235: @13 = @23[2] */
  for (rr=(&w13), ss=w23+2; ss!=w23+3; ss+=1) *rr++ = *ss;
  /* #236: @16 = (@12*@13) */
  w16  = (w12*w13);
  /* #237: @5 = (@5-@16) */
  w5 -= w16;
  /* #238: @16 = @23[3] */
  for (rr=(&w16), ss=w23+3; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #239: @15 = (@14*@16) */
  w15  = (w14*w16);
  /* #240: @5 = (@5-@15) */
  w5 -= w15;
  /* #241: @15 = (@4*@7) */
  w15  = (w4*w7);
  /* #242: @25 = (@8*@6) */
  w25  = (w8*w6);
  /* #243: @15 = (@15+@25) */
  w15 += w25;
  /* #244: @25 = (@12*@16) */
  w25  = (w12*w16);
  /* #245: @15 = (@15+@25) */
  w15 += w25;
  /* #246: @25 = (@14*@13) */
  w25  = (w14*w13);
  /* #247: @15 = (@15-@25) */
  w15 -= w25;
  /* #248: @25 = (@4*@13) */
  w25  = (w4*w13);
  /* #249: @11 = (@8*@16) */
  w11  = (w8*w16);
  /* #250: @25 = (@25-@11) */
  w25 -= w11;
  /* #251: @11 = (@12*@6) */
  w11  = (w12*w6);
  /* #252: @25 = (@25+@11) */
  w25 += w11;
  /* #253: @11 = (@14*@7) */
  w11  = (w14*w7);
  /* #254: @25 = (@25+@11) */
  w25 += w11;
  /* #255: @4 = (@4*@16) */
  w4 *= w16;
  /* #256: @8 = (@8*@13) */
  w8 *= w13;
  /* #257: @4 = (@4+@8) */
  w4 += w8;
  /* #258: @12 = (@12*@7) */
  w12 *= w7;
  /* #259: @4 = (@4-@12) */
  w4 -= w12;
  /* #260: @14 = (@14*@6) */
  w14 *= w6;
  /* #261: @4 = (@4+@14) */
  w4 += w14;
  /* #262: @23 = vertcat(@28, @5, @15, @25, @4, @29, @29, @29) */
  rr=w23;
  *rr++ = w5;
  *rr++ = w15;
  *rr++ = w25;
  *rr++ = w4;
  /* #263: @9 = @23[:4] */
  for (rr=w9, ss=w23+0; ss!=w23+4; ss+=1) *rr++ = *ss;
  /* #264: (@19[6:22:4] = @9) */
  for (rr=w19+6, ss=w9; rr!=w19+22; rr+=4) *rr = *ss++;
  /* #265: output[1][0] = @19 */
  casadi_copy(w19, 37, res[1]);
  return 0;
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_0_fun_jac_ut_xt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_0_fun_jac_ut_xt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_0_fun_jac_ut_xt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_0_fun_jac_ut_xt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_0_fun_jac_ut_xt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_0_fun_jac_ut_xt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_0_fun_jac_ut_xt_incref(void) {
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_0_fun_jac_ut_xt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int SAM_equation_system_cost_y_0_fun_jac_ut_xt_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int SAM_equation_system_cost_y_0_fun_jac_ut_xt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real SAM_equation_system_cost_y_0_fun_jac_ut_xt_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* SAM_equation_system_cost_y_0_fun_jac_ut_xt_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* SAM_equation_system_cost_y_0_fun_jac_ut_xt_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* SAM_equation_system_cost_y_0_fun_jac_ut_xt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s5;
    case 1: return casadi_s6;
    case 2: return casadi_s7;
    case 3: return casadi_s8;
    case 4: return casadi_s9;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* SAM_equation_system_cost_y_0_fun_jac_ut_xt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s9;
    case 1: return casadi_s10;
    case 2: return casadi_s11;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_0_fun_jac_ut_xt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 13;
  if (sz_res) *sz_res = 5;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 172;
  return 0;
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_0_fun_jac_ut_xt_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 13*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 5*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 172*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
