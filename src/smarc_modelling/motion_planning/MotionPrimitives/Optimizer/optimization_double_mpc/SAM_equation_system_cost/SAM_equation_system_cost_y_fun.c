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
  #define CASADI_PREFIX(ID) SAM_equation_system_cost_y_fun_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_dot CASADI_PREFIX(dot)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)

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

static const casadi_int casadi_s0[23] = {19, 1, 0, 19, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
static const casadi_int casadi_s1[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s2[4] = {0, 1, 0, 0};
static const casadi_int casadi_s3[3] = {0, 0, 0};
static const casadi_int casadi_s4[29] = {25, 1, 0, 25, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};

/* SAM_equation_system_cost_y_fun:(i0[19],i1[6],i2[0],i3[],i4[25])->(o0[25]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_int i;
  casadi_real *rr, *ss;
  const casadi_real *cs;
  casadi_real *w0=w+0, *w1=w+19, *w2=w+22, *w3=w+47, w4, w5, w6, w7, w8, *w9=w+55, *w10=w+59, w11, w12, w13, w14, w15, *w16=w+68, *w17=w+74;
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
  /* #18: @5 = @9[0] */
  for (rr=(&w5), ss=w9+0; ss!=w9+1; ss+=1) *rr++ = *ss;
  /* #19: @6 = (@4*@5) */
  w6  = (w4*w5);
  /* #20: @7 = @2[4] */
  for (rr=(&w7), ss=w2+4; ss!=w2+5; ss+=1) *rr++ = *ss;
  /* #21: @8 = @9[1] */
  for (rr=(&w8), ss=w9+1; ss!=w9+2; ss+=1) *rr++ = *ss;
  /* #22: @11 = (@7*@8) */
  w11  = (w7*w8);
  /* #23: @6 = (@6-@11) */
  w6 -= w11;
  /* #24: @11 = @2[5] */
  for (rr=(&w11), ss=w2+5; ss!=w2+6; ss+=1) *rr++ = *ss;
  /* #25: @12 = @9[2] */
  for (rr=(&w12), ss=w9+2; ss!=w9+3; ss+=1) *rr++ = *ss;
  /* #26: @13 = (@11*@12) */
  w13  = (w11*w12);
  /* #27: @6 = (@6-@13) */
  w6 -= w13;
  /* #28: @13 = @2[6] */
  for (rr=(&w13), ss=w2+6; ss!=w2+7; ss+=1) *rr++ = *ss;
  /* #29: @14 = @9[3] */
  for (rr=(&w14), ss=w9+3; ss!=w9+4; ss+=1) *rr++ = *ss;
  /* #30: @15 = (@13*@14) */
  w15  = (w13*w14);
  /* #31: @6 = (@6-@15) */
  w6 -= w15;
  /* #32: output[0][1] = @6 */
  if (res[0]) res[0][3] = w6;
  /* #33: @6 = (@4*@8) */
  w6  = (w4*w8);
  /* #34: @15 = (@7*@5) */
  w15  = (w7*w5);
  /* #35: @6 = (@6+@15) */
  w6 += w15;
  /* #36: @15 = (@11*@14) */
  w15  = (w11*w14);
  /* #37: @6 = (@6+@15) */
  w6 += w15;
  /* #38: @15 = (@13*@12) */
  w15  = (w13*w12);
  /* #39: @6 = (@6-@15) */
  w6 -= w15;
  /* #40: output[0][2] = @6 */
  if (res[0]) res[0][4] = w6;
  /* #41: @6 = (@4*@12) */
  w6  = (w4*w12);
  /* #42: @15 = (@7*@14) */
  w15  = (w7*w14);
  /* #43: @6 = (@6-@15) */
  w6 -= w15;
  /* #44: @15 = (@11*@5) */
  w15  = (w11*w5);
  /* #45: @6 = (@6+@15) */
  w6 += w15;
  /* #46: @15 = (@13*@8) */
  w15  = (w13*w8);
  /* #47: @6 = (@6+@15) */
  w6 += w15;
  /* #48: output[0][3] = @6 */
  if (res[0]) res[0][5] = w6;
  /* #49: @4 = (@4*@14) */
  w4 *= w14;
  /* #50: @7 = (@7*@12) */
  w7 *= w12;
  /* #51: @4 = (@4+@7) */
  w4 += w7;
  /* #52: @11 = (@11*@8) */
  w11 *= w8;
  /* #53: @4 = (@4-@11) */
  w4 -= w11;
  /* #54: @13 = (@13*@5) */
  w13 *= w5;
  /* #55: @4 = (@4+@13) */
  w4 += w13;
  /* #56: output[0][4] = @4 */
  if (res[0]) res[0][6] = w4;
  /* #57: @16 = @0[7:13] */
  for (rr=w16, ss=w0+7; ss!=w0+13; ss+=1) *rr++ = *ss;
  /* #58: @17 = @2[7:13] */
  for (rr=w17, ss=w2+7; ss!=w2+13; ss+=1) *rr++ = *ss;
  /* #59: @16 = (@16-@17) */
  for (i=0, rr=w16, cs=w17; i<6; ++i) (*rr++) -= (*cs++);
  /* #60: output[0][5] = @16 */
  if (res[0]) casadi_copy(w16, 6, res[0]+7);
  /* #61: @16 = @0[13:19] */
  for (rr=w16, ss=w0+13; ss!=w0+19; ss+=1) *rr++ = *ss;
  /* #62: @17 = @2[13:19] */
  for (rr=w17, ss=w2+13; ss!=w2+19; ss+=1) *rr++ = *ss;
  /* #63: @16 = (@16-@17) */
  for (i=0, rr=w16, cs=w17; i<6; ++i) (*rr++) -= (*cs++);
  /* #64: output[0][6] = @16 */
  if (res[0]) casadi_copy(w16, 6, res[0]+13);
  /* #65: @16 = input[1][0] */
  casadi_copy(arg[1], 6, w16);
  /* #66: output[0][7] = @16 */
  if (res[0]) casadi_copy(w16, 6, res[0]+19);
  return 0;
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_fun(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_fun_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_fun_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_fun_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_fun_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_fun_release(int mem) {
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_fun_incref(void) {
}

CASADI_SYMBOL_EXPORT void SAM_equation_system_cost_y_fun_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int SAM_equation_system_cost_y_fun_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int SAM_equation_system_cost_y_fun_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real SAM_equation_system_cost_y_fun_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* SAM_equation_system_cost_y_fun_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* SAM_equation_system_cost_y_fun_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* SAM_equation_system_cost_y_fun_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* SAM_equation_system_cost_y_fun_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_fun_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 80;
  return 0;
}

CASADI_SYMBOL_EXPORT int SAM_equation_system_cost_y_fun_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 9*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 2*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 80*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
