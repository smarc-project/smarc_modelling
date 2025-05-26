/*
 * Copyright (c) The acados authors.
 *
 * This file is part of acados.
 *
 * The 2-Clause BSD License
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.;
 */


// standard
#include <stdio.h>
#include <stdlib.h>
// acados
#include "acados/utils/print.h"
#include "acados/utils/math.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"
#include "acados_solver_SAM_equation_system.h"

// blasfeo
#include "blasfeo_d_aux_ext_dep.h"

#define NX     SAM_EQUATION_SYSTEM_NX
#define NP     SAM_EQUATION_SYSTEM_NP
#define NU     SAM_EQUATION_SYSTEM_NU
#define NBX0   SAM_EQUATION_SYSTEM_NBX0
#define NP_GLOBAL   SAM_EQUATION_SYSTEM_NP_GLOBAL


int main()
{

    SAM_equation_system_solver_capsule *acados_ocp_capsule = SAM_equation_system_acados_create_capsule();
    // there is an opportunity to change the number of shooting intervals in C without new code generation
    int N = SAM_EQUATION_SYSTEM_N;
    // allocate the array and fill it accordingly
    double* new_time_steps = NULL;
    int status = SAM_equation_system_acados_create_with_discretization(acados_ocp_capsule, N, new_time_steps);

    if (status)
    {
        printf("SAM_equation_system_acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    ocp_nlp_config *nlp_config = SAM_equation_system_acados_get_nlp_config(acados_ocp_capsule);
    ocp_nlp_dims *nlp_dims = SAM_equation_system_acados_get_nlp_dims(acados_ocp_capsule);
    ocp_nlp_in *nlp_in = SAM_equation_system_acados_get_nlp_in(acados_ocp_capsule);
    ocp_nlp_out *nlp_out = SAM_equation_system_acados_get_nlp_out(acados_ocp_capsule);
    ocp_nlp_solver *nlp_solver = SAM_equation_system_acados_get_nlp_solver(acados_ocp_capsule);
    void *nlp_opts = SAM_equation_system_acados_get_nlp_opts(acados_ocp_capsule);

    // initial condition
    double lbx0[NBX0];
    double ubx0[NBX0];
    lbx0[0] = 3.947382191056128;
    ubx0[0] = 3.947382191056128;
    lbx0[1] = 1.4700959582754711;
    ubx0[1] = 1.4700959582754711;
    lbx0[2] = 1.2782744528515448;
    ubx0[2] = 1.2782744528515448;
    lbx0[3] = 0.9697369054729348;
    ubx0[3] = 0.9697369054729348;
    lbx0[4] = -0.0024075125910284698;
    ubx0[4] = -0.0024075125910284698;
    lbx0[5] = 0.005911448868071184;
    ubx0[5] = 0.005911448868071184;
    lbx0[6] = 0.06062221061306733;
    ubx0[6] = 0.06062221061306733;
    lbx0[7] = 0.5340135670265422;
    ubx0[7] = 0.5340135670265422;
    lbx0[8] = 0.35107170051556463;
    ubx0[8] = 0.35107170051556463;
    lbx0[9] = 0.011209180476735569;
    ubx0[9] = 0.011209180476735569;
    lbx0[10] = -0.012376512969304442;
    ubx0[10] = -0.012376512969304442;
    lbx0[11] = 0.06672057237242282;
    ubx0[11] = 0.06672057237242282;
    lbx0[12] = 0.21398147500431325;
    ubx0[12] = 0.21398147500431325;
    lbx0[13] = 83.48241786144587;
    ubx0[13] = 83.48241786144587;
    lbx0[14] = 10.229882753083082;
    ubx0[14] = 10.229882753083082;
    lbx0[15] = -0.122173046995634;
    ubx0[15] = -0.122173046995634;
    lbx0[16] = -0.1221730475894728;
    ubx0[16] = -0.1221730475894728;
    lbx0[17] = 794.1899202110542;
    ubx0[17] = 794.1899202110542;
    lbx0[18] = 792.9833844760742;
    ubx0[18] = 792.9833844760742;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);

    // initialization for state values
    double x_init[NX];
    x_init[0] = 0.0;
    x_init[1] = 0.0;
    x_init[2] = 0.0;
    x_init[3] = 0.0;
    x_init[4] = 0.0;
    x_init[5] = 0.0;
    x_init[6] = 0.0;
    x_init[7] = 0.0;
    x_init[8] = 0.0;
    x_init[9] = 0.0;
    x_init[10] = 0.0;
    x_init[11] = 0.0;
    x_init[12] = 0.0;
    x_init[13] = 0.0;
    x_init[14] = 0.0;
    x_init[15] = 0.0;
    x_init[16] = 0.0;
    x_init[17] = 0.0;
    x_init[18] = 0.0;

    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    u0[4] = 0.0;
    u0[5] = 0.0;

    // prepare evaluation
    int NTIMINGS = 1;
    double min_time = 1e12;
    double kkt_norm_inf;
    double elapsed_time;
    int sqp_iter;

    double xtraj[NX * (N+1)];
    double utraj[NU * N];

    // solve ocp in loop
    for (int ii = 0; ii < NTIMINGS; ii++)
    {
        // initialize solution
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x_init);
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
        }
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x_init);
        status = SAM_equation_system_acados_solve(acados_ocp_capsule);
        ocp_nlp_get(nlp_solver, "time_tot", &elapsed_time);
        min_time = MIN(elapsed_time, min_time);
    }

    /* print solution and statistics */
    for (int ii = 0; ii <= nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "x", &xtraj[ii*NX]);
    for (int ii = 0; ii < nlp_dims->N; ii++)
        ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, ii, "u", &utraj[ii*NU]);

    printf("\n--- xtraj ---\n");
    d_print_exp_tran_mat( NX, N+1, xtraj, NX);
    printf("\n--- utraj ---\n");
    d_print_exp_tran_mat( NU, N, utraj, NU );
    // ocp_nlp_out_print(nlp_solver->dims, nlp_out);

    printf("\nsolved ocp %d times, solution printed above\n\n", NTIMINGS);

    if (status == ACADOS_SUCCESS)
    {
        printf("SAM_equation_system_acados_solve(): SUCCESS!\n");
    }
    else
    {
        printf("SAM_equation_system_acados_solve() failed with status %d.\n", status);
    }

    // get solution
    ocp_nlp_out_get(nlp_config, nlp_dims, nlp_out, 0, "kkt_norm_inf", &kkt_norm_inf);
    ocp_nlp_get(nlp_solver, "sqp_iter", &sqp_iter);

    SAM_equation_system_acados_print_stats(acados_ocp_capsule);

    printf("\nSolver info:\n");
    printf(" SQP iterations %2d\n minimum time for %d solve %f [ms]\n KKT %e\n",
           sqp_iter, NTIMINGS, min_time*1000, kkt_norm_inf);



    // free solver
    status = SAM_equation_system_acados_free(acados_ocp_capsule);
    if (status) {
        printf("SAM_equation_system_acados_free() returned status %d. \n", status);
    }
    // free solver capsule
    status = SAM_equation_system_acados_free_capsule(acados_ocp_capsule);
    if (status) {
        printf("SAM_equation_system_acados_free_capsule() returned status %d. \n", status);
    }

    return status;
}
