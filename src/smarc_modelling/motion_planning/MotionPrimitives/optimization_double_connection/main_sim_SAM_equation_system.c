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
#include "acados_c/sim_interface.h"
#include "acados_sim_solver_SAM_equation_system.h"

#define NX     SAM_EQUATION_SYSTEM_NX
#define NZ     SAM_EQUATION_SYSTEM_NZ
#define NU     SAM_EQUATION_SYSTEM_NU
#define NP     SAM_EQUATION_SYSTEM_NP


int main()
{
    int status = 0;
    SAM_equation_system_sim_solver_capsule *capsule = SAM_equation_system_acados_sim_solver_create_capsule();
    status = SAM_equation_system_acados_sim_create(capsule);

    if (status)
    {
        printf("acados_create() returned status %d. Exiting.\n", status);
        exit(1);
    }

    sim_config *acados_sim_config = SAM_equation_system_acados_get_sim_config(capsule);
    sim_in *acados_sim_in = SAM_equation_system_acados_get_sim_in(capsule);
    sim_out *acados_sim_out = SAM_equation_system_acados_get_sim_out(capsule);
    void *acados_sim_dims = SAM_equation_system_acados_get_sim_dims(capsule);

    // initial condition
    double x_current[NX];
    x_current[0] = 0.0;
    x_current[1] = 0.0;
    x_current[2] = 0.0;
    x_current[3] = 0.0;
    x_current[4] = 0.0;
    x_current[5] = 0.0;
    x_current[6] = 0.0;
    x_current[7] = 0.0;
    x_current[8] = 0.0;
    x_current[9] = 0.0;
    x_current[10] = 0.0;
    x_current[11] = 0.0;
    x_current[12] = 0.0;
    x_current[13] = 0.0;
    x_current[14] = 0.0;
    x_current[15] = 0.0;
    x_current[16] = 0.0;
    x_current[17] = 0.0;
    x_current[18] = 0.0;

  
    x_current[0] = 4.557752171945946;
    x_current[1] = -0.7830821578868158;
    x_current[2] = 0.4395271245832924;
    x_current[3] = 0.9472029839715882;
    x_current[4] = -0.01339096849137078;
    x_current[5] = -0.05265761139411095;
    x_current[6] = -0.3229301281944523;
    x_current[7] = 1.3246326388579783;
    x_current[8] = 0.23557194081263383;
    x_current[9] = -0.013921859628079757;
    x_current[10] = -0.05482385732213416;
    x_current[11] = -0.03474024185515258;
    x_current[12] = -0.24977149253323003;
    x_current[13] = 79;
    x_current[14] = 93.5;
    x_current[15] = 0;
    x_current[16] = 0.08726646259971647;
    x_current[17] = 600;
    x_current[18] = 600;
    
  


    // initial value for control input
    double u0[NU];
    u0[0] = 0.0;
    u0[1] = 0.0;
    u0[2] = 0.0;
    u0[3] = 0.0;
    u0[4] = 0.0;
    u0[5] = 0.0;

  


    int n_sim_steps = 3;
    // solve ocp in loop
    for (int ii = 0; ii < n_sim_steps; ii++)
    {
        // set inputs
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "x", x_current);
        sim_in_set(acados_sim_config, acados_sim_dims,
            acados_sim_in, "u", u0);

        // solve
        status = SAM_equation_system_acados_sim_solve(capsule);
        if (status != ACADOS_SUCCESS)
        {
            printf("acados_solve() failed with status %d.\n", status);
        }

        // get outputs
        sim_out_get(acados_sim_config, acados_sim_dims,
               acados_sim_out, "x", x_current);

    

        // print solution
        printf("\nx_current, %d\n", ii);
        for (int jj = 0; jj < NX; jj++)
        {
            printf("%e\n", x_current[jj]);
        }
    }

    printf("\nPerformed %d simulation steps with acados integrator successfully.\n\n", n_sim_steps);

    // free solver
    status = SAM_equation_system_acados_sim_free(capsule);
    if (status) {
        printf("SAM_equation_system_acados_sim_free() returned status %d. \n", status);
    }

    SAM_equation_system_acados_sim_solver_free_capsule(capsule);

    return status;
}
