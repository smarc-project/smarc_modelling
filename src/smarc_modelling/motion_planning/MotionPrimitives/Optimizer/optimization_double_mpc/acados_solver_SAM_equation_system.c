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
#include <assert.h>
// acados
// #include "acados/utils/print.h"
#include "acados_c/ocp_nlp_interface.h"
#include "acados_c/external_function_interface.h"

// example specific
#include "SAM_equation_system_model/SAM_equation_system_model.h"


#include "SAM_equation_system_constraints/SAM_equation_system_constraints.h"
#include "SAM_equation_system_cost/SAM_equation_system_cost.h"



#include "acados_solver_SAM_equation_system.h"

#define NX     SAM_EQUATION_SYSTEM_NX
#define NZ     SAM_EQUATION_SYSTEM_NZ
#define NU     SAM_EQUATION_SYSTEM_NU
#define NP     SAM_EQUATION_SYSTEM_NP
#define NP_GLOBAL     SAM_EQUATION_SYSTEM_NP_GLOBAL
#define NY0    SAM_EQUATION_SYSTEM_NY0
#define NY     SAM_EQUATION_SYSTEM_NY
#define NYN    SAM_EQUATION_SYSTEM_NYN

#define NBX    SAM_EQUATION_SYSTEM_NBX
#define NBX0   SAM_EQUATION_SYSTEM_NBX0
#define NBU    SAM_EQUATION_SYSTEM_NBU
#define NG     SAM_EQUATION_SYSTEM_NG
#define NBXN   SAM_EQUATION_SYSTEM_NBXN
#define NGN    SAM_EQUATION_SYSTEM_NGN

#define NH     SAM_EQUATION_SYSTEM_NH
#define NHN    SAM_EQUATION_SYSTEM_NHN
#define NH0    SAM_EQUATION_SYSTEM_NH0
#define NPHI   SAM_EQUATION_SYSTEM_NPHI
#define NPHIN  SAM_EQUATION_SYSTEM_NPHIN
#define NPHI0  SAM_EQUATION_SYSTEM_NPHI0
#define NR     SAM_EQUATION_SYSTEM_NR

#define NS     SAM_EQUATION_SYSTEM_NS
#define NS0    SAM_EQUATION_SYSTEM_NS0
#define NSN    SAM_EQUATION_SYSTEM_NSN

#define NSBX   SAM_EQUATION_SYSTEM_NSBX
#define NSBU   SAM_EQUATION_SYSTEM_NSBU
#define NSH0   SAM_EQUATION_SYSTEM_NSH0
#define NSH    SAM_EQUATION_SYSTEM_NSH
#define NSHN   SAM_EQUATION_SYSTEM_NSHN
#define NSG    SAM_EQUATION_SYSTEM_NSG
#define NSPHI0 SAM_EQUATION_SYSTEM_NSPHI0
#define NSPHI  SAM_EQUATION_SYSTEM_NSPHI
#define NSPHIN SAM_EQUATION_SYSTEM_NSPHIN
#define NSGN   SAM_EQUATION_SYSTEM_NSGN
#define NSBXN  SAM_EQUATION_SYSTEM_NSBXN



// ** solver data **

SAM_equation_system_solver_capsule * SAM_equation_system_acados_create_capsule(void)
{
    void* capsule_mem = malloc(sizeof(SAM_equation_system_solver_capsule));
    SAM_equation_system_solver_capsule *capsule = (SAM_equation_system_solver_capsule *) capsule_mem;

    return capsule;
}


int SAM_equation_system_acados_free_capsule(SAM_equation_system_solver_capsule *capsule)
{
    free(capsule);
    return 0;
}


int SAM_equation_system_acados_create(SAM_equation_system_solver_capsule* capsule)
{
    int N_shooting_intervals = SAM_EQUATION_SYSTEM_N;
    double* new_time_steps = NULL; // NULL -> don't alter the code generated time-steps
    return SAM_equation_system_acados_create_with_discretization(capsule, N_shooting_intervals, new_time_steps);
}


int SAM_equation_system_acados_update_time_steps(SAM_equation_system_solver_capsule* capsule, int N, double* new_time_steps)
{
    if (N != capsule->nlp_solver_plan->N) {
        fprintf(stderr, "SAM_equation_system_acados_update_time_steps: given number of time steps (= %d) " \
            "differs from the currently allocated number of " \
            "time steps (= %d)!\n" \
            "Please recreate with new discretization and provide a new vector of time_stamps!\n",
            N, capsule->nlp_solver_plan->N);
        return 1;
    }

    ocp_nlp_config * nlp_config = capsule->nlp_config;
    ocp_nlp_dims * nlp_dims = capsule->nlp_dims;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &new_time_steps[i]);
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &new_time_steps[i]);
    }
    return 0;
}

/**
 * Internal function for SAM_equation_system_acados_create: step 1
 */
void SAM_equation_system_acados_create_set_plan(ocp_nlp_plan_t* nlp_solver_plan, const int N)
{
    assert(N == nlp_solver_plan->N);

    /************************************************
    *  plan
    ************************************************/

    nlp_solver_plan->nlp_solver = SQP_RTI;

    nlp_solver_plan->ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;
    nlp_solver_plan->relaxed_ocp_qp_solver_plan.qp_solver = PARTIAL_CONDENSING_HPIPM;

    nlp_solver_plan->nlp_cost[0] = NONLINEAR_LS;
    for (int i = 1; i < N; i++)
        nlp_solver_plan->nlp_cost[i] = NONLINEAR_LS;

    nlp_solver_plan->nlp_cost[N] = NONLINEAR_LS;

    for (int i = 0; i < N; i++)
    {
        nlp_solver_plan->nlp_dynamics[i] = CONTINUOUS_MODEL;
        nlp_solver_plan->sim_solver_plan[i].sim_solver = IRK;
    }

    nlp_solver_plan->nlp_constraints[0] = BGH;

    for (int i = 1; i < N; i++)
    {
        nlp_solver_plan->nlp_constraints[i] = BGH;
    }
    nlp_solver_plan->nlp_constraints[N] = BGH;

    nlp_solver_plan->regularization = NO_REGULARIZE;

    nlp_solver_plan->globalization = MERIT_BACKTRACKING;
}


static ocp_nlp_dims* SAM_equation_system_acados_create_setup_dimensions(SAM_equation_system_solver_capsule* capsule)
{
    ocp_nlp_plan_t* nlp_solver_plan = capsule->nlp_solver_plan;
    const int N = nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;

    /************************************************
    *  dimensions
    ************************************************/
    #define NINTNP1MEMS 18
    int* intNp1mem = (int*)malloc( (N+1)*sizeof(int)*NINTNP1MEMS );

    int* nx    = intNp1mem + (N+1)*0;
    int* nu    = intNp1mem + (N+1)*1;
    int* nbx   = intNp1mem + (N+1)*2;
    int* nbu   = intNp1mem + (N+1)*3;
    int* nsbx  = intNp1mem + (N+1)*4;
    int* nsbu  = intNp1mem + (N+1)*5;
    int* nsg   = intNp1mem + (N+1)*6;
    int* nsh   = intNp1mem + (N+1)*7;
    int* nsphi = intNp1mem + (N+1)*8;
    int* ns    = intNp1mem + (N+1)*9;
    int* ng    = intNp1mem + (N+1)*10;
    int* nh    = intNp1mem + (N+1)*11;
    int* nphi  = intNp1mem + (N+1)*12;
    int* nz    = intNp1mem + (N+1)*13;
    int* ny    = intNp1mem + (N+1)*14;
    int* nr    = intNp1mem + (N+1)*15;
    int* nbxe  = intNp1mem + (N+1)*16;
    int* np  = intNp1mem + (N+1)*17;

    for (int i = 0; i < N+1; i++)
    {
        // common
        nx[i]     = NX;
        nu[i]     = NU;
        nz[i]     = NZ;
        ns[i]     = NS;
        // cost
        ny[i]     = NY;
        // constraints
        nbx[i]    = NBX;
        nbu[i]    = NBU;
        nsbx[i]   = NSBX;
        nsbu[i]   = NSBU;
        nsg[i]    = NSG;
        nsh[i]    = NSH;
        nsphi[i]  = NSPHI;
        ng[i]     = NG;
        nh[i]     = NH;
        nphi[i]   = NPHI;
        nr[i]     = NR;
        nbxe[i]   = 0;
        np[i]     = NP;
    }

    // for initial state
    nbx[0] = NBX0;
    nsbx[0] = 0;
    ns[0] = NS0;
    nbxe[0] = 19;
    ny[0] = NY0;
    nh[0] = NH0;
    nsh[0] = NSH0;
    nsphi[0] = NSPHI0;
    nphi[0] = NPHI0;


    // terminal - common
    nu[N]   = 0;
    nz[N]   = 0;
    ns[N]   = NSN;
    // cost
    ny[N]   = NYN;
    // constraint
    nbx[N]   = NBXN;
    nbu[N]   = 0;
    ng[N]    = NGN;
    nh[N]    = NHN;
    nphi[N]  = NPHIN;
    nr[N]    = 0;

    nsbx[N]  = NSBXN;
    nsbu[N]  = 0;
    nsg[N]   = NSGN;
    nsh[N]   = NSHN;
    nsphi[N] = NSPHIN;

    /* create and set ocp_nlp_dims */
    ocp_nlp_dims * nlp_dims = ocp_nlp_dims_create(nlp_config);

    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nx", nx);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nu", nu);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "nz", nz);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "ns", ns);
    ocp_nlp_dims_set_opt_vars(nlp_config, nlp_dims, "np", np);

    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "np_global", 0);
    ocp_nlp_dims_set_global(nlp_config, nlp_dims, "n_global_data", 0);

    for (int i = 0; i <= N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbx", &nbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbu", &nbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbx", &nsbx[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsbu", &nsbu[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "ng", &ng[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsg", &nsg[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nbxe", &nbxe[i]);
    }
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, 0, "ny", &ny[0]);
    for (int i = 1; i < N; i++)
        ocp_nlp_dims_set_cost(nlp_config, nlp_dims, i, "ny", &ny[i]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nh", &nh[0]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, 0, "nsh", &nsh[0]);

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nh", &nh[i]);
        ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, i, "nsh", &nsh[i]);
    }
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nh", &nh[N]);
    ocp_nlp_dims_set_constraints(nlp_config, nlp_dims, N, "nsh", &nsh[N]);
    ocp_nlp_dims_set_cost(nlp_config, nlp_dims, N, "ny", &ny[N]);

    free(intNp1mem);

    return nlp_dims;
}


/**
 * Internal function for SAM_equation_system_acados_create: step 3
 */
void SAM_equation_system_acados_create_setup_functions(SAM_equation_system_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;

    /************************************************
    *  external functions
    ************************************************/

#define MAP_CASADI_FNC(__CAPSULE_FNC__, __MODEL_BASE_FNC__) do{ \
        capsule->__CAPSULE_FNC__.casadi_fun = & __MODEL_BASE_FNC__ ;\
        capsule->__CAPSULE_FNC__.casadi_n_in = & __MODEL_BASE_FNC__ ## _n_in; \
        capsule->__CAPSULE_FNC__.casadi_n_out = & __MODEL_BASE_FNC__ ## _n_out; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_in = & __MODEL_BASE_FNC__ ## _sparsity_in; \
        capsule->__CAPSULE_FNC__.casadi_sparsity_out = & __MODEL_BASE_FNC__ ## _sparsity_out; \
        capsule->__CAPSULE_FNC__.casadi_work = & __MODEL_BASE_FNC__ ## _work; \
        external_function_external_param_casadi_create(&capsule->__CAPSULE_FNC__, &ext_fun_opts); \
    } while(false)

    external_function_opts ext_fun_opts;
    external_function_opts_set_to_default(&ext_fun_opts);


    ext_fun_opts.external_workspace = true;
    // constraints.constr_type == "BGH" and dims.nh > 0
    capsule->nl_constr_h_fun_jac = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun_jac[i], SAM_equation_system_constr_h_fun_jac_uxt_zt);
    }
    capsule->nl_constr_h_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++) {
        MAP_CASADI_FNC(nl_constr_h_fun[i], SAM_equation_system_constr_h_fun);
    }

    // nonlinear least squares function
    MAP_CASADI_FNC(cost_y_0_fun, SAM_equation_system_cost_y_0_fun);
    MAP_CASADI_FNC(cost_y_0_fun_jac_ut_xt, SAM_equation_system_cost_y_0_fun_jac_ut_xt);




    // implicit dae
    capsule->impl_dae_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun[i], SAM_equation_system_impl_dae_fun);
    }

    capsule->impl_dae_fun_jac_x_xdot_z = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_fun_jac_x_xdot_z[i], SAM_equation_system_impl_dae_fun_jac_x_xdot_z);
    }

    capsule->impl_dae_jac_x_xdot_u_z = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*N);
    for (int i = 0; i < N; i++) {
        MAP_CASADI_FNC(impl_dae_jac_x_xdot_u_z[i], SAM_equation_system_impl_dae_jac_x_xdot_u_z);
    }

    // nonlinear least squares cost
    capsule->cost_y_fun = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(cost_y_fun[i], SAM_equation_system_cost_y_fun);
    }

    capsule->cost_y_fun_jac_ut_xt = (external_function_external_param_casadi *) malloc(sizeof(external_function_external_param_casadi)*(N-1));
    for (int i = 0; i < N-1; i++)
    {
        MAP_CASADI_FNC(cost_y_fun_jac_ut_xt[i], SAM_equation_system_cost_y_fun_jac_ut_xt);
    }
    // nonlinear least square function
    MAP_CASADI_FNC(cost_y_e_fun, SAM_equation_system_cost_y_e_fun);
    MAP_CASADI_FNC(cost_y_e_fun_jac_ut_xt, SAM_equation_system_cost_y_e_fun_jac_ut_xt);

#undef MAP_CASADI_FNC
}


/**
 * Internal function for SAM_equation_system_acados_create: step 4
 */
void SAM_equation_system_acados_create_set_default_parameters(SAM_equation_system_solver_capsule* capsule)
{

    const int N = capsule->nlp_solver_plan->N;
    // initialize parameters to nominal value
    double* p = calloc(NP, sizeof(double));

    for (int i = 0; i <= N; i++) {
        SAM_equation_system_acados_update_params(capsule, i, p, NP);
    }
    free(p);


    // no global parameters defined
}


/**
 * Internal function for SAM_equation_system_acados_create: step 5
 */
void SAM_equation_system_acados_setup_nlp_in(SAM_equation_system_solver_capsule* capsule, const int N, double* new_time_steps)
{
    assert(N == capsule->nlp_solver_plan->N);
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;

    int tmp_int = 0;

    /************************************************
    *  nlp_in
    ************************************************/
//    ocp_nlp_in * nlp_in = ocp_nlp_in_create(nlp_config, nlp_dims);
//    capsule->nlp_in = nlp_in;
    ocp_nlp_in * nlp_in = capsule->nlp_in;

    // set up time_steps and cost_scaling

    if (new_time_steps)
    {
        // NOTE: this sets scaling and time_steps
        SAM_equation_system_acados_update_time_steps(capsule, N, new_time_steps);
    }
    else
    {
        // set time_steps
    
        double time_step = 0.1;
        for (int i = 0; i < N; i++)
        {
            ocp_nlp_in_set(nlp_config, nlp_dims, nlp_in, i, "Ts", &time_step);
        }
        // set cost scaling
        double* cost_scaling = malloc((N+1)*sizeof(double));
        cost_scaling[0] = 0.1;
        cost_scaling[1] = 0.1;
        cost_scaling[2] = 0.1;
        cost_scaling[3] = 0.1;
        cost_scaling[4] = 0.1;
        cost_scaling[5] = 0.1;
        cost_scaling[6] = 0.1;
        cost_scaling[7] = 0.1;
        cost_scaling[8] = 0.1;
        cost_scaling[9] = 0.1;
        cost_scaling[10] = 0.1;
        cost_scaling[11] = 0.1;
        cost_scaling[12] = 0.1;
        cost_scaling[13] = 0.1;
        cost_scaling[14] = 0.1;
        cost_scaling[15] = 0.1;
        cost_scaling[16] = 0.1;
        cost_scaling[17] = 0.1;
        cost_scaling[18] = 0.1;
        cost_scaling[19] = 0.1;
        cost_scaling[20] = 0.1;
        cost_scaling[21] = 0.1;
        cost_scaling[22] = 0.1;
        cost_scaling[23] = 0.1;
        cost_scaling[24] = 0.1;
        cost_scaling[25] = 1;
        for (int i = 0; i <= N; i++)
        {
            ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "scaling", &cost_scaling[i]);
        }
        free(cost_scaling);
    }


    /**** Dynamics ****/
    for (int i = 0; i < N; i++)
    {
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "impl_dae_fun", &capsule->impl_dae_fun[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_fun_jac_x_xdot_z", &capsule->impl_dae_fun_jac_x_xdot_z[i]);
        ocp_nlp_dynamics_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i,
                                   "impl_dae_jac_x_xdot_u", &capsule->impl_dae_jac_x_xdot_u_z[i]);
    }

    /**** Cost ****/
    double* yref_0 = calloc(NY0, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "yref", yref_0);
    free(yref_0);

   double* W_0 = calloc(NY0*NY0, sizeof(double));
    // change only the non-zero elements:
    W_0[0+(NY0) * 0] = 150;
    W_0[1+(NY0) * 1] = 150;
    W_0[2+(NY0) * 2] = 150;
    W_0[3+(NY0) * 3] = 1500;
    W_0[4+(NY0) * 4] = 1500;
    W_0[5+(NY0) * 5] = 1500;
    W_0[6+(NY0) * 6] = 1500;
    W_0[7+(NY0) * 7] = 130;
    W_0[8+(NY0) * 8] = 130;
    W_0[9+(NY0) * 9] = 130;
    W_0[10+(NY0) * 10] = 100;
    W_0[11+(NY0) * 11] = 100;
    W_0[12+(NY0) * 12] = 100;
    W_0[19+(NY0) * 19] = 0.001;
    W_0[20+(NY0) * 20] = 0.001;
    W_0[21+(NY0) * 21] = 1;
    W_0[22+(NY0) * 22] = 1;
    W_0[23+(NY0) * 23] = 0.00001;
    W_0[24+(NY0) * 24] = 0.00001;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, 0, "W", W_0);
    free(W_0);
    double* yref = calloc(NY, sizeof(double));
    // change only the non-zero elements:

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "yref", yref);
    }
    free(yref);
    double* W = calloc(NY*NY, sizeof(double));
    // change only the non-zero elements:
    W[0+(NY) * 0] = 150;
    W[1+(NY) * 1] = 150;
    W[2+(NY) * 2] = 150;
    W[3+(NY) * 3] = 1500;
    W[4+(NY) * 4] = 1500;
    W[5+(NY) * 5] = 1500;
    W[6+(NY) * 6] = 1500;
    W[7+(NY) * 7] = 130;
    W[8+(NY) * 8] = 130;
    W[9+(NY) * 9] = 130;
    W[10+(NY) * 10] = 100;
    W[11+(NY) * 11] = 100;
    W[12+(NY) * 12] = 100;
    W[19+(NY) * 19] = 0.001;
    W[20+(NY) * 20] = 0.001;
    W[21+(NY) * 21] = 1;
    W[22+(NY) * 22] = 1;
    W[23+(NY) * 23] = 0.00001;
    W[24+(NY) * 24] = 0.00001;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, i, "W", W);
    }
    free(W);
    double* yref_e = calloc(NYN, sizeof(double));
    // change only the non-zero elements:
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "yref", yref_e);
    free(yref_e);

    double* W_e = calloc(NYN*NYN, sizeof(double));
    // change only the non-zero elements:
    W_e[0+(NYN) * 0] = 150;
    W_e[1+(NYN) * 1] = 150;
    W_e[2+(NYN) * 2] = 150;
    W_e[3+(NYN) * 3] = 1500;
    W_e[4+(NYN) * 4] = 1500;
    W_e[5+(NYN) * 5] = 1500;
    W_e[6+(NYN) * 6] = 1500;
    W_e[7+(NYN) * 7] = 130;
    W_e[8+(NYN) * 8] = 130;
    W_e[9+(NYN) * 9] = 130;
    W_e[10+(NYN) * 10] = 100;
    W_e[11+(NYN) * 11] = 100;
    W_e[12+(NYN) * 12] = 100;
    ocp_nlp_cost_model_set(nlp_config, nlp_dims, nlp_in, N, "W", W_e);
    free(W_e);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun", &capsule->cost_y_0_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, 0, "nls_y_fun_jac", &capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 1; i < N; i++)
    {
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun", &capsule->cost_y_fun[i-1]);
        ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nls_y_fun_jac", &capsule->cost_y_fun_jac_ut_xt[i-1]);
    }
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun", &capsule->cost_y_e_fun);
    ocp_nlp_cost_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, N, "nls_y_fun_jac", &capsule->cost_y_e_fun_jac_ut_xt);






    /**** Constraints ****/

    // bounds for initial stage
    // x0
    int* idxbx0 = malloc(NBX0 * sizeof(int));
    idxbx0[0] = 0;
    idxbx0[1] = 1;
    idxbx0[2] = 2;
    idxbx0[3] = 3;
    idxbx0[4] = 4;
    idxbx0[5] = 5;
    idxbx0[6] = 6;
    idxbx0[7] = 7;
    idxbx0[8] = 8;
    idxbx0[9] = 9;
    idxbx0[10] = 10;
    idxbx0[11] = 11;
    idxbx0[12] = 12;
    idxbx0[13] = 13;
    idxbx0[14] = 14;
    idxbx0[15] = 15;
    idxbx0[16] = 16;
    idxbx0[17] = 17;
    idxbx0[18] = 18;

    double* lubx0 = calloc(2*NBX0, sizeof(double));
    double* lbx0 = lubx0;
    double* ubx0 = lubx0 + NBX0;
    // change only the non-zero elements:
    lbx0[0] = 5.384794092572777;
    ubx0[0] = 5.384794092572777;
    lbx0[1] = -1.1898531369999112;
    ubx0[1] = -1.1898531369999112;
    lbx0[2] = 0.5572830606077207;
    ubx0[2] = 0.5572830606077207;
    lbx0[3] = 0.937342685798445;
    ubx0[3] = 0.937342685798445;
    lbx0[4] = -0.016601626767975157;
    ubx0[4] = -0.016601626767975157;
    lbx0[5] = -0.046210334096826754;
    ubx0[5] = -0.046210334096826754;
    lbx0[6] = -0.34071818500045375;
    ubx0[6] = -0.34071818500045375;
    lbx0[7] = 0.00868913794358206;
    ubx0[7] = 0.00868913794358206;
    lbx0[8] = 0.18340816215062372;
    ubx0[8] = 0.18340816215062372;
    lbx0[9] = 0.052034534298667535;
    ubx0[9] = 0.052034534298667535;
    lbx0[10] = 0.0005557344399972908;
    ubx0[10] = 0.0005557344399972908;
    lbx0[11] = 0.07089391298937316;
    ubx0[11] = 0.07089391298937316;
    lbx0[12] = 0.1803275053833251;
    ubx0[12] = 0.1803275053833251;
    lbx0[13] = 75.55677766503656;
    ubx0[13] = 75.55677766503656;
    lbx0[14] = 91.36085621225766;
    ubx0[14] = 91.36085621225766;
    lbx0[15] = 0.1221730442165815;
    ubx0[15] = 0.1221730442165815;
    lbx0[16] = 0.12217304591780558;
    ubx0[16] = 0.12217304591780558;
    lbx0[17] = -300.0000022930386;
    ubx0[17] = -300.0000022930386;
    lbx0[18] = -300.0000024776283;
    ubx0[18] = -300.0000024776283;

    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbx", idxbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "lbx", lbx0);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "ubx", ubx0);
    free(idxbx0);
    free(lubx0);
    // idxbxe_0
    int* idxbxe_0 = malloc(19 * sizeof(int));
    idxbxe_0[0] = 0;
    idxbxe_0[1] = 1;
    idxbxe_0[2] = 2;
    idxbxe_0[3] = 3;
    idxbxe_0[4] = 4;
    idxbxe_0[5] = 5;
    idxbxe_0[6] = 6;
    idxbxe_0[7] = 7;
    idxbxe_0[8] = 8;
    idxbxe_0[9] = 9;
    idxbxe_0[10] = 10;
    idxbxe_0[11] = 11;
    idxbxe_0[12] = 12;
    idxbxe_0[13] = 13;
    idxbxe_0[14] = 14;
    idxbxe_0[15] = 15;
    idxbxe_0[16] = 16;
    idxbxe_0[17] = 17;
    idxbxe_0[18] = 18;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, 0, "idxbxe", idxbxe_0);
    free(idxbxe_0);








    /* constraints that are the same for initial and intermediate */
    // u
    int* idxbu = malloc(NBU * sizeof(int));
    idxbu[0] = 0;
    idxbu[1] = 1;
    idxbu[2] = 2;
    idxbu[3] = 3;
    idxbu[4] = 4;
    idxbu[5] = 5;
    double* lubu = calloc(2*NBU, sizeof(double));
    double* lbu = lubu;
    double* ubu = lubu + NBU;
    lbu[0] = -10;
    ubu[0] = 10;
    lbu[1] = -15;
    ubu[1] = 15;
    lbu[2] = -7;
    ubu[2] = 7;
    lbu[3] = -7;
    ubu[3] = 7;
    lbu[4] = -1000;
    ubu[4] = 1000;
    lbu[5] = -1000;
    ubu[5] = 1000;

    for (int i = 0; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbu", idxbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbu", lbu);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubu", ubu);
    }
    free(idxbu);
    free(lubu);








    // x
    int* idxbx = malloc(NBX * sizeof(int));
    idxbx[0] = 0;
    idxbx[1] = 1;
    idxbx[2] = 2;
    idxbx[3] = 3;
    idxbx[4] = 4;
    idxbx[5] = 5;
    idxbx[6] = 6;
    idxbx[7] = 7;
    idxbx[8] = 8;
    idxbx[9] = 9;
    idxbx[10] = 10;
    idxbx[11] = 11;
    idxbx[12] = 12;
    idxbx[13] = 13;
    idxbx[14] = 14;
    idxbx[15] = 15;
    idxbx[16] = 16;
    idxbx[17] = 17;
    idxbx[18] = 18;
    double* lubx = calloc(2*NBX, sizeof(double));
    double* lbx = lubx;
    double* ubx = lubx + NBX;
    lbx[0] = -1000;
    ubx[0] = 1000;
    lbx[1] = -1000;
    ubx[1] = 1000;
    lbx[2] = -1000;
    ubx[2] = 1000;
    lbx[3] = -1000;
    ubx[3] = 1000;
    lbx[4] = -1000;
    ubx[4] = 1000;
    lbx[5] = -1000;
    ubx[5] = 1000;
    lbx[6] = -1000;
    ubx[6] = 1000;
    lbx[7] = -1000;
    ubx[7] = 1000;
    lbx[8] = -1000;
    ubx[8] = 1000;
    lbx[9] = -1000;
    ubx[9] = 1000;
    lbx[10] = -1000;
    ubx[10] = 1000;
    lbx[11] = -1000;
    ubx[11] = 1000;
    lbx[12] = -1000;
    ubx[12] = 1000;
    ubx[13] = 100;
    ubx[14] = 100;
    lbx[15] = -7;
    ubx[15] = 7;
    lbx[16] = -7;
    ubx[16] = 7;
    lbx[17] = -1000;
    ubx[17] = 1000;
    lbx[18] = -1000;
    ubx[18] = 1000;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "idxbx", idxbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lbx", lbx);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "ubx", ubx);
    }
    free(idxbx);
    free(lubx);




    // set up nonlinear constraints for stage 1 to N-1
    double* luh = calloc(2*NH, sizeof(double));
    double* lh = luh;
    double* uh = luh + NH;
    lh[0] = 0.1;
    lh[1] = -2.4;
    lh[2] = -0.4;
    lh[3] = 0.1;
    lh[4] = -2.4;
    lh[5] = -0.4;
    uh[0] = 9.9;
    uh[1] = 2.4;
    uh[2] = 2.9;
    uh[3] = 9.9;
    uh[4] = 2.4;
    uh[5] = 2.9;

    for (int i = 1; i < N; i++)
    {
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun_jac",
                                      &capsule->nl_constr_h_fun_jac[i-1]);
        ocp_nlp_constraints_model_set_external_param_fun(nlp_config, nlp_dims, nlp_in, i, "nl_constr_h_fun",
                                      &capsule->nl_constr_h_fun[i-1]);
        
        
        
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "lh", lh);
        ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, i, "uh", uh);
    }
    free(luh);



    /* terminal constraints */

    // set up bounds for last stage
    // x
    int* idxbx_e = malloc(NBXN * sizeof(int));
    idxbx_e[0] = 0;
    idxbx_e[1] = 1;
    idxbx_e[2] = 2;
    idxbx_e[3] = 3;
    idxbx_e[4] = 4;
    idxbx_e[5] = 5;
    idxbx_e[6] = 6;
    idxbx_e[7] = 7;
    idxbx_e[8] = 8;
    idxbx_e[9] = 9;
    idxbx_e[10] = 10;
    idxbx_e[11] = 11;
    idxbx_e[12] = 12;
    idxbx_e[13] = 13;
    idxbx_e[14] = 14;
    idxbx_e[15] = 15;
    idxbx_e[16] = 16;
    idxbx_e[17] = 17;
    idxbx_e[18] = 18;
    double* lubx_e = calloc(2*NBXN, sizeof(double));
    double* lbx_e = lubx_e;
    double* ubx_e = lubx_e + NBXN;
    lbx_e[0] = -1000;
    ubx_e[0] = 1000;
    lbx_e[1] = -1000;
    ubx_e[1] = 1000;
    lbx_e[2] = -1000;
    ubx_e[2] = 1000;
    lbx_e[3] = -1000;
    ubx_e[3] = 1000;
    lbx_e[4] = -1000;
    ubx_e[4] = 1000;
    lbx_e[5] = -1000;
    ubx_e[5] = 1000;
    lbx_e[6] = -1000;
    ubx_e[6] = 1000;
    lbx_e[7] = -1000;
    ubx_e[7] = 1000;
    lbx_e[8] = -1000;
    ubx_e[8] = 1000;
    lbx_e[9] = -1000;
    ubx_e[9] = 1000;
    lbx_e[10] = -1000;
    ubx_e[10] = 1000;
    lbx_e[11] = -1000;
    ubx_e[11] = 1000;
    lbx_e[12] = -1000;
    ubx_e[12] = 1000;
    ubx_e[13] = 100;
    ubx_e[14] = 100;
    lbx_e[15] = -7;
    ubx_e[15] = 7;
    lbx_e[16] = -7;
    ubx_e[16] = 7;
    lbx_e[17] = -1000;
    ubx_e[17] = 1000;
    lbx_e[18] = -1000;
    ubx_e[18] = 1000;
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "idxbx", idxbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "lbx", lbx_e);
    ocp_nlp_constraints_model_set(nlp_config, nlp_dims, nlp_in, N, "ubx", ubx_e);
    free(idxbx_e);
    free(lubx_e);












}


static void SAM_equation_system_acados_create_set_opts(SAM_equation_system_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    void *nlp_opts = capsule->nlp_opts;

    /************************************************
    *  opts
    ************************************************/



    int fixed_hess = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "fixed_hess", &fixed_hess);
    double globalization_alpha_min = 0.05;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_alpha_min", &globalization_alpha_min);

    double globalization_alpha_reduction = 0.7;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "globalization_alpha_reduction", &globalization_alpha_reduction);



    int globalization_line_search_use_sufficient_descent = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_line_search_use_sufficient_descent", &globalization_line_search_use_sufficient_descent);

    int globalization_use_SOC = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_use_SOC", &globalization_use_SOC);

    double globalization_eps_sufficient_descent = 0.0001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_eps_sufficient_descent", &globalization_eps_sufficient_descent);

    int with_solution_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_solution_sens_wrt_params", &with_solution_sens_wrt_params);

    int with_value_sens_wrt_params = false;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "with_value_sens_wrt_params", &with_value_sens_wrt_params);

    double solution_sens_qp_t_lam_min = 0.000000001;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "solution_sens_qp_t_lam_min", &solution_sens_qp_t_lam_min);

    int globalization_full_step_dual = 0;
    ocp_nlp_solver_opts_set(nlp_config, capsule->nlp_opts, "globalization_full_step_dual", &globalization_full_step_dual);

    // set collocation type (relevant for implicit integrators)
    sim_collocation_type collocation_type = GAUSS_LEGENDRE;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_collocation_type", &collocation_type);

    // set up sim_method_num_steps
    // all sim_method_num_steps are identical
    int sim_method_num_steps = 1;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_steps", &sim_method_num_steps);

    // set up sim_method_num_stages
    // all sim_method_num_stages are identical
    int sim_method_num_stages = 4;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_num_stages", &sim_method_num_stages);

    int newton_iter_val = 3;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_iter", &newton_iter_val);

    double newton_tol_val = 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_newton_tol", &newton_tol_val);

    // set up sim_method_jac_reuse
    bool tmp_bool = (bool) 0;
    for (int i = 0; i < N; i++)
        ocp_nlp_solver_opts_set_at_stage(nlp_config, nlp_opts, i, "dynamics_jac_reuse", &tmp_bool);

    double levenberg_marquardt = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "levenberg_marquardt", &levenberg_marquardt);

    /* options QP solver */
    int qp_solver_cond_N;const int qp_solver_cond_N_ori = 25;
    qp_solver_cond_N = N < qp_solver_cond_N_ori ? N : qp_solver_cond_N_ori; // use the minimum value here
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    int nlp_solver_ext_qp_res = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "ext_qp_res", &nlp_solver_ext_qp_res);

    bool store_iterates = false;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "store_iterates", &store_iterates);
    // set HPIPM mode: should be done before setting other QP solver options
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_hpipm_mode", "ROBUST");



    int qp_solver_t0_init = 2;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_t0_init", &qp_solver_t0_init);




    int as_rti_iter = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_iter", &as_rti_iter);

    int as_rti_level = 4;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "as_rti_level", &as_rti_level);

    int rti_log_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_residuals", &rti_log_residuals);

    int rti_log_only_available_residuals = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "rti_log_only_available_residuals", &rti_log_only_available_residuals);

    int qp_solver_iter_max = 50;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_iter_max", &qp_solver_iter_max);


    double qp_solver_tol_stat = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_stat", &qp_solver_tol_stat);
    double qp_solver_tol_eq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_eq", &qp_solver_tol_eq);
    double qp_solver_tol_ineq = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_ineq", &qp_solver_tol_ineq);
    double qp_solver_tol_comp = 0.000001;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_tol_comp", &qp_solver_tol_comp);

    int print_level = 0;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "print_level", &print_level);
    int qp_solver_cond_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_cond_ric_alg", &qp_solver_cond_ric_alg);

    int qp_solver_ric_alg = 1;
    ocp_nlp_solver_opts_set(nlp_config, nlp_opts, "qp_ric_alg", &qp_solver_ric_alg);


    int ext_cost_num_hess = 0;
}


/**
 * Internal function for SAM_equation_system_acados_create: step 7
 */
void SAM_equation_system_acados_set_nlp_out(SAM_equation_system_solver_capsule* capsule)
{
    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;

    // initialize primal solution
    double* xu0 = calloc(NX+NU, sizeof(double));
    double* x0 = xu0;

    // initialize with x0
    x0[0] = 5.384794092572777;
    x0[1] = -1.1898531369999112;
    x0[2] = 0.5572830606077207;
    x0[3] = 0.937342685798445;
    x0[4] = -0.016601626767975157;
    x0[5] = -0.046210334096826754;
    x0[6] = -0.34071818500045375;
    x0[7] = 0.00868913794358206;
    x0[8] = 0.18340816215062372;
    x0[9] = 0.052034534298667535;
    x0[10] = 0.0005557344399972908;
    x0[11] = 0.07089391298937316;
    x0[12] = 0.1803275053833251;
    x0[13] = 75.55677766503656;
    x0[14] = 91.36085621225766;
    x0[15] = 0.1221730442165815;
    x0[16] = 0.12217304591780558;
    x0[17] = -300.0000022930386;
    x0[18] = -300.0000024776283;


    double* u0 = xu0 + NX;

    for (int i = 0; i < N; i++)
    {
        // x0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", x0);
        // u0
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", u0);
    }
    ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, N, "x", x0);
    free(xu0);
}


/**
 * Internal function for SAM_equation_system_acados_create: step 9
 */
int SAM_equation_system_acados_create_precompute(SAM_equation_system_solver_capsule* capsule) {
    int status = ocp_nlp_precompute(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    if (status != ACADOS_SUCCESS) {
        printf("\nocp_nlp_precompute failed!\n\n");
        exit(1);
    }

    return status;
}


int SAM_equation_system_acados_create_with_discretization(SAM_equation_system_solver_capsule* capsule, int N, double* new_time_steps)
{
    // If N does not match the number of shooting intervals used for code generation, new_time_steps must be given.
    if (N != SAM_EQUATION_SYSTEM_N && !new_time_steps) {
        fprintf(stderr, "SAM_equation_system_acados_create_with_discretization: new_time_steps is NULL " \
            "but the number of shooting intervals (= %d) differs from the number of " \
            "shooting intervals (= %d) during code generation! Please provide a new vector of time_stamps!\n", \
             N, SAM_EQUATION_SYSTEM_N);
        return 1;
    }

    // number of expected runtime parameters
    capsule->nlp_np = NP;

    // 1) create and set nlp_solver_plan; create nlp_config
    capsule->nlp_solver_plan = ocp_nlp_plan_create(N);
    SAM_equation_system_acados_create_set_plan(capsule->nlp_solver_plan, N);
    capsule->nlp_config = ocp_nlp_config_create(*capsule->nlp_solver_plan);

    // 2) create and set dimensions
    capsule->nlp_dims = SAM_equation_system_acados_create_setup_dimensions(capsule);

    // 3) create and set nlp_opts
    capsule->nlp_opts = ocp_nlp_solver_opts_create(capsule->nlp_config, capsule->nlp_dims);
    SAM_equation_system_acados_create_set_opts(capsule);

    // 4) create nlp_in
    capsule->nlp_in = ocp_nlp_in_create(capsule->nlp_config, capsule->nlp_dims);

    // 5) setup functions, nlp_in and default parameters
    SAM_equation_system_acados_create_setup_functions(capsule);
    SAM_equation_system_acados_setup_nlp_in(capsule, N, new_time_steps);
    SAM_equation_system_acados_create_set_default_parameters(capsule);

    // 6) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // 7) create and set nlp_out
    // 7.1) nlp_out
    capsule->nlp_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    // 7.2) sens_out
    capsule->sens_out = ocp_nlp_out_create(capsule->nlp_config, capsule->nlp_dims);
    SAM_equation_system_acados_set_nlp_out(capsule);

    // 8) do precomputations
    int status = SAM_equation_system_acados_create_precompute(capsule);

    return status;
}

/**
 * This function is for updating an already initialized solver with a different number of qp_cond_N. It is useful for code reuse after code export.
 */
int SAM_equation_system_acados_update_qp_solver_cond_N(SAM_equation_system_solver_capsule* capsule, int qp_solver_cond_N)
{
    // 1) destroy solver
    ocp_nlp_solver_destroy(capsule->nlp_solver);

    // 2) set new value for "qp_cond_N"
    const int N = capsule->nlp_solver_plan->N;
    if(qp_solver_cond_N > N)
        printf("Warning: qp_solver_cond_N = %d > N = %d\n", qp_solver_cond_N, N);
    ocp_nlp_solver_opts_set(capsule->nlp_config, capsule->nlp_opts, "qp_cond_N", &qp_solver_cond_N);

    // 3) continue with the remaining steps from SAM_equation_system_acados_create_with_discretization(...):
    // -> 8) create solver
    capsule->nlp_solver = ocp_nlp_solver_create(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_opts, capsule->nlp_in);

    // -> 9) do precomputations
    int status = SAM_equation_system_acados_create_precompute(capsule);
    return status;
}


int SAM_equation_system_acados_reset(SAM_equation_system_solver_capsule* capsule, int reset_qp_solver_mem)
{

    // set initialization to all zeros

    const int N = capsule->nlp_solver_plan->N;
    ocp_nlp_config* nlp_config = capsule->nlp_config;
    ocp_nlp_dims* nlp_dims = capsule->nlp_dims;
    ocp_nlp_out* nlp_out = capsule->nlp_out;
    ocp_nlp_in* nlp_in = capsule->nlp_in;
    ocp_nlp_solver* nlp_solver = capsule->nlp_solver;

    double* buffer = calloc(NX+NU+NZ+2*NS+2*NSN+2*NS0+NBX+NBU+NG+NH+NPHI+NBX0+NBXN+NHN+NH0+NPHIN+NGN, sizeof(double));

    for(int i=0; i<N+1; i++)
    {
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "x", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "u", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "sl", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "su", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "lam", buffer);
        ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "z", buffer);
        if (i<N)
        {
            ocp_nlp_out_set(nlp_config, nlp_dims, nlp_out, i, "pi", buffer);
            ocp_nlp_set(nlp_solver, i, "xdot_guess", buffer);
            ocp_nlp_set(nlp_solver, i, "z_guess", buffer);
        }
    }
    // get qp_status: if NaN -> reset memory
    int qp_status;
    ocp_nlp_get(capsule->nlp_solver, "qp_status", &qp_status);
    if (reset_qp_solver_mem || (qp_status == 3))
    {
        // printf("\nin reset qp_status %d -> resetting QP memory\n", qp_status);
        ocp_nlp_solver_reset_qp_memory(nlp_solver, nlp_in, nlp_out);
    }

    free(buffer);
    return 0;
}




int SAM_equation_system_acados_update_params(SAM_equation_system_solver_capsule* capsule, int stage, double *p, int np)
{
    int solver_status = 0;

    int casadi_np = 25;
    if (casadi_np != np) {
        printf("acados_update_params: trying to set %i parameters for external functions."
            " External function has %i parameters. Exiting.\n", np, casadi_np);
        exit(1);
    }
    ocp_nlp_in_set(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, "parameter_values", p);

    return solver_status;
}


int SAM_equation_system_acados_update_params_sparse(SAM_equation_system_solver_capsule * capsule, int stage, int *idx, double *p, int n_update)
{
    ocp_nlp_in_set_params_sparse(capsule->nlp_config, capsule->nlp_dims, capsule->nlp_in, stage, idx, p, n_update);

    return 0;
}


int SAM_equation_system_acados_set_p_global_and_precompute_dependencies(SAM_equation_system_solver_capsule* capsule, double* data, int data_len)
{

    // printf("No global_data, SAM_equation_system_acados_set_p_global_and_precompute_dependencies does nothing.\n");
    return 0;
}




int SAM_equation_system_acados_solve(SAM_equation_system_solver_capsule* capsule)
{
    // solve NLP
    int solver_status = ocp_nlp_solve(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



int SAM_equation_system_acados_setup_qp_matrices_and_factorize(SAM_equation_system_solver_capsule* capsule)
{
    int solver_status = ocp_nlp_setup_qp_matrices_and_factorize(capsule->nlp_solver, capsule->nlp_in, capsule->nlp_out);

    return solver_status;
}



void SAM_equation_system_acados_batch_solve(SAM_equation_system_solver_capsule ** capsules, int * status_out, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_solve(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}


void SAM_equation_system_acados_batch_setup_qp_matrices_and_factorize(SAM_equation_system_solver_capsule ** capsules, int * status_out, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        status_out[i] = ocp_nlp_setup_qp_matrices_and_factorize(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}


void SAM_equation_system_acados_batch_eval_params_jac(SAM_equation_system_solver_capsule ** capsules, int N_batch)
{

    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_params_jac(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out);
    }


    return;
}



void SAM_equation_system_acados_batch_eval_solution_sens_adj_p(SAM_equation_system_solver_capsule ** capsules, const char *field, int stage, double *out, int offset, int N_batch)
{


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_eval_solution_sens_adj_p(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->sens_out, field, stage, out + i*offset);
    }


    return;
}


void SAM_equation_system_acados_batch_set_flat(SAM_equation_system_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_set_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_set_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}



void SAM_equation_system_acados_batch_get_flat(SAM_equation_system_solver_capsule ** capsules, const char *field, double *data, int N_data, int N_batch)
{
    int offset = ocp_nlp_dims_get_total_from_attr(capsules[0]->nlp_solver->config, capsules[0]->nlp_solver->dims, capsules[0]->nlp_out, field);

    if (N_batch*offset != N_data)
    {
        printf("batch_get_flat: wrong input dimension, expected %d, got %d\n", N_batch*offset, N_data);
        exit(1);
    }


    for (int i = 0; i < N_batch; i++)
    {
        ocp_nlp_get_all(capsules[i]->nlp_solver, capsules[i]->nlp_in, capsules[i]->nlp_out, field, data + i * offset);
    }


    return;
}


int SAM_equation_system_acados_free(SAM_equation_system_solver_capsule* capsule)
{
    // before destroying, keep some info
    const int N = capsule->nlp_solver_plan->N;
    // free memory
    ocp_nlp_solver_opts_destroy(capsule->nlp_opts);
    ocp_nlp_in_destroy(capsule->nlp_in);
    ocp_nlp_out_destroy(capsule->nlp_out);
    ocp_nlp_out_destroy(capsule->sens_out);
    ocp_nlp_solver_destroy(capsule->nlp_solver);
    ocp_nlp_dims_destroy(capsule->nlp_dims);
    ocp_nlp_config_destroy(capsule->nlp_config);
    ocp_nlp_plan_destroy(capsule->nlp_solver_plan);

    /* free external function */
    // dynamics
    for (int i = 0; i < N; i++)
    {
        external_function_external_param_casadi_free(&capsule->impl_dae_fun[i]);
        external_function_external_param_casadi_free(&capsule->impl_dae_fun_jac_x_xdot_z[i]);
        external_function_external_param_casadi_free(&capsule->impl_dae_jac_x_xdot_u_z[i]);
    }
    free(capsule->impl_dae_fun);
    free(capsule->impl_dae_fun_jac_x_xdot_z);
    free(capsule->impl_dae_jac_x_xdot_u_z);

    // cost
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_0_fun_jac_ut_xt);
    for (int i = 0; i < N - 1; i++)
    {
        external_function_external_param_casadi_free(&capsule->cost_y_fun[i]);
        external_function_external_param_casadi_free(&capsule->cost_y_fun_jac_ut_xt[i]);
    }
    free(capsule->cost_y_fun);
    free(capsule->cost_y_fun_jac_ut_xt);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun);
    external_function_external_param_casadi_free(&capsule->cost_y_e_fun_jac_ut_xt);

    // constraints
    for (int i = 0; i < N-1; i++)
    {
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun_jac[i]);
        external_function_external_param_casadi_free(&capsule->nl_constr_h_fun[i]);
    }
    free(capsule->nl_constr_h_fun_jac);
    free(capsule->nl_constr_h_fun);



    return 0;
}


void SAM_equation_system_acados_print_stats(SAM_equation_system_solver_capsule* capsule)
{
    int nlp_iter, stat_m, stat_n, tmp_int;
    ocp_nlp_get(capsule->nlp_solver, "nlp_iter", &nlp_iter);
    ocp_nlp_get(capsule->nlp_solver, "stat_n", &stat_n);
    ocp_nlp_get(capsule->nlp_solver, "stat_m", &stat_m);


    double stat[960];
    ocp_nlp_get(capsule->nlp_solver, "statistics", stat);

    int nrow = nlp_iter+1 < stat_m ? nlp_iter+1 : stat_m;


    printf("iter\tqp_stat\tqp_iter\n");
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < stat_n + 1; j++)
        {
            tmp_int = (int) stat[i + j * nrow];
            printf("%d\t", tmp_int);
        }
        printf("\n");
    }
}

int SAM_equation_system_acados_custom_update(SAM_equation_system_solver_capsule* capsule, double* data, int data_len)
{
    (void)capsule;
    (void)data;
    (void)data_len;
    printf("\ndummy function that can be called in between solver calls to update parameters or numerical data efficiently in C.\n");
    printf("nothing set yet..\n");
    return 1;

}



ocp_nlp_in *SAM_equation_system_acados_get_nlp_in(SAM_equation_system_solver_capsule* capsule) { return capsule->nlp_in; }
ocp_nlp_out *SAM_equation_system_acados_get_nlp_out(SAM_equation_system_solver_capsule* capsule) { return capsule->nlp_out; }
ocp_nlp_out *SAM_equation_system_acados_get_sens_out(SAM_equation_system_solver_capsule* capsule) { return capsule->sens_out; }
ocp_nlp_solver *SAM_equation_system_acados_get_nlp_solver(SAM_equation_system_solver_capsule* capsule) { return capsule->nlp_solver; }
ocp_nlp_config *SAM_equation_system_acados_get_nlp_config(SAM_equation_system_solver_capsule* capsule) { return capsule->nlp_config; }
void *SAM_equation_system_acados_get_nlp_opts(SAM_equation_system_solver_capsule* capsule) { return capsule->nlp_opts; }
ocp_nlp_dims *SAM_equation_system_acados_get_nlp_dims(SAM_equation_system_solver_capsule* capsule) { return capsule->nlp_dims; }
ocp_nlp_plan_t *SAM_equation_system_acados_get_nlp_plan(SAM_equation_system_solver_capsule* capsule) { return capsule->nlp_solver_plan; }
