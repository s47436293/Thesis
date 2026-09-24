
}



CudaDeviceFunction void Equilibrium(real_t rho_, real_t theta_, real_t u_, real_t fe_[3]) {
    const real_t cs2 = 1.0/3.0;
    real_t A = pow(u_,2)/(2.0*cs2*cs2) - pow(u_,2)/(2.0*cs2) + (theta_ - 1.0)*(1.0 - cs2)/(2.0*cs2);
    fe_[0] = (2.0/3.0) * rho_ * (1.0 - pow(u_,2) * 0.5 / cs2 - 0.5 * (theta_ - 1.0));
    fe_[1] = (1.0/6.0) * rho_ * (1.0 + u_/cs2 + A);
    fe_[2] = (1.0/6.0) * rho_ * (1.0 - u_/cs2 + A);
}

CudaDeviceFunction real_t CalcPhi(real_t dm, real_t dp) {
    const real_t eps = 1.0e-14;
    real_t den = fabs(dp) < eps ? eps : dp;
    real_t r = dm/den;
    real_t phi = 2.0*r/(1.0 + r*r + eps);
    return phi > 0.0 ? phi : 0.0;
}
CudaDeviceFunction real_t getS() { return entr(0,0); }

CudaDeviceFunction real_t AdvectEntropy() {
    const real_t chi = 1.0/3.0;
    
    real_t sp1 = entr(1,0),  sp2 = entr(2,0);
    real_t sm1 = entr(-1,0), sm2 = entr(-2,0);
    real_t sp0 = entr(0,0);
    
    real_t deltas_ph  = sp1 - sp0;
    real_t deltas_mh  = sp0 - sm1;
    real_t deltas_3ph = sp2 - sp1;
    real_t deltas_3mh = sm1 - sm2;

    real_t phi_ri   = CalcPhi(deltas_mh,  deltas_ph);
    real_t phi_rip1 = CalcPhi(deltas_ph,  deltas_3ph);
    real_t phi_rim1 = CalcPhi(deltas_3mh, deltas_mh);

    real_t SL_ph = sp0 + phi_ri/4   * ((1.0-chi)*deltas_mh  + (1.0+chi)*deltas_ph);
    real_t SL_mh = sm1 + phi_rim1/4 * ((1.0-chi)*deltas_3mh + (1.0+chi)*deltas_mh);

    real_t SR_ph = sp1 - phi_rip1/4 * ((1.0-chi)*deltas_3ph + (1.0+chi)*deltas_ph);
    real_t SR_mh = sp0 - phi_ri/4   * ((1.0-chi)*deltas_ph  + (1.0+chi)*deltas_mh);

    real_t S_ph = (u(0,0)  >= 0.0) ? SL_ph : SR_ph;
    real_t S_mh = (u(-1,0) >= 0.0) ? SL_mh : SR_mh;

    return u(0,0)*(S_ph - S_mh);
}

CudaDeviceFunction real_t EntropyRHS() {
    real_t p = rho(0,0)*(1.0/3.0)*theta(0,0);
    real_t tau = mu_latt/p;
    real_t tau_t = tau + 0.5;

    real_t conv = AdvectEntropy();
    real_t lam = mu_latt*CP()/Pr;
    real_t diff_fourier = (-lam)*(theta(1,0) - 2.0*theta(0,0) + theta(-1,0));
    real_t dudx = 0.5*(u(1,0) - u(-1,0));
    real_t phi = -(tau/tau_t)*a1(0,0)*dudx;

    return -conv + (diff_fourier + phi)/(rho(0,0)*theta(0,0));
}

CudaDeviceFunction void Init() {
    real_t xr = (real_t)X;
    real_t ramp = 0.5*(1.0 - tanh((xr - x_split)/delta_x));

    real_t rho_init   = rho_R + (rho_L - rho_R)*ramp;
    real_t theta_init = theta_inf;

    rho   = rho_init;
    u     = 0.0;
    theta = theta_init;
    entr  = SFromState(rho_init, theta_init);
    a1    = 0.0;

    real_t fe[3];
    Equilibrium(rho_init, theta_init, 0.0, fe);
    f[0] = fe[0];
    f[1] = fe[1];
    f[2] = fe[2];
}

CudaDeviceFunction void CalcNonLocal(){
    
}

CudaDeviceFunction void Run() {
    const real_t cs2 = 1.0/3.0;
    const real_t kappa = (D_dim + 2.0)/D_dim;

    real_t d = f[0] + f[1] + f[2];
    real_t v = (f[1] - f[2])/d;
    real_t th = theta(0,0);
    real_t p_loc = d*cs2*th;
    real_t tau = mu_latt/p_loc;
    real_t tau_t = tau + 0.5;
    real_t omega = 1.0/tau_t;
    real_t dudx = 0.5*(u(1,0) - u(-1,0));

    real_t Gm = rho(-1,0)*u(-1,0)*(1.0 - theta(-1,0) - u(-1,0)*u(-1,0));
    real_t G0 = rho(0,0)*u(0,0)*(1.0 - theta(0,0) - u(0,0)*u(0,0));
    real_t Gp = rho(1,0)*u(1,0)*(1.0 - theta(1,0) - u(1,0)*u(1,0));

    real_t dGb = G0 - Gm;
    real_t dGf = Gp - G0;

    real_t sgn = (real_t)((u(0,0) > 0.0) - (u(0,0) < 0.0));
    real_t E1 = 0.5*(1.0 + sgn)*dGb + 0.5*(1.0 - sgn)*dGf;
    real_t E2 = p_loc * (kappa - gamma_g) * dudx;
    real_t E = E1 + E2;
    real_t psi[3] = {-E, 0.5*E, 0.5*E};

    real_t fe[3];
    Equilibrium(d, th, v, fe);

    const real_t H2[3] = {-1.0/3.0, 2.0/3.0, 2.0/3.0};
    real_t a1_pr = 0.0;
    for (int k = 0; k < 3; k++)
        a1_pr += H2[k]*(f[k] - fe[k] + 0.5*psi[k]);

    real_t a1_fd = -tau_t * p_loc * (2.0 - 2.0/D_dim) * dudx;
    real_t a1_new = sigma*a1_pr + (1.0 - sigma)*a1_fd;
    a1 = a1_new;

    const real_t proj[3] = {-1.0, 0.5, 0.5};
    for (int k = 0; k < 3; k++)
        f[k] = fe[k] + (1.0 - omega)*a1_new*proj[k] + 0.5*psi[k];

    rho = d;
    u = v;
    real_t s_new = entr(0,0) + EntropyRHS();
    entr = s_new;
    theta = ThetaFromS(d, s_new);
}



CudaDeviceFunction real_t getRho()   { return rho(0,0); }
CudaDeviceFunction real_t getTheta() { return theta(0,0); }

CudaDeviceFunction real_t getA1()    { return a1(0,0); }

CudaDeviceFunction vector_t getU() {
    vector_t v;
    v.x = u(0,0);
    v.y = 0.0;
    v.z = 0.0;
    return v;
}

CudaDeviceFunction float2 Color() {
    float2 ret; ret.x = 0; ret.y = 1; return ret;
}