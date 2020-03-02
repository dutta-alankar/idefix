#include <cstdio>
#include "idefix.hpp"
#include "timeIntegrator.hpp"

TimeIntegrator::TimeIntegrator(Input & input, DataBlock &datain, Physics &physics) {
    this->data=datain;
    this->phys=physics;

    nstages=input.nstages;
    
    dt=input.firstDt;
    ncycles=0;
    cfl=0.9;

    std::cout << "init dt:" << dt << std::endl;

    if(nstages>1) {
        // Temporary array to store initial field in the RK2-3 loops
        V0 = IdefixArray4D<real>("TimeIntegrator_V0", NVAR, data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    }
    if(nstages==2) {
        wc[0] = 0.5;
        w0[0] = 0.5;
    }
    if(nstages==3) {
        wc[0] = 0.25;
        w0[0] = 0.5;
        wc[1] = 2.0/3.0;
        w0[1] = 1.0/3.0;
    }
    InvDtHyp = IdefixArray3D<real>("TimeIntegrator_InvDtHyp", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);
    InvDtPar = IdefixArray3D<real>("TimeIntegrator_InvDtPar", data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]);

    // Dummy dt initialisation

    

}

// Compute one Stage of the time Integrator
void TimeIntegrator::Stage() {
    
    Kokkos::Profiling::pushRegion("TimeIntegrator::Stage");
    // Convert current state into conservative variable and save it
    phys.ConvertPrimToCons(data);

    // Loop on all of the directions
    for(int dir = 0 ; dir < DIMENSIONS ; dir++) {
        // Step one: extrapolate the variables to the sides, result is stored in the physics object
        phys.ExtrapolatePrimVar(data, dir);

        // Step 2: compute the intercell flux with our Riemann solver, store the resulting InvDt
        phys.CalcRiemannFlux(data, InvDtHyp, dir);

        // Step 3: compute the resulting evolution of the conserved variables, stored in Uc
        phys.CalcRightHandSide(data, dir, dt);
    }

    // Convert back into primitive variables
    phys.ConvertConsToPrim(data);

    // Apply Boundary conditions
    phys.SetBoundary(data);

    Kokkos::Profiling::popRegion();
}


// Compute one full cycle of the time Integrator
void TimeIntegrator::Cycle() {
    // Do one cycle
    IdefixArray4D<real> Vc = data.Vc;
    IdefixArray4D<real> V0 = this->V0;
    IdefixArray3D<real> InvDtHypLoc=this->InvDtHyp;
    IdefixArray3D<real> InvDtParLoc=this->InvDtPar;

    Kokkos::Profiling::pushRegion("TimeIntegrator::Cycle");

    std::cout << "TimeIntegrator: t=" << t << " Cycle " << ncycles << " dt=" << dt << std::endl;

    // Store initial stage for multi-stage time integrators
    if(nstages>1) Kokkos::deep_copy(V0,Vc);

    for(int stage=0; stage < nstages ; stage++) {
        // Update Vc
        Stage();
        // Is this the last stage?
        if(stage<nstages-1) {
            // No!
            real wcs=wc[stage];
            real w0s=w0[stage];

            idefix_for("Cycle-update",0,NVAR,0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
                        KOKKOS_LAMBDA (int n, int k, int j, int i) {
                            Vc(n,k,j,i) = wcs*Vc(n,k,j,i) + w0s*V0(n,k,j,i);
            });
        }
    }
    
    // Update current time
    t=t+dt;

    // Compute next time_step
    real newdt;
    Kokkos::parallel_reduce("Timestep_reduction",
                            Kokkos::MDRangePolicy<Kokkos::Rank<3, Kokkos::Iterate::Right, Kokkos::Iterate::Right>>
                            ({0,0,0},{data.np_tot[KDIR], data.np_tot[JDIR], data.np_tot[IDIR]}),
                            KOKKOS_LAMBDA (int k, int j, int i, real &dtmin) {
        real InvDt;
        InvDt = SQRT(InvDtHypLoc(k,j,i) * InvDtHypLoc(k,j,i) + InvDtParLoc(k,j,i) * InvDtParLoc(k,j,i));

		dtmin=FMIN(ONE_F/InvDt,dtmin);
    }, Kokkos::Min<real>(newdt) );

    newdt=newdt*cfl;

    if(newdt>1.1*dt) {
        dt=1.1*dt;
    }
    else dt=newdt;

    ncycles++;

    Kokkos::Profiling::popRegion();
}

real TimeIntegrator::getDt() {
    return(dt);
}

real TimeIntegrator::getT() {
    return (t);
}

void TimeIntegrator::setDt(real dtin) {
    dt=dtin;
} 

long int TimeIntegrator::getNcycles() {
    return(ncycles);
}
