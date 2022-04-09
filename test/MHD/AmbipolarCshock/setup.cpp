#include "idefix.hpp"
#include "setup.hpp"

/*********************************************/
/**
Customized random number generator
Allow one to have consistant random numbers
generators on different architectures.
**/
/*********************************************/

static real cs;

// Default constructor


void AmbipolarFunction(DataBlock &data, real t, IdefixArray3D<real> &xAin ) {
    IdefixArray3D<real> xA = xAin;
    IdefixArray4D<real> Vc = data.hydro.Vc;
    idefix_for("AmbipolarFunction",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,data.np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
        xA(k,j,i) = 2.0/Vc(RHO,k,j,i);
    });

}
// User-defined boundaries
void UserdefBoundary(DataBlock& data, int dir, BoundarySide side, real t) {
    if( (dir==IDIR) && (side == left)) {
        IdefixArray4D<real> Vc = data.hydro.Vc;
        IdefixArray4D<real> Vs = data.hydro.Vs;

        int ighost = data.nghost[IDIR];
        idefix_for("UserDefBoundary",0,data.np_tot[KDIR],0,data.np_tot[JDIR],0,ighost,
                    KOKKOS_LAMBDA (int k, int j, int i) {
                        Vc(RHO,k,j,i) = Vc(RHO,k,j,ighost);
                        #if HAVE_ENERGY
                        Vc(PRS,k,j,i) = Vc(PRS,k,j,ighost);
                        #endif
                        Vc(VX1,k,j,i) = Vc(VX1,k,j,ighost);
                        Vc(VX2,k,j,i) = Vc(VX2,k,j,ighost);
                        Vc(BX2,k,j,i) = Vc(BX2,k,j,ighost);
                        #if COMPONENTS == 3
                        Vc(VX3,k,j,i) = Vc(VX3,k,j,ighost);
                        Vc(BX3,k,j,i) = Vc(BX3,k,j,ighost);
                        #endif
                        D_EXPAND(                          ,
                                    Vs(BX2s,k,j,i) = Vs(BX2s,k,j,ighost);  ,
                                    Vs(BX3s,k,j,i) = Vs(BX3s,k,j,ighost); )
        });
    }
    if( (dir==IDIR) && (side == right)) {
            IdefixArray4D<real> Vc = data.hydro.Vc;
            IdefixArray4D<real> Vs = data.hydro.Vs;

            int ighost = data.end[IDIR]-1;
        idefix_for("UserDefBoundary",0,data.np_tot[KDIR],0,data.np_tot[JDIR],data.end[IDIR],data.np_tot[IDIR],
                        KOKKOS_LAMBDA (int k, int j, int i) {
                                Vc(RHO,k,j,i) = Vc(RHO,k,j,2*ighost-i+1);
                                #if HAVE_ENERGY
                                Vc(PRS,k,j,i) = Vc(PRS,k,j,2*ighost-i+1);
                                #endif
                                Vc(VX1,k,j,i) = -Vc(VX1,k,j,2*ighost-i+1);
                                Vc(VX2,k,j,i) = Vc(VX2,k,j,2*ighost-i+1);
                                Vc(BX2,k,j,i) = Vc(BX2,k,j,2*ighost-i+1);
                                #if COMPONENTS == 3
                                Vc(VX3,k,j,i) = Vc(VX3,k,j,2*ighost-i+1);
                                Vc(BX3,k,j,i) = Vc(BX3,k,j,2*ighost-i+1);
                                #endif
                                D_EXPAND(                          ,
                                         Vs(BX2s,k,j,i) = Vs(BX2s,k,j,2*ighost-i+1);  ,
                                         Vs(BX3s,k,j,i) = Vs(BX3s,k,j,2*ighost-i+1); )

            });
        }



}


// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
    data.hydro.EnrollUserDefBoundary(&UserdefBoundary);
    data.hydro.EnrollAmbipolarDiffusivity(&AmbipolarFunction);
    cs=input.Get<real>("Hydro","csiso",1);
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
    // Create a host copy
    DataBlockHost d(data);
    real x,y,z;

    real M=50;
    real theta=M_PI/4.0;
    real A=10.0;

    // D obtained analytically
    real D=17.7532;

    real vs=M*cs;

    real vin=vs*(1-1/D);

    real B0 = vs/A;
    for(int k = 0; k < d.np_tot[KDIR] ; k++) {
        for(int j = 0; j < d.np_tot[JDIR] ; j++) {
            for(int i = 0; i < d.np_tot[IDIR] ; i++) {
                x=d.x[IDIR](i);
                y=d.x[JDIR](j);
                z=d.x[KDIR](k);

                d.Vc(RHO,k,j,i) = 1.0;
#if HAVE_ENERGY
                d.Vc(PRS,k,j,i) = 1.0;
#endif
                d.Vc(VX1,k,j,i) = vin;
                d.Vc(VX2,k,j,i) = 0.0;
                d.Vs(BX1s,k,j,i) = B0*cos(theta);
#if DIMENSIONS >= 2
                d.Vs(BX2s,k,j,i) = B0*sin(theta);
#else
                d.Vc(BX2,k,j,i) = B0*sin(theta);
#endif

            }
        }
    }

    // Send it all, if needed
    d.SyncToDevice();
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock & data) {

}
