#include <random>
#include <chrono>
#include "idefix.hpp"
#include "setup.hpp"
#include "units.hpp"
#include "lookupTable.hpp"
#include "fluid.hpp"
#include "fluid_defs.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "gridHost.hpp"

// -----------------------------------------------------------------------------
// All setup parameters live in this single object. They are set once in the
// Setup constructor and never change afterwards, so device kernels just copy the
// fields they need into local variables and capture those by value (exactly like
// idfx::units / physical constants). This replaces the old pile of loose globals
// plus the user_params[] host array and its user_params_dev device mirror.
// -----------------------------------------------------------------------------
namespace {
struct TRMLParams {
  // read from the input file
  real z1, z2, sig, tanh_a, k_ini, amp, mach_r;
  real Tcl, chi, kappa_0, eta1_0, eta2_0, TcoolFloor;
  real gamma;                                   // adiabatic index
  real xbeg, xend, ybeg, yend, zbeg, zend;      // global grid extents
  real tstop;
  // derived once in the constructor
  real del_rho_by_rhow, P0, uflow;
  real tcool_ib, temperature_mix;
  // cooling-table interpolation data
  std::unique_ptr<LookupTable<1>> cooltable;
};
TRMLParams prm;
}  // namespace

// Power-law interpolation of the cooling function Lambda(T) from a lookup table:
// binary-search the (monotonic) temperature grid for the samples bracketing
// `temperature`, then interpolate as Lambda = Lambda_lo * (T/T_lo)^alpha.
static real interpolateLambda(real temperature,
                              const IdefixHostArray1D<real> &temperature_data,
                              const IdefixHostArray1D<real> &Lambda_cool_data) {
  int T_indx_lo = 0, T_indx_hi=temperature_data.extent(0)-1;
  int T_indx_mid;
  // binary search on the interpolation table
  while (T_indx_lo<=T_indx_hi) {
    T_indx_mid = (T_indx_lo + T_indx_hi)/2;
    if (temperature < temperature_data(T_indx_mid)) {
      T_indx_hi = T_indx_mid-1;
    } else if (temperature > temperature_data(T_indx_mid)) {
      T_indx_lo = T_indx_mid+1;
    } else {
      T_indx_lo = T_indx_mid;
      T_indx_hi = T_indx_mid;
      break;
    }
  }
  if (T_indx_lo!=T_indx_hi) {
    // swap
    T_indx_mid = T_indx_lo;
    T_indx_lo = T_indx_hi;
    T_indx_hi = T_indx_mid;
  }
  real temperature_lo = temperature_data(T_indx_lo);
  real temperature_hi = temperature_data(T_indx_hi);
  real Lambda_lo = Lambda_cool_data(T_indx_lo);
  real Lambda_hi = Lambda_cool_data(T_indx_hi);
  // T_ref = temperature_hi
  real alpha = std::log(Lambda_hi/Lambda_lo)/std::log(temperature_hi/temperature_lo);
  return Lambda_lo * pow((temperature/temperature_lo), alpha);
}

// user-defined function declarations
void MakeAnalysis(DataBlock &);
void ComputeUserVars(DataBlock &, UserDefVariablesContainer &);
void conductionKappa(DataBlock &, const real , IdefixArray3D<real> &);
void viscosityEta(DataBlock &, const real , IdefixArray3D<real> &, IdefixArray3D<real> &);
void UserdefBoundary(Hydro *, int , BoundarySide , real );

// Initialisation routine. Can be used to allocate
// Arrays or variables which are used later on
Setup::Setup(Input &input, Grid &grid, DataBlock &data, Output &output) {
  prm.gamma = input.GetOrSet<real>("Hydro","gamma",0, 5.0/3.0);
  output.EnrollAnalysis(&MakeAnalysis);
  output.EnrollUserDefVariables(&ComputeUserVars);
  if (input.CheckEntry("Hydro", "TDiffusion")>=0) {
    data.hydro->thermalDiffusion->EnrollThermalDiffusivity(&conductionKappa);
  }
  if (input.CheckEntry("Hydro", "viscosity")>=0) {
    data.hydro->viscosity->EnrollViscousDiffusivity(&viscosityEta);
  }
  data.hydro->boundary->EnrollUserDefBoundary(&UserdefBoundary);

  prm.z1      = input.Get<real>("Setup", "z1", 0);
  prm.z2      = input.Get<real>("Setup", "z2", 0);
  prm.sig     = input.Get<real>("Setup", "sig", 0);
  prm.tanh_a  = input.Get<real>("Setup", "tanh_a", 0);
  prm.k_ini   = input.Get<real>("Setup", "k_ini", 0);
  prm.amp     = input.Get<real>("Setup", "amp", 0);
  prm.mach_r  = input.Get<real>("Setup", "mach_r", 0);
  prm.Tcl     = input.Get<real>("Setup", "Tcl", 0);
  prm.chi     = input.Get<real>("Setup", "chi", 0);
  prm.kappa_0 = input.Get<real>("Setup", "kappa_0", 0);
  prm.eta1_0  = input.Get<real>("Setup", "eta1_0", 0);
  prm.eta2_0  = input.Get<real>("Setup", "eta2_0", 0);
  // [Hydro] Cooling line: [0]=Tabulated [1]=table file [2]=Townsend [3]="TcoolFloor" [4]=value.
  // The whole line may be absent (no cooling): fall back to sensible defaults so
  // the setup still works. CheckEntry() returns the token count, or -1 if absent.
  const int nCooling = input.CheckEntry("Hydro","Cooling");
  prm.TcoolFloor = (nCooling > 4) ? input.Get<real>("Hydro","Cooling",4) : 1.0e+04;

  prm.xbeg = input.Get<real>("Grid", "X1-grid", 1);
  prm.xend = input.Get<real>("Grid", "X1-grid", 4);
  prm.ybeg = input.Get<real>("Grid", "X2-grid", 1);
  prm.yend = input.Get<real>("Grid", "X2-grid", 4);
  prm.zbeg = input.Get<real>("Grid", "X3-grid", 1);
  prm.zend = input.Get<real>("Grid", "X3-grid", 4);
  prm.tstop = input.Get<real>("TimeIntegrator", "tstop", 0);

  const std::string cooltable_file = (nCooling > 1) ? input.Get<std::string>("Hydro","Cooling",1)
                                                    : "cooltable.dat";  // [1] = cooling table file
  prm.cooltable = std::make_unique<LookupTable<1>>(cooltable_file, ' ');

  const real kB = idfx::units.k_B;
  const real mu = 0.609;
  const real m_p = idfx::units.m_p;
  const real XH = 0.71;

  // Code-to-physical (CGS) unit conversion factors
  const real vel_unit = idfx::units.GetVelocity();
  const real rho_unit = idfx::units.GetDensity();
  const real len_unit = idfx::units.GetLength();

  // assuming rho_unit is hot gas density
  prm.del_rho_by_rhow = prm.chi-ONE_F;
  prm.P0 = prm.chi*kB*prm.Tcl/(mu*m_p)/pow(vel_unit, 2);
  prm.uflow = sqrt(prm.gamma*kB*prm.chi*prm.Tcl/(mu*m_p))*HALF_F*prm.mach_r/vel_unit;

  // data on host
  IdefixHostArray1D<real> temperature_data_host = prm.cooltable->xinHost;
  IdefixHostArray1D<real> Lambda_cool_data_host = prm.cooltable->dataHost;
  // fastest isobarically cooling gas is approximately at 1.8e+04 K (plot T^2/Lambda(T) and see minima)
  prm.temperature_mix = sqrt(prm.chi)*prm.Tcl;
  real temperature = prm.temperature_mix;
  real Lambda_mix = interpolateLambda(temperature, temperature_data_host, Lambda_cool_data_host);
  prm.tcool_ib = (prm.gamma/(prm.gamma-ONE_F))*pow(kB/(mu*XH),2)*pow(temperature,2)/(prm.P0*rho_unit*pow(vel_unit,2)*Lambda_mix); // cgs
  prm.tcool_ib /= (len_unit/vel_unit); // code
}

// This routine initialize the flow
// Note that data is on the device.
// One can therefore define locally
// a datahost and sync it, if needed
void Setup::InitFlow(DataBlock &data) {
  // Create a host copy
  DataBlockHost d(data);
  [[maybe_unused]] int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = d.beg[IDIR];
  iend = d.end[IDIR];
  jbeg = d.beg[JDIR];
  jend = d.end[JDIR];
  kbeg = d.beg[KDIR];
  kend = d.end[KDIR];

  const real Lx1 = prm.xend - prm.xbeg;
  const real Lx2 = prm.yend - prm.ybeg;
  [[maybe_unused]] const real Lx3 = prm.zend - prm.zbeg;

  // setup parameters, read once from `prm` (the host loop below uses them by name)
  const real z1 = prm.z1, z2 = prm.z2, tanh_a = prm.tanh_a, sig = prm.sig;
  const real k_ini = prm.k_ini, amp = prm.amp, uflow = prm.uflow;
  const real del_rho_by_rhow = prm.del_rho_by_rhow;
  [[maybe_unused]] const real P0 = prm.P0;

  std::default_random_engine generator;
  // seed
  generator.seed(54321);
  /*
  // time based seed for non-reproducible sequence
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  generator.seed(seed);
  // random device based seed
  std::random_device rd;
  generator.seed(rd());
  */
  std::uniform_real_distribution<real> distribution(-ONE_F, ONE_F);
  real random_level = 1.0e-08; // adjust the level of randomness in the initial conditions

  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        // Lecoanet
        d.Vc(RHO,k,j,i)     =  ONE_F + del_rho_by_rhow * 
                               HALF_F * ( tanh((d.x[KDIR](k)-z1)/tanh_a) - tanh((d.x[KDIR](k)-z2)/tanh_a) );

        d.Vc(RHO,k,j,i)    *= (ONE_F+random_level*distribution(generator));

        d.Vc(VX1,k,j,i) = uflow * ( tanh((d.x[KDIR](k)-z1)/tanh_a) - tanh((d.x[KDIR](k)-z2)/tanh_a) - ONE_F );
        d.Vc(VX3,k,j,i) = amp * uflow * sin(2*M_PI*k_ini*d.x[IDIR](i)/Lx1) * sin(2*M_PI*k_ini*d.x[JDIR](j)/Lx2) *
                                            ( exp(-pow((d.x[KDIR](k)-z1)/sig,2)) + exp(-pow((d.x[KDIR](k)-z2)/sig,2)) );
        d.Vc(VX2,k,j,i) = ZERO_F;

        d.Vc(VX1,k,j,i)    *= (ONE_F+random_level*distribution(generator));
        d.Vc(VX3,k,j,i)    *= (ONE_F+random_level*distribution(generator));
        d.Vc(VX2,k,j,i)    += (random_level*distribution(generator));

        d.Vc(TRG,k,j,i)  = HALF_F * ( tanh((d.x[KDIR](k)-z2)/tanh_a) - tanh((d.x[KDIR](k)-z1)/tanh_a) + 2*ONE_F ); 
        d.Vc(TRG,k,j,i) *= (ONE_F+random_level*distribution(generator));
        #if HAVE_ENERGY
        d.Vc(PRS,k,j,i) = P0*(ONE_F+random_level*distribution(generator));
        #endif
      }
    }
  }
  // Send it all, if needed
  d.SyncToDevice();
}

// Release setup-owned globals that hold Kokkos Views (the cooling table).
// `prm` has static storage duration, so it is destroyed only after main()
// returns -- i.e. after Kokkos::finalize(). Kokkos aborts if a View is freed
// after finalize ("... deallocated after Kokkos::finalize was called"), so
// main() must call this while Kokkos is still alive (before Kokkos::finalize()).
void CleanupGlobalInSetup() {
  prm.cooltable.reset();
}

void conductionKappa(DataBlock &data, const real t, IdefixArray3D<real> &kappaArr) {
  // Create a host copy
  // DataBlockHost d(data);

  [[maybe_unused]] int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data.beg[IDIR];
  iend = data.end[IDIR];
  jbeg = data.beg[JDIR];
  jend = data.end[JDIR];
  kbeg = data.beg[KDIR];
  kend = data.end[KDIR];
  IdefixArray4D<real> Vc = data.hydro->Vc;

  // idfx::units is host-only and the setup parameters live in `prm`: read them on
  // the host and capture by value into the device lambda below.
  const real kB = idfx::units.k_B;
  const real m_p = idfx::units.m_p;
  const real vel_unit = idfx::units.GetVelocity();
  const real rho_unit = idfx::units.GetDensity();
  const real len_unit = idfx::units.GetLength();
  const real mu = 0.609;
  [[maybe_unused]] const real XH = 0.71;
  const real kappa_const = prm.kappa_0;

  idefix_for("conductionKappaLoop",kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA (int k, int j, int i) {
      [[maybe_unused]] real T_k_ref = 8.0e+04; // K
      [[maybe_unused]] real temperature = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
      [[maybe_unused]] real alpha = -3.0;
      // see PLUTO user manual Eq. 8.11
      kappaArr(k,j,i) = kappa_const *(mu*m_p/kB)/(rho_unit*vel_unit*len_unit);
      kappaArr(k,j,i) *= pow(temperature/T_k_ref, alpha);
    });
}

void viscosityEta(DataBlock &data, const real t, IdefixArray3D<real> &eta1, IdefixArray3D<real> &eta2) {
  // Create a host copy
  // DataBlockHost d(data);

  [[maybe_unused]] int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = data.beg[IDIR];
  iend = data.end[IDIR];
  jbeg = data.beg[JDIR];
  jend = data.end[JDIR];
  kbeg = data.beg[KDIR];
  kend = data.end[KDIR];
  // idfx::units is host-only and the setup parameters live in `prm`: read them on
  // the host and capture by value into the device lambda below.
  const real vel_unit = idfx::units.GetVelocity();
  const real rho_unit = idfx::units.GetDensity();
  const real len_unit = idfx::units.GetLength();
  const real eta1_const = prm.eta1_0;
  const real eta2_const = prm.eta2_0;

  idefix_for("viscosityEtaLoop",kbeg, kend, jbeg, jend, ibeg, iend,
    KOKKOS_LAMBDA (int k, int j, int i) {
      eta1(k,j,i) = eta1_const /(rho_unit*vel_unit*len_unit); // shear
      eta2(k,j,i) = eta2_const /(rho_unit*vel_unit*len_unit); // bulk
    });
}

void UserdefBoundary(Hydro *hydro, int dir, BoundarySide side, real t) {
  IdefixArray4D<real> Vc = hydro->Vc;
  [[maybe_unused]] IdefixArray1D<real> x1 = hydro->data->x[IDIR];
  [[maybe_unused]] IdefixArray1D<real> x2 = hydro->data->x[JDIR];
  IdefixArray1D<real> x3 = hydro->data->x[KDIR];
  
  const int nxi = hydro->data->np_int[IDIR];
  const int nxj = hydro->data->np_int[JDIR];
  const int nxk = hydro->data->np_int[KDIR];

  const int ighost = hydro->data->nghost[IDIR];
  const int jghost = hydro->data->nghost[JDIR];
  const int kghost = hydro->data->nghost[KDIR];

  if (dir==KDIR) {
    // Zero-gradient (copy from boundary cell) for all variables.
    // RHO, PRS, VX1 will be overridden below with fixed values.
    // VX2, VX3, and tracers get zero-gradient by this loop.
    hydro->boundary->BoundaryForAll("BoundaryInflowOutflow", dir, side,
    KOKKOS_LAMBDA (int n, int k, int j, int i) {
      // side = 0 on the left and =1 on the right
      const int kref = kghost + side*(nxk-1);
      Vc(n,k,j,i) = Vc(n,kref,j,i);
    });
  }
  /* Fix RHO, PRS, VX1 at their boundary profile values */
  /*
  if (dir==KDIR) {
    hydro->boundary->BoundaryFor("BoundaryFixPhasesShear", dir, side,
    KOKKOS_LAMBDA (int k, int j, int i) {
      [[maybe_unused]] real kB = idfx::units.k_B;
      [[maybe_unused]] real mu = 0.609;
      [[maybe_unused]] real m_p = idfx::units.m_p;
      [[maybe_unused]] real XH = 0.71;

      // Code-to-physical (CGS) unit conversion factors
      [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
      [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
      [[maybe_unused]] real len_unit = idfx::units.GetLength();
      [[maybe_unused]] real chi = user_params_dev(8);
      [[maybe_unused]] real Tcl = user_params_dev(7);
      [[maybe_unused]] real mach_r = user_params_dev(6);
      [[maybe_unused]] real tanh_a = user_params_dev(3);
      [[maybe_unused]] real z1 = user_params_dev(0);
      [[maybe_unused]] real z2 = user_params_dev(1);
      [[maybe_unused]] real gamma_adaiabatic = user_params_dev(19);
      [[maybe_unused]] real uflow = sqrt(gamma_adaiabatic*kB*chi*Tcl/(mu*m_p))*HALF_F*mach_r/vel_unit;
      [[maybe_unused]] real del_rho_by_rhow = chi-ONE_F;
      // Use boundary cell coordinate so all ghost cells receive the same fixed value
      const int kref = kghost + side*(nxk-1);
      const real z = x3(kref);
      Vc(RHO,k,j,i) = ONE_F + del_rho_by_rhow *
                      HALF_F * ( tanh((z-z1)/tanh_a) - tanh((z-z2)/tanh_a) );
      Vc(PRS,k,j,i) = chi*kB*Tcl/(mu*m_p)/pow(vel_unit, 2);
      Vc(VX1,k,j,i) = uflow * ( tanh((z-z1)/tanh_a) - tanh((z-z2)/tanh_a) - ONE_F );
    });
  }*/
}

void ComputeUserVars(DataBlock & data, UserDefVariablesContainer &variables) {
  // Mirror data on Host
  DataBlockHost d(data);

  // Sync it
  d.SyncFromDevice();

  if (prm.cooltable==nullptr) IDEFIX_ERROR("ComputeUserVars: cooltable data not assigned!");

  // Make references to the user-defined arrays (variables is a container of IdefixHostArray3D)
  // Note that the labels should match the variable names in the input file
  IdefixHostArray3D<real> temp_K  = variables["temp_K"];
  IdefixHostArray3D<real> num_dens  = variables["num_dens"];
  IdefixHostArray3D<real> mach_x1 = variables["mach_x1"];
  IdefixHostArray3D<real> mach_x2 = variables["mach_x2"];
  IdefixHostArray3D<real> mach_x3 = variables["mach_x3"];
  IdefixHostArray3D<real> E_dot_cool = variables["E_dot_cool"];
  IdefixHostArray3D<real> cell_vol = variables["cell_vol"];
  IdefixHostArray3D<real> delta_eng_total_cool_output = variables["delta_eng_total_cool"];

  IdefixArray3D<real> delta_eng_total_cool_dev; // this is a device array and we will mirror it on host to update it
  IdefixHostArray3D<real> delta_eng_total_cool_host;
  if (data.hydro->coolingOn) {
    delta_eng_total_cool_dev = data.hydro->radCooling->delta_eng_total;
    delta_eng_total_cool_host = data.hydro->radCooling->delta_eng_total_host;
    Kokkos::deep_copy(delta_eng_total_cool_host, delta_eng_total_cool_dev);
  }

  // data on host
  IdefixHostArray1D<real> temperature_data = prm.cooltable->xinHost;
  IdefixHostArray1D<real> Lambda_cool_data = prm.cooltable->dataHost;

  const real kB = idfx::units.k_B;
  const real mu = 0.609;
  const real m_p = idfx::units.m_p;
  const real XH = 0.71;

  // Code-to-physical (CGS) unit conversion factors
  const real vel_unit = idfx::units.GetVelocity();
  const real rho_unit = idfx::units.GetDensity();
  const real len_unit = idfx::units.GetLength();

  // setup parameters needed below, pulled once from `prm`
  const real gamma_adaiabatic = prm.gamma;
  const real chi = prm.chi;
  const real Tcl = prm.Tcl;
  const real TcoolFloor = prm.TcoolFloor;

  const real cs_hot = sqrt(gamma_adaiabatic*kB*chi*Tcl/(mu*m_p))/vel_unit;

  for(int k = 0; k < d.np_tot[KDIR] ; k++) {
    for(int j = 0; j < d.np_tot[JDIR] ; j++) {
      for(int i = 0; i < d.np_tot[IDIR] ; i++) {
        cell_vol(k,j,i) = d.dV(k,j,i);
        temp_K(k,j,i)  = d.Vc(PRS,k,j,i)/d.Vc(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
        mach_x1(k,j,i) = d.Vc(VX1,k,j,i)/cs_hot;
        mach_x2(k,j,i) = d.Vc(VX2,k,j,i)/cs_hot;
        mach_x3(k,j,i) = d.Vc(VX3,k,j,i)/cs_hot;
        num_dens(k,j,i) = d.Vc(RHO,k,j,i)*rho_unit/(mu*m_p);

        real temperature = temp_K(k,j,i);
        if (data.hydro->coolingOn) {
          /*
          if ( (temperature<(1.2*Tcl)) || (temperature>(0.9*chi*Tcl)) )
            delta_eng_total_cool_output(k,j,i) = ZERO_F;
          else 
          */
          delta_eng_total_cool_output(k,j,i) = delta_eng_total_cool_host(k,j,i);
        } else {
          delta_eng_total_cool_output(k,j,i) = ZERO_F;
        }
        
        if (data.hydro->coolingOn) {
          real Lambda = ZERO_F;
          if ((temperature<=TcoolFloor) || (d.Vc(TRG,k,j,i)>0.99)) {
            Lambda = ZERO_F;
          // } 
          // if ( (temperature<(1.2*Tcl)) || (temperature>(0.9*chi*Tcl)) ) {
          //   Lambda = ZERO_F;
          } else {
            Lambda = interpolateLambda(temperature, temperature_data, Lambda_cool_data);
          }
          E_dot_cool(k,j,i) = (pow(d.Vc(RHO,k,j,i)*rho_unit*XH/m_p, 2)*Lambda*d.dV(k,j,i)*pow(len_unit, 3)); // cgs
        } else {
          E_dot_cool(k,j,i) = ZERO_F; 
        }
      }
    }
  }
  // if ((d.t+d.dt)>tstop) 
}

// Analyse data to produce an output
void MakeAnalysis(DataBlock & data) {
  /*DISCLAIMER:: profiles only work when MPI decomposition is only along z-axis, i.e. prp to shear layer*/
  // data is on device and d is on Host
  // Mirror data on Host
  // DataBlockHost d(data);

  // Sync it
  // d.SyncFromDevice();
  /*
  [[maybe_unused]] real Lx1 = user_params[14]-user_params[13];
  [[maybe_unused]] real Lx2 = user_params[16]-user_params[15];
  [[maybe_unused]] real Lx3 = user_params[18]-user_params[17];

  // idfx::cout << "DEBUG: Analysis(): " << std::endl;
  if (cooltable_data==nullptr) IDEFIX_ERROR("MakeAnalysis: cooltable data not assigned!");

  // data on device
  IdefixArray1D<real> temperature_data = cooltable_data->xinDev;
  IdefixArray1D<real> Lambda_cool_data = cooltable_data->dataDev;
  DataBlock *data_dev_ptr = &data;
  IdefixArray1D<real> user_params_dev = *user_params_dev_ptr;

  [[maybe_unused]] real kB = idfx::units.k_B;
  [[maybe_unused]] real mu = 0.609;
  [[maybe_unused]] real m_p = idfx::units.m_p;
  [[maybe_unused]] real XH = 0.71;

  // Code-to-physical (CGS) unit conversion factors
  [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
  [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
  [[maybe_unused]] real len_unit = idfx::units.GetLength();

  [[maybe_unused]] Grid *grid_dev = data.mygrid;
  // Make a local copy of the grid
  [[maybe_unused]] GridHost gridHost(*grid_dev);
  gridHost.SyncFromDevice();


  // Ensure time considered is unique
  static real time_old = -ONE_F;
  real time_now = d.t;

  real dt = (time_old>0)?((time_now-time_old)*(len_unit/vel_unit)):ZERO_F;
  std::string analysis_filename_summary = "analysis-lambda.dat";
  std::string analysis_filename_profile = "analysis-profile.dat";
  if (idfx::prank==0) {
    if (time_old<0) {
      std::ofstream file;
      file.open(analysis_filename_summary, std::ios::out | std::ios::trunc);
      if (!file.is_open()) {
        IDEFIX_ERROR ("Problem opening summary analysis file!");
      }
      file << "# t_cool(T=" << std::scientific << temperature_mix << "K) = " << ((tcool_ib*(len_unit/vel_unit))/(365*24*60*60*1.0e+06)) << " Myr" << std::endl;
      file << "# P_ini/kB = " << std::scientific << (P0*rho_unit*pow(vel_unit,2)/kB) << " K/cm^3" << std::endl;
      file << "# Lx = " << std::scientific << (Lx1*len_unit/3.086e+18) << " pc" << std::endl;
      file << "# v_shear = " << std::scientific << (2.0*uflow*vel_unit/1.0e+05) << " km/s" << std::endl;
      file << "# " << "time" << '\t' << "Q/L^2 (erg/s/cm^2)" << '\t' << "Da" << '\t' << "vturbx (km/s)" << '\t' << "vturby (km/s)" << '\t' << "vturbz (km/s)" << std::endl;
      file.close();

      file.open(analysis_filename_profile, std::ios::out | std::ios::trunc);
      if (!file.is_open()) {
        IDEFIX_ERROR ("Problem opening profile analysis file!");
      }
      file << "# t_cool(T=" << std::scientific << temperature_mix << "K) = " << ((tcool_ib*(len_unit/vel_unit))/(365*24*60*60*1.0e+06)) << " Myr" << std::endl;
      file << "# P_ini/kB = " << std::scientific << (P0*rho_unit*pow(vel_unit,2)/kB) << " K/cm^3" << std::endl;
      file << "# Lx = " << std::scientific << (Lx1*len_unit/3.086e+18) << " pc" << std::endl;
      file << "# v_shear = " << std::scientific << (2.0*uflow*vel_unit/1.0e+05) << " km/s" << std::endl;
      file << "# " << "time" << std::endl;
      file << "# " << "<vz>" << std::endl;
      file << "# " << "<vy>" << std::endl;
      file << "# " << "<vx>" << std::endl;
      file << "# " << "<sig_vz>" << std::endl;
      file << "# " << "<sig_vy>" << std::endl;
      file << "# " << "<sig_vx>" << std::endl;
      file << "# " << "-----------------------------------" << std::endl;
      file << "# " << "z = ";
      for (int k=gridHost.nghost[KDIR]; k<(gridHost.np_int[KDIR]+gridHost.nghost[KDIR]); k++) {
        file << gridHost.x[KDIR](k) << "  ";
      }
      file << std::endl;
      file.close();
    }
  }
  if (time_now==time_old) {
    return;
  }
  else {
    time_old = time_now;
  }
  // initialize the profile array
  [[maybe_unused]] real profile[gridHost.np_int[KDIR]];
  for (int k=0; k<gridHost.np_int[KDIR]; k++) {
    profile[k] = ZERO_F;
  }

  std::ofstream file_summary;
  std::ofstream file_profile;
  std::ios_base::openmode mode = std::ios::out |  std::ios::app;
  if (idfx::prank==0) {
    // Write the data in ascii to our file
    file_summary.open(analysis_filename_summary, mode);
    if (!file_summary.is_open()) {
      IDEFIX_ERROR ("Problem opening summary analysis file!");
    }
    // Write the data in ascii to our file
    file_profile.open(analysis_filename_profile, mode);
    if (!file_profile.is_open()) {
      IDEFIX_ERROR ("Problem opening profile analysis file!");
    }
    file_summary.precision(10);
    file_profile.precision(10);

    file_profile << "# " << std::scientific << d.t << std::endl;
  }

  [[maybe_unused]] IdefixHostArray1D<real> x1 = d.x[IDIR];
  [[maybe_unused]] IdefixHostArray1D<real> x2 = d.x[JDIR];
  [[maybe_unused]] IdefixHostArray1D<real> x3 = d.x[KDIR];
  [[maybe_unused]] IdefixHostArray3D<real> dV = d.dV;

  [[maybe_unused]] int ibeg, iend, jbeg, jend, kbeg, kend;
  ibeg = d.beg[IDIR];
  iend = d.end[IDIR];
  jbeg = d.beg[JDIR];
  jend = d.end[JDIR];
  kbeg = d.beg[KDIR];
  kend = d.end[KDIR];

  [[maybe_unused]] IdefixArray4D<real> Vc_dev = data_dev_ptr->hydro->Vc;
  [[maybe_unused]] IdefixArray3D<real> dV_dev = data_dev_ptr->dV;

  real Q_Lam = ZERO_F;
  
  idefix_reduce("Sum_Q",
              kbeg,kend,
              jbeg,jend,
              ibeg,iend,
              KOKKOS_LAMBDA (int k, int j, int i, real &localSum) {
                  [[maybe_unused]] real kB = idfx::units.k_B;
                  [[maybe_unused]] real mu = 0.609;
                  [[maybe_unused]] real m_p = idfx::units.m_p;
                  [[maybe_unused]] real XH = 0.71;

                  // Code-to-physical (CGS) unit conversion factors
                  [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
                  [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
                  [[maybe_unused]] real len_unit = idfx::units.GetLength();

                  [[maybe_unused]] real Lx1 = user_params_dev(14)-user_params_dev(13);
                  [[maybe_unused]] real Lx2 = user_params_dev(16)-user_params_dev(15);
                  [[maybe_unused]] real Lx3 = user_params_dev(18)-user_params_dev(17);

                  [[maybe_unused]] real Tcl = user_params_dev(7);
                  [[maybe_unused]] real TcoolFloor = user_params_dev(12);
                  [[maybe_unused]] real chi = user_params_dev(8);

                  [[maybe_unused]] real temperature = Vc_dev(PRS,k,j,i)/Vc_dev(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  [[maybe_unused]] real Lambda;
                  if (temperature<=TcoolFloor) {
                    Lambda = ZERO_F;
                  }
                  else if ( (temperature<(1.005*Tcl)) || (temperature>(0.998*chi*Tcl)) ) {
                    Lambda = ZERO_F;
                  }
                  else {
                    int T_indx_lo = 0, T_indx_hi=temperature_data.extent(0)-1;
                    int T_indx_mid;
                    // binary search on the interpolation table
                    while (T_indx_lo<=T_indx_hi) {
                      T_indx_mid = (T_indx_lo + T_indx_hi)/2;
                      if (temperature < temperature_data(T_indx_mid)) {
                        T_indx_hi = T_indx_mid-1;
                      } else if (temperature > temperature_data(T_indx_mid)) {
                        T_indx_lo = T_indx_mid+1;
                      } else {
                        T_indx_lo = T_indx_mid;
                        T_indx_hi = T_indx_mid;
                        break;
                      }
                    }
                    if (T_indx_lo!=T_indx_hi) {
                      // swap
                      T_indx_mid = T_indx_lo;
                      T_indx_lo = T_indx_hi;
                      T_indx_hi = T_indx_mid;
                    }

                    real temperature_lo = temperature_data(T_indx_lo);
                    real temperature_hi = temperature_data(T_indx_hi);
                    real Lambda_lo = Lambda_cool_data(T_indx_lo);
                    real Lambda_hi = Lambda_cool_data(T_indx_hi);
                    // T_ref = temperature_hi
                    real alpha = std::log(Lambda_hi/Lambda_lo)/std::log(temperature_hi/temperature_lo);
                    Lambda = Lambda_lo * pow((temperature/temperature_lo), alpha);
                  }
                  localSum += (pow(Vc_dev(RHO,k,j,i)*rho_unit*XH/m_p, 2)*Lambda*dV_dev(k,j,i)/(Lx1*Lx2)); // mixed units
              },
              Kokkos::Sum<real> (Q_Lam));
  Q_Lam *= len_unit; // cgs

  
  IdefixArray2D<int> bounds_dev ("bound_indices", 3, 2); // begs, ends
  IdefixHostArray2D<int> bounds_host = Kokkos::create_mirror_view(Kokkos::HostSpace(),bounds_dev);
  for (int dir=0; dir<3; dir++) {
    bounds_host(dir,0) = d.beg[dir];
    bounds_host(dir,1) = d.end[dir];
  }
  Kokkos::deep_copy(bounds_dev, bounds_host);
  
  real tmp[gridHost.np_int[KDIR]];
  real vel_avg_prof[3][gridHost.np_int[KDIR]];
  for (int dir=3; dir>0; dir--) {
    IdefixArray1D<real> local_profile("local_profile_array", gridHost.np_int[KDIR]);
    IdefixHostArray1D<real> local_profile_host = Kokkos::create_mirror_view(Kokkos::HostSpace(),local_profile); // might not need this for a GPU aware MPI

    for (int k=kbeg; k<kend; k++) {
      [[maybe_unused]] real in_plane = ZERO_F;
      [[maybe_unused]] int nghost = user_params[25];
      idefix_reduce("prp_trml_vel_averaging_numrt",
                jbeg,jend,
                ibeg,iend,
                k, k+1,
                KOKKOS_LAMBDA (const int k, const int j, const int i, real &localSum) {
                  [[maybe_unused]] real kB = idfx::units.k_B;
                  [[maybe_unused]] real mu = 0.609;
                  [[maybe_unused]] real m_p = idfx::units.m_p;
                  [[maybe_unused]] real XH = 0.71;

                  // Code-to-physical (CGS) unit conversion factors
                  [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
                  [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
                  [[maybe_unused]] real len_unit = idfx::units.GetLength();
                  [[maybe_unused]] real Tcl = user_params_dev(7);
                  [[maybe_unused]] real chi = user_params_dev(8);
                  [[maybe_unused]] real temperature = Vc_dev(PRS,k,j,i)/Vc_dev(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  localSum += (Vc_dev(RHO,k,j,i)*Vc_dev(VX1+dir-1,k,j,i)*dV_dev(k,j,i));
                },
                Kokkos::Sum<real> (in_plane));
      local_profile_host(k-nghost) = in_plane;
    }
    for (int k=kbeg; k<kend; k++) {
      [[maybe_unused]] real in_plane = ZERO_F;
      [[maybe_unused]] int nghost = user_params[25];
      idefix_reduce("prp_trml_vel_averaging_denom",
                jbeg,jend,
                ibeg,iend,
                k, k+1,
                KOKKOS_LAMBDA (const int k, const int j, const int i, real &localSum) {
                  [[maybe_unused]] real kB = idfx::units.k_B;
                  [[maybe_unused]] real mu = 0.609;
                  [[maybe_unused]] real m_p = idfx::units.m_p;
                  [[maybe_unused]] real XH = 0.71;

                  // Code-to-physical (CGS) unit conversion factors
                  [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
                  [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
                  [[maybe_unused]] real len_unit = idfx::units.GetLength();
                  [[maybe_unused]] real Tcl = user_params_dev(7);
                  [[maybe_unused]] real chi = user_params_dev(8);
                  [[maybe_unused]] real temperature = Vc_dev(PRS,k,j,i)/Vc_dev(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  localSum += (Vc_dev(RHO,k,j,i)*dV_dev(k,j,i));
                },
                Kokkos::Sum<real> (in_plane));
      local_profile_host(k-nghost) = local_profile_host(k-nghost)/in_plane;
    }
    Kokkos::deep_copy(local_profile, local_profile_host);

    for (int k=0; k<gridHost.np_int[KDIR]; k++) {
      profile[k] = local_profile_host(k);
    }

    #ifdef WITH_MPI
    real tmp_vel_profile[gridHost.np_int[KDIR]];
    for (int k=0; k<gridHost.np_int[KDIR]; k++) {
      tmp_vel_profile[k] = profile[k];
    }
    MPI_Allreduce(tmp_vel_profile, profile, gridHost.np_int[KDIR], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    #endif

    if (idfx::prank==0) {
    file_profile << " ";
      for (int k=0; k<gridHost.np_int[KDIR]; k++) {
        vel_avg_prof[dir-1][k] = profile[k];
        file_profile << std::scientific << vel_avg_prof[dir-1][k] << '\t';
      }
      file_profile << std::endl;
    }
    
    idefix_for("local_profile_reset",
              0, gridHost.np_int[KDIR],
              KOKKOS_LAMBDA(const int k) {
                local_profile(k) = ZERO_F;  
              }
    );
    Kokkos::deep_copy(local_profile_host, local_profile);
    
  } 
  
  real vel_turb_prof[3][gridHost.np_int[KDIR]];
  for (int dir=3; dir>0; dir--) {
    IdefixHostArray1D<real> local_profile("local_profile_array", gridHost.np_int[KDIR]);
    for (int k=kbeg; k<kend; k++) {
      [[maybe_unused]] real in_plane = ZERO_F;
      [[maybe_unused]] int nghost = user_params[25];
      idefix_reduce("prp_trml_vel_turb_numrt",
                jbeg,jend,
                ibeg,iend,
                k, k+1,
                KOKKOS_LAMBDA (const int k, const int j, const int i, real &localSum) {
                  [[maybe_unused]] real kB = idfx::units.k_B;
                  [[maybe_unused]] real mu = 0.609;
                  [[maybe_unused]] real m_p = idfx::units.m_p;
                  [[maybe_unused]] real XH = 0.71;

                  // Code-to-physical (CGS) unit conversion factors
                  [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
                  [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
                  [[maybe_unused]] real len_unit = idfx::units.GetLength();
                  [[maybe_unused]] real Tcl = user_params_dev(7);
                  [[maybe_unused]] real chi = user_params_dev(8);
                  [[maybe_unused]] real temperature = Vc_dev(PRS,k,j,i)/Vc_dev(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  localSum += (Vc_dev(RHO,k,j,i)*Vc_dev(VX1+dir-1,k,j,i)*Vc_dev(VX1+dir-1,k,j,i)*dV_dev(k,j,i));
                },
                Kokkos::Sum<real> (in_plane));
      profile[k-nghost] = in_plane;
    }

    #ifdef WITH_MPI
    real tmp_vturb_profile_numrt[gridHost.np_int[KDIR]];
    for (int k=0; k<gridHost.np_int[KDIR]; k++) {
      tmp_vturb_profile_numrt[k] = profile[k];
    }
    MPI_Allreduce(tmp_vturb_profile_numrt, profile, gridHost.np_int[KDIR], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    #endif

    // local profile reset
    for (int k=kbeg; k<kend; k++) {
      [[maybe_unused]] int nghost = user_params[25];
      local_profile_host(k-nghost) = ZERO_F;
    }
    for (int k=kbeg; k<kend; k++) {
      [[maybe_unused]] real in_plane = ZERO_F;
      [[maybe_unused]] int nghost = user_params[25];
      idefix_reduce("prp_trml_vel_turb_denom",
                jbeg,jend,
                ibeg,iend,
                k, k+1,
                KOKKOS_LAMBDA (const int k, const int j, const int i, real &localSum) {
                  [[maybe_unused]] real kB = idfx::units.k_B;
                  [[maybe_unused]] real mu = 0.609;
                  [[maybe_unused]] real m_p = idfx::units.m_p;
                  [[maybe_unused]] real XH = 0.71;

                  // Code-to-physical (CGS) unit conversion factors
                  [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
                  [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
                  [[maybe_unused]] real len_unit = idfx::units.GetLength();
                  [[maybe_unused]] real Tcl = user_params_dev(7);
                  [[maybe_unused]] real chi = user_params_dev(8);
                  [[maybe_unused]] real temperature = Vc_dev(PRS,k,j,i)/Vc_dev(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  localSum += (Vc_dev(RHO,k,j,i)*dV_dev(k,j,i));
                },
                Kokkos::Sum<real> (in_plane));
      local_profile_host(k-nghost) = in_plane;
    }
    
    for (int k=0; k<gridHost.np_int[KDIR]; k++) {
      tmp[k] = local_profile_host(k);
    }

    #ifdef WITH_MPI
    real tmp_vturb_profile_denom[gridHost.np_int[KDIR]];
    for (int k=0; k<gridHost.np_int[KDIR]; k++) {
      tmp_vturb_profile_denom[k] = tmp[k];
    }
    MPI_Allreduce(tmp_vturb_profile_denom, tmp, gridHost.np_int[KDIR], MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    #endif
    for (int k=0; k<gridHost.np_int[KDIR]; k++) {
      profile[k] = profile[k]/tmp[k];
    }

    if (idfx::prank==0) {
      file_profile << " ";
      for (int k=0; k<gridHost.np_int[KDIR]; k++) {
        vel_turb_prof[dir-1][k] = sqrt(std::abs(profile[k] - vel_avg_prof[dir-1][k]*vel_avg_prof[dir-1][k]));
        file_profile << std::scientific << vel_turb_prof[dir-1][k] << '\t';
      }
      file_profile << std::endl;
    }
    
    idefix_for("local_profile_reset",
              0, gridHost.np_int[KDIR],
              KOKKOS_LAMBDA(const int k) {
                local_profile(k) = ZERO_F;  
              }
    );
    Kokkos::deep_copy(local_profile_host, local_profile);
  // }

  [[maybe_unused]] real numrt = ZERO_F;
  [[maybe_unused]] real denom = ZERO_F;
  [[maybe_unused]] real vol_tot = ZERO_F;
  idefix_reduce("Sum_mass",
              kbeg,kend,
              jbeg,jend,
              ibeg,iend,
              KOKKOS_LAMBDA (int k, int j, int i, real &localSum) {
                  [[maybe_unused]] real kB = idfx::units.k_B;
                  [[maybe_unused]] real mu = 0.609;
                  [[maybe_unused]] real m_p = idfx::units.m_p;
                  [[maybe_unused]] real XH = 0.71;

                  // Code-to-physical (CGS) unit conversion factors
                  [[maybe_unused]] real vel_unit = idfx::units.GetVelocity();
                  [[maybe_unused]] real rho_unit = idfx::units.GetDensity();
                  [[maybe_unused]] real len_unit = idfx::units.GetLength();
                  [[maybe_unused]] real Tcl = user_params_dev(7);
                  [[maybe_unused]] real chi = user_params_dev(8);
                  [[maybe_unused]] real temperature = Vc_dev(PRS,k,j,i)/Vc_dev(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  if (!( (temperature<(1.005*Tcl)) || (temperature>(0.998*chi*Tcl)) )) 
                    localSum += (Vc_dev(RHO,k,j,i)*dV_dev(k,j,i));
              },
              Kokkos::Sum<real> (denom));

  // real v_turb = sqrt(vturb_1*vturb_1 + vturb_2*vturb_2 + vturb_3*vturb_3);
  real v_turb[3];
  for (int dir=3; dir>0; dir--) {
    v_turb[dir-1] = -99999999;
    for (int k=0; k<gridHost.np_int[KDIR]; k++) {
      v_turb[dir-1] = (v_turb[dir-1]<vel_turb_prof[dir-1][k])?vel_turb_prof[dir-1][k]:vel_turb_prof[dir-1][k]; 
    }
  }
  #ifdef WITH_MPI
    real tmp_v_turb[3];
    for (int dir=0; dir<3; dir++) {
      tmp_v_turb[dir] = v_turb[dir];
    }
    MPI_Allreduce(tmp_v_turb, v_turb, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  #endif

  real t_mix = (v_turb[KDIR]>ZERO_F)?(Lx1/v_turb[KDIR]):ZERO_F;
  real Da = t_mix/tcool_ib;
  // idfx::cout << "DEBUG: " << std::scientific << d.t << '\t' << Q_Lam << '\t' << Da << '\t' << (tcool_ib*(len_unit/vel_unit)) << '\t' << (t_mix*(len_unit/vel_unit)) << std::endl;

  if (idfx::prank==0) {
    file_summary << "  " << std::scientific << d.t << '\t' << Q_Lam << '\t' << Da << '\t' << (v_turb[0]*vel_unit/1.0e+05) << '\t' << (v_turb[1]*vel_unit/1.0e+05) << '\t' << (v_turb[2]*vel_unit/1.0e+05) << std::endl;
    file_summary.close();
    file_profile.close();
  }
  */
}