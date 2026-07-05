// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_FRAME_BOOST_HPP_
#define FLUID_FRAME_BOOST_HPP_

#include <string>
#include <iostream>
#include <vector>
#include <array>
#include <fstream>
#include <filesystem> // C++17 feature to check file existence
#include <iomanip> // For setprecision

#include "idefix.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "fluid_defs.hpp"

/* Useful bulk reduction code; not utilized right now
// Define a struct to hold the reduction values.
// This struct and its reduction identity must be defined outside the function.
struct ReductionData {
  real dV_trml;
  real rho_dV_trml;

  // Add the join method here to define how two ReductionData objects are combined.
  KOKKOS_FORCEINLINE_FUNCTION void join(const volatile ReductionData& src) volatile {
    dV_trml += src.dV_trml;
    rho_dV_trml += src.rho_dV_trml;
  }

  KOKKOS_INLINE_FUNCTION
  ReductionData& operator+=(const ReductionData& src) {
    dV_trml += src.dV_trml;
    rho_dV_trml += src.rho_dV_trml;
    return *this;
  }
};

// Specialize the Kokkos::reduction_identity template for the struct.
// This tells Kokkos how to initialize the struct for reductions.
namespace Kokkos {
template <>
struct reduction_identity<ReductionData> {
  KOKKOS_FORCEINLINE_FUNCTION static ReductionData sum() {
    return {ZERO_F, ZERO_F};
  }
};
}  // namespace Kokkos
*/

// Forward class hydro declaration
template <typename Phys> class Fluid;
class DataBlock;

class FrameBoost {
 public:
  template <typename Phys>
  FrameBoost(Input &, Grid &, Fluid<Phys> *);

  void ShowConfig();      // print configuration
  void CalculateBoost();  // calculate the boost velocity
  void ApplyBoost();      // apply the boost velocity
  void saveToText(const std::vector<std::array<real, 3>>&, const std::string&); // save boost history to a text file
  void loadOrInitialize(std::vector<std::array<real, 3>>&, const std::string&); // load or initialize boost history
  void RestartBoost();    // handle boost state on restart

 private:
  DataBlock* data;
  Grid &grid;
  Input &input;

  // The only genuine device data the boost touches: the simulation fields.
  IdefixArray4D<real> &Vc;
  IdefixArray4D<real> &Vs;
  IdefixArray4D<real> &Uc;

  // ---------------------------------------------------------------------------
  // Boost state. Every quantity the boost logic needs is a small scalar that
  // lives on the host; the two real device kernels (the temperature reductions
  // and ApplyBoost) receive the values they need by value. This removes all the
  // single-element device arrays, their host mirrors, and the deep_copy traffic
  // that used to shuttle these scalars back and forth (same idea as setup.cpp).
  // ---------------------------------------------------------------------------
  // constant after construction
  real temp_intervals[2]{};   // [lo, hi] temperature window that defines the TRML
  real boost_region[3]{};     // boost activation region
  real interval_boost{};      // minimum time between boosts

  // evolving boost state
  real boost_vel[2]{};        // [previous, current] boost velocity
  real trml_pos{};            // tracked TRML interface position
  real trml_pos_ini{};        // initial TRML position (z2 or restart value)
  real time_old{};            // time of the previous boost
  real time_new{};            // current time being processed
  real dt_now{};              // current timestep
  real vz_lab{};              // cumulative lab-frame velocity
  real posz_trml_extent[2]{}; // [min, max] z of the TRML interface
  real posz_trml_com{};       // TRML center of mass (z)
  real boost_restart_time{};  // simulation time at which a restart is evaluated
  bool first_boost{false};

  real time_now{};            // last time CalculateBoost processed
  real dt_step{};
  real trml_posz_host{};      // interface position recorded in the boost history

  std::vector<std::array<real, 3>> boost_history;
  std::string boost_history_file;

  bool initialized{false};
  bool on_from_ini{false};
  bool restarted{false};
  bool first_apply{true};
};

#include "fluid.hpp"

template<typename Phys>
FrameBoost::FrameBoost(Input &input, Grid &grid, Fluid<Phys> *hydroin):
                       data(hydroin->data),
                       grid(grid),
                       input(input),
                       Vc(hydroin->Vc),
                       Vs(hydroin->Vs),
                       Uc(hydroin->Uc) {
  idfx::pushRegion("FrameBoost::FrameBoost");
  this->initialized = true;
  if(input.CheckEntry("Boost", "on")>=0) {
    this->on_from_ini = (input.Get<int>("Boost", "on", 0)==0)? false: true;
  } else {
    idfx::popRegion();
    return;
  }
  this->boost_history_file = input.Get<std::string>("Boost", "history_file", 0);
  this->time_now = ZERO_F;
  this->dt_step = ZERO_F;

  if(input.CheckEntry("Setup","z2")>=0) {
    trml_pos_ini = input.Get<real>("Setup","z2",0);
  } else {
    trml_pos_ini = ZERO_F;
  }
  time_new = -ONE_F;
  vz_lab = ZERO_F;
  first_boost = false;
  time_old = ZERO_F; // data->t;
  boost_vel[0] = ZERO_F;
  boost_vel[1] = ZERO_F;
  posz_trml_extent[0] = input.Get<real>("Grid", "X3-grid", 4);
  posz_trml_extent[1] = input.Get<real>("Grid", "X3-grid", 1);
  posz_trml_com = ZERO_F;
  boost_restart_time = data->t;
  trml_pos = trml_pos_ini;

  /*
  if (!hydroin->haveTracer || (hydroin->nTracer<1)) {
    IDEFIX_ERROR("FrameBoost: The number of passive tracers should be >= 1");
  }
  */
  // hydroin->haveSourceTerms = true;

  if(input.CheckEntry("Boost","temperature")>=0) {
      temp_intervals[0] = input.Get<real>("Boost","temperature",0);
      temp_intervals[1] = input.Get<real>("Boost","temperature",1);
  } else {
    IDEFIX_ERROR("Unknown/missing temperature options for Boost in idefix.ini.");
  }
  if(input.CheckEntry("Boost","interval_time")>=0) {
      interval_boost = input.Get<real>("Boost","interval_time",0);
  } else {
    IDEFIX_ERROR("Unknown/missing time interval option for Boost in idefix.ini.");
  }

  if(input.CheckEntry("Boost","region")>=0) {
      boost_region[0] = input.Get<real>("Boost","region",0);
      boost_region[1] = input.Get<real>("Boost","region",1);
      boost_region[2] = input.Get<real>("Boost","region",2);
  } else {
    IDEFIX_ERROR("Unknown/missing region options for Boost in idefix.ini.");
  }

  this->boost_history.reserve(10000); // Crucial for performance
  // this->RestartBoost();
  idfx::popRegion();
}

#endif // FLUID_FRAME_BOOST_HPP_
