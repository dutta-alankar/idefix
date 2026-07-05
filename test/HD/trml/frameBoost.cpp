// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#include <string>
#include <sstream>

#include "idefix.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "units.hpp"
#include "frameBoost.hpp"
#include "dataBlock.hpp"
#include "fluid.hpp"

void FrameBoost::RestartBoost() {
  idfx::pushRegion("FrameBoost::RestartBoost");
  // On restart we keep the same boost velocity as before, so we don't recalculate
  // it here; we just make sure it is applied correctly in ApplyBoost() and that
  // the history is preserved. If we were not boosting from the start, there is
  // nothing to restart.
  if (!on_from_ini) {
    idfx::popRegion();
    return;
  }

  auto &boost_history = this->boost_history;

  this->loadOrInitialize(boost_history, this->boost_history_file);
  #ifdef WITH_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  #endif
  if (boost_history.empty() &&
      (boost_restart_time > input.Get<real>("TimeIntegrator", "first_dt", 0))) {
    idfx::cout << "FrameBoost: Restart history is empty, keeping default boost state." << std::endl;
    idfx::popRegion();
    return;
  }

  if (this->restarted) {
    idfx::cout << "FrameBoost: Restarting with existing boost history at time="
               << boost_restart_time << std::endl;
    idfx::cout << "FrameBoost: Last boost time="
               << boost_history.back()[0] << ", boost_vel=" << boost_history.back()[1]
               << ", interface_pos=" << boost_history.back()[2] << std::endl;
  } else {
    idfx::popRegion();
    return;
  }

  time_old = boost_history.back()[0];
  if (boost_history.size() >= 2) {
    boost_vel[0] = boost_history[boost_history.size() - 2][1];
    boost_vel[1] = boost_history.back()[1];
  } else {
    boost_vel[0] = boost_history.back()[1];
    boost_vel[1] = boost_history.back()[1];
  }
  trml_pos = boost_history.back()[2];

  real totalSum = 0.0;
  for (const auto& row : boost_history) {
    totalSum += row[1]; // Index 1 is the velocity boost
  }
  vz_lab = totalSum; // cumulative boost velocity from history
  idfx::popRegion();
}

void FrameBoost::CalculateBoost() {
  idfx::pushRegion("FrameBoost::CalculateBoost");
  DataBlock* data = this->data;
  if (!on_from_ini) {
    boost_vel[0] = ZERO_F;
    boost_vel[1] = ZERO_F;
    idfx::popRegion();
    return;
  }

  // Genuine device data used by the reductions below.
  IdefixArray1D<real> x3 = data->x[KDIR];
  IdefixArray3D<real> dV = data->dV;
  IdefixArray4D<real> Vc = this->Vc;

  // Host-only unit factors and the (constant) temperature window; captured by
  // value into the device reductions, exactly like idfx::units in setup.cpp.
  const real kB = idfx::units.k_B;
  const real m_p = idfx::units.m_p;
  const real vel_unit = idfx::units.GetVelocity();
  const real mu = 0.609;
  const real temp_lo = temp_intervals[0];
  const real temp_hi = temp_intervals[1];

  // re-initialize every iteration
  posz_trml_extent[0] = input.Get<real>("Grid", "X3-grid", 4);
  posz_trml_extent[1] = input.Get<real>("Grid", "X3-grid", 1);

  if (time_now==data->t && !this->restarted) {
    boost_vel[0] = ZERO_F;
    boost_vel[1] = ZERO_F;
    idfx::popRegion();
    return;
  } else {
    time_now = data->t;
    dt_step  = data->dt;
  }

  int ibeg = data->beg[IDIR];
  int iend = data->end[IDIR];
  int jbeg = data->beg[JDIR];
  int jend = data->end[JDIR];
  int kbeg = data->beg[KDIR];
  int kend = data->end[KDIR];

  real myMin = posz_trml_extent[0];   // Note that the result will be stored on the host!
  idefix_reduce("Minimum",
                kbeg,kend,
                jbeg,jend,
                ibeg,iend,
                KOKKOS_LAMBDA (int k, int j, int i, real &localMin) {
                  [[maybe_unused]] real temperature = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  if ((temperature>=temp_lo) && (temperature<=temp_hi))
                    localMin = std::fmin(localMin, x3(k));
                },
                Kokkos::Min<real>(myMin));
  #ifdef WITH_MPI
  {
    real send_elem = myMin;
    MPI_Allreduce(&send_elem, &myMin, 1, realMPI, MPI_MIN, MPI_COMM_WORLD);
  }
  #endif
  real myMax = posz_trml_extent[1];   // Note that the result will be stored on the host!
  idefix_reduce("Maximum",
                kbeg,kend,
                jbeg,jend,
                ibeg,iend,
                KOKKOS_LAMBDA (int k, int j, int i, real &localMax) {
                  [[maybe_unused]] real temperature = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  if ((temperature>=temp_lo) && (temperature<=temp_hi))
                    localMax = std::fmax(localMax, x3(k));
                },
                Kokkos::Max<real>(myMax));
  #ifdef WITH_MPI
  {
    real send_elem = myMax;
    MPI_Allreduce(&send_elem, &myMax, 1, realMPI, MPI_MAX, MPI_COMM_WORLD);
  }
  #endif

  posz_trml_extent[0] = myMin;
  posz_trml_extent[1] = myMax;

  // if (data->t >= 7.5) {
  //   idfx::cout << "FrameBoost: TRML interface at time=" << std::scientific << std::setprecision(2) << time_now
  //              << " is between " << posz_trml_extent[0] << " and " << posz_trml_extent[1]
  //              << " in the z-direction, based on the temperature range between "
  //              << temp_intervals[0] << " and " << temp_intervals[1] << std::endl;
  // }

  if (myMin>myMax) {
    // idfx::cout << "FrameBoost: No cells found in the specified temperature range for TRML. No boost will be applied." << std::endl;
    boost_vel[0] = ZERO_F;
    boost_vel[1] = ZERO_F;
    idfx::popRegion();
    return;
  }

  // store center of mass
  real myAvg = ZERO_F;   // Note that the result will be stored on the host!
  idefix_reduce("Average",
                kbeg,kend,
                jbeg,jend,
                ibeg,iend,
                KOKKOS_LAMBDA (int k, int j, int i, real &localAvg) {
                  [[maybe_unused]] real temperature = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  if ((temperature>=temp_lo) && (temperature<=temp_hi))
                    localAvg += x3(k)*Vc(RHO,k,j,i)*dV(k,j,i);
                },
                Kokkos::Sum<real>(myAvg));
  #ifdef WITH_MPI
  {
    real send_elem = myAvg;
    MPI_Allreduce(&send_elem, &myAvg, 1, realMPI, MPI_SUM, MPI_COMM_WORLD);
  }
  #endif
  posz_trml_com = myAvg;
  myAvg = ZERO_F;   // Note that the result will be stored on the host!
  idefix_reduce("Average",
                kbeg,kend,
                jbeg,jend,
                ibeg,iend,
                KOKKOS_LAMBDA (int k, int j, int i, real &localAvg) {
                  [[maybe_unused]] real temperature = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
                  if ((temperature>=temp_lo) && (temperature<=temp_hi))
                    localAvg += Vc(RHO,k,j,i)*dV(k,j,i);
                },
                Kokkos::Sum<real>(myAvg));
  #ifdef WITH_MPI
  {
    real send_elem = myAvg;
    MPI_Allreduce(&send_elem, &myAvg, 1, realMPI, MPI_SUM, MPI_COMM_WORLD);
  }
  #endif
  posz_trml_com = posz_trml_com/myAvg;

  // if (data->t >= 7.5) {
  //   idfx::cout << "FrameBoost: TRML center of mass at time=" << std::scientific << std::setprecision(2) << time_now
  //              << " is at " << posz_trml_com
  //              << " in the z-direction, based on the temperature range between "
  //              << temp_intervals[0] << " and " << temp_intervals[1] << std::endl;
  // }

  /*
  // Alternative single-pass reduction (kept for reference), see ReductionData in frameBoost.hpp:
  Kokkos::View<ReductionData, Kokkos::DefaultExecutionSpace::memory_space> reduction_result("reduction_result");
  Kokkos::parallel_reduce("combined_reduction",
    Kokkos::MDRangePolicy<Kokkos::Rank<3>>({kbeg, jbeg, ibeg}, {kend + 1, jend + 1, iend + 1}),
    KOKKOS_LAMBDA(int k, int j, int i, ReductionData& lsum) {
      [[maybe_unused]] real temperature = Vc(PRS,k,j,i)/Vc(RHO,k,j,i)*(mu*m_p/kB)*pow(vel_unit,2);
      [[maybe_unused]] real rho_dV = Vc(RHO,k,j,i) * dV(k,j,i);
      if ((temperature>=temp_lo) && (temperature<=temp_hi)) {
        lsum.rho_dV_trml += rho_dV;
        lsum.dV_trml += dV(k,j,i);
      }
    },
  Kokkos::Sum<ReductionData, Kokkos::DefaultExecutionSpace::memory_space>(reduction_result));
  */

  this->time_new = data->t;
  this->dt_now   = data->dt;

  // -------------------------------------------------------------------------
  // Final boost-velocity update. This was a single-thread device kernel
  // ("calculate_and_assign_boost_vel"); it is pure scalar logic operating on
  // the (now host-resident) boost state, so it runs directly on the host.
  // -------------------------------------------------------------------------
  {
    real interface_pos = HALF_F*(posz_trml_extent[0] + posz_trml_extent[1]);
    // XXX: assuming unity box size in the smallest direction
    if ((posz_trml_com<=boost_region[0]) || ((posz_trml_extent[1]-posz_trml_extent[0])>0.4)) {
      // TRML left edge is outside the left boost region; use the center of mass.
      interface_pos = posz_trml_com; // posz_trml_extent[0];
    }
    real interface_vel = ZERO_F;
    if (FABS(time_new-time_old)>1.0e-08) {
      interface_vel = (interface_pos-trml_pos)/(time_new-time_old);
    }
    bool boost_time_criterion = false; // boost based on time interval
    // time elapsed since last boost
    real elapsed = time_new - time_old; // FMIN(interval_boost, time_new - time_old);
    if ( (elapsed<=interval_boost && ((elapsed+dt_now)>=interval_boost)) ||
          elapsed>interval_boost ) { // last clause is true when restart/first boost begins
      time_old = time_new;
      trml_pos = interface_pos;
      boost_time_criterion = true;
    }
    boost_vel[0] = boost_vel[1];

    // Check condition and update boost_vel.
    if ( (((interface_pos >= boost_region[1]) && (interface_pos <= boost_region[2])) ||
          (interface_pos <= boost_region[0]) ) && boost_time_criterion ) {
      boost_vel[1] = interface_vel; // ZERO_F;
      first_boost = true;
    }
    /*
    else if (!first_boost) {
      boost_vel[0] = ZERO_F;
      boost_vel[1] = ZERO_F;
    }
    */
    if (FABS(boost_vel[0]-boost_vel[1])>1.0e-08) {
      vz_lab += boost_vel[1];
    }
  }

  this->trml_posz_host = HALF_F*(posz_trml_extent[0] + posz_trml_extent[1]);
  if ((posz_trml_com<=boost_region[0]) || ((posz_trml_extent[1]-posz_trml_extent[0]))>0.4) {
    this->trml_posz_host = posz_trml_com; // posz_trml_extent[0];
  }
  /*
  bool override = false;
  if ((FABS(boost_vel[0]-boost_vel[1])>1.0e-08) || override) {
    idfx::cout << "DEBUG: " << std::scientific << std::setprecision(4)
              << " t= " << time_new
              << " left position: " << posz_trml_extent[0]
              << " right position: " << posz_trml_extent[1]
              << " com: " << posz_trml_com
              << " vz: "
              << " old= " << boost_vel[0]
              << " new= " << boost_vel[1]
              << " boost region: "
              << boost_region[0] << ", "
              << boost_region[1] << ", "
              << boost_region[2] << std::endl;
  }
  */
  idfx::popRegion();
}

void FrameBoost::ApplyBoost() {
  idfx::pushRegion("FrameBoost::ApplyBoost");
  // idfx::cout << "DEBUG: FrameBoost::ApplyBoost" << std::endl;
  if (this->first_apply) {
    this->RestartBoost();
    first_apply = false;
  }
  this->CalculateBoost();
  DataBlock *data = this->data;
  Input &input = this->input;
  // Boost velocities captured by value into the device kernel.
  const real bv0 = boost_vel[0];
  const real bv1 = boost_vel[1];
  [[maybe_unused]] IdefixArray4D<real> Uc = this->Uc;
  [[maybe_unused]] IdefixArray4D<real> Vc = this->Vc;

  idefix_for("ApplyBoost",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
    KOKKOS_LAMBDA (int k, int j, int i) {
      // real momz_old = Uc(MX3,k,j,i);
      if (FABS(bv0-bv1)>1.0e-08) {
        Vc(VX3,k,j,i) -= bv1;
        Uc(ENG,k,j,i) += (HALF_F*Uc(RHO,k,j,i)*pow(bv1,2)-bv1*Uc(MX3,k,j,i));
        Uc(MX3,k,j,i) -= (Uc(RHO,k,j,i)*bv1);
      }
    });
  idfx::popRegion();
  idfx::pushRegion("FrameBoost::ApplyBoost saving boost data");
  // Save the boost data for the next step
  auto &boost_history = this->boost_history;

  if (boost_history.empty()) {
    real z2;
    if(input.CheckEntry("Setup","z2")>=0) {
      z2 = input.Get<real>("Setup","z2",0);
    } else {
      z2 = ZERO_F;
    }
    boost_history.push_back({ZERO_F, ZERO_F, z2});
  }
  if (FABS(bv0-bv1)>1.0e-08) {
    real trml_posz_host = this->trml_posz_host;
    real time = data->t;
    real boost_vel_now = bv1;
    boost_history.push_back({time, boost_vel_now, trml_posz_host});
    this->saveToText(boost_history, this->boost_history_file);
    #ifdef WITH_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
  }
  idfx::popRegion();
}

void FrameBoost::ShowConfig() {
  if (!initialized) {
    IDEFIX_ERROR("Frame boost not initialized.");
  }
  if (on_from_ini) {
    std::ios old_state(nullptr);
    old_state.copyfmt(std::cout);
    idfx::cout << "FrameBoost: Operating for temperature values between "
               << std::scientific << std::setprecision(1)
               << temp_intervals[0] << " K and " << temp_intervals[1] << " K";
    std::cout.copyfmt(old_state);
    idfx::cout << " and active when the TRML interface is in the region between "
               << boost_region[1] << " and " << boost_region[2]
               << " less than region " << boost_region[0]
               << " perpendicular to the shear direction." << std::endl;
  } else {
    if(input.CheckEntry("Boost", "on")>=0)
      idfx::cout << "FrameBoost: Turned off!" << std::endl;
  }
}

void FrameBoost::saveToText(const std::vector<std::array<real, 3>>& boost_table_matrix, const std::string& filename) {
  if (idfx::prank==0) {
    std::ofstream outFile(filename);
    idfx::pushRegion("FrameBoost::saveToText");
    if (outFile.is_open()) {
      // Set precision to 15-17 digits for double to avoid data loss
      outFile << std::scientific << std::setprecision(15);

      for (const auto& row : boost_table_matrix) {
        outFile << row[0] << " " << row[1] << " " << row[2] << "\n";
      }

      outFile.close();
      // idfx::cout << "Data successfully saved to " << filename << std::endl;
    } else {
      std::cerr << "Error: Could not open file for writing." << std::endl;
    }
    idfx::popRegion();
  }
}

void FrameBoost::loadOrInitialize(std::vector<std::array<real, 3>>& boost_table_matrix, const std::string& filename) {
  idfx::pushRegion("FrameBoost::loadOrInitialize");
  DataBlock* data = this->data;
  // Record the (restart) evaluation time.
  boost_restart_time = data->t;
  // Check if file exists and starts at a later time
  if (std::filesystem::exists(filename) &&
      boost_restart_time > input.Get<real>("TimeIntegrator","first_dt",0)) {
    std::ifstream inFile(filename);
    real v1, v2, v3;
    idfx::cout << "FrameBoost: Found existing file. Loading data..." << std::endl;

    // Read triples of real values until the end of the file
    while (inFile >> v1 >> v2 >> v3) {
      // only load rows where the time value (v1) is <= the current simulation time
      if (v1<=boost_restart_time) {
        boost_table_matrix.push_back({v1, v2, v3});
        trml_pos_ini = v3; // Update trml_pos_ini with the last position from the file
      }
    }

    inFile.close();
    if (!boost_table_matrix.empty()) {
      this->restarted = true;
      idfx::cout << "FrameBoost: Resuming with " << boost_table_matrix.size() << " of the existing rows." << std::endl;
      trml_pos = trml_pos_ini;
    }
  } else {
    if (boost_restart_time > input.Get<real>("TimeIntegrator","first_dt",0)) {
      idfx::cout << "FrameBoost: No existing file found at t = " << boost_restart_time << ". Either boost has not started at restart time or boost is inactive." << std::endl;
    }
    // Optional: boost_table_matrix.reserve(1000);
  }
  idfx::popRegion();
}
