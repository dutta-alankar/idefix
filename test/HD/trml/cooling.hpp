// ***********************************************************************************
// Idefix MHD astrophysical code
// Copyright(C) Geoffroy R. J. Lesur <geoffroy.lesur@univ-grenoble-alpes.fr>
// and other code contributors
// Licensed under CeCILL 2.1 License, see COPYING for more information
// ***********************************************************************************

#ifndef FLUID_COOLING_COOLING_HPP_
#define FLUID_COOLING_COOLING_HPP_

#include <string>
#include <functional>
#include "idefix.hpp"
#include "input.hpp"
#include "grid.hpp"
#include "fluid_defs.hpp"
#include "eos.hpp"
#include "lookupTable.hpp"

// Forward class hydro declaration
template <typename Phys> class Fluid;
class DataBlock;
class EquationOfState;

class RadCooling {
 public:
  template <typename Phys>
  RadCooling(Input &, Grid &, Fluid<Phys> *);
  void ShowConfig();                      // print configuration

  // Function for internal use (but public to allow for device lambda capture)
  void InitArrays();

  LookupTable<1> cooltable_data;
  IdefixArray1D<real> temperature_data;
  IdefixArray1D<real> Lambda_cool_data;

  real TcoolFloor;
  /* TRML extras */
  real Tcl, chi; 

  IdefixArray3D<real> delta_eng;  // Source terms of the cooling operator
  /* TRML extras */
  IdefixArray3D<real> delta_eng_total;  // Accumulated the energy loss due to cooling for every cell
  IdefixHostArray3D<real> delta_eng_total_host;  // Accumulated the energy loss synced on host
  std::string cooling_type;
  std::string cooling_integration;
  std::string cooltable_filename;

  std::function<void(real)> rad_cooling_int_func;
  void TownsendIntegration(real);
  void CalculateCoolingSource(real);            // add the source term to fluid equation

 private:
  DataBlock* data;
  EquationOfState eos;

  IdefixArray4D<real> &Vc;
  IdefixArray4D<real> &Vs;
};

#endif // FLUID_COOLING_COOLING_HPP_
