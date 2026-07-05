add_idefix_source(frameBoost.cpp)

replace_idefix_source(${IDEFIX_DIR}/src/fluid/cooling/cooling.cpp cooling.cpp)
replace_idefix_source(${IDEFIX_DIR}/src/timeIntegrator.cpp timeIntegrator.cpp)
replace_idefix_source(${IDEFIX_DIR}/src/main.cpp main.cpp)
replace_idefix_source(${IDEFIX_DIR}/src/output/output.cpp output.cpp)

# all local header files are chosen first
target_include_directories(idefix BEFORE PUBLIC ${CMAKE_CURRENT_LIST_DIR})

# ---------------------------------------------------------------------------
# Override the framework's src/fluid/fluid.hpp with THIS directory's copy for
# every translation unit, without modifying anything under src/.
#
# Why a plain -I override (target_include_directories BEFORE) is not enough:
# headers/sources inside src/fluid/ (e.g. viscosity.hpp, viscosity.cpp,
# thermalDiffusion.cpp, ...) do  #include "fluid.hpp"  with quotes. The C++
# quote-include rule searches the *including file's own directory first*, so
# they always find their sibling src/fluid/fluid.hpp before any -I path. If
# some TUs then saw the framework fluid.hpp and others saw our local one, the
# two different Fluid<Phys> layouts would be an ODR violation (silent memory
# corruption at runtime), not just a compile error.
#
# Force-including our local fluid.hpp makes the compiler parse it as the very
# first line of every idefix TU. That sets the shared FLUID_FLUID_HPP_ guard
# with OUR definition, so every later  #include "fluid.hpp"  (framework or
# local) is a no-op. One consistent definition everywhere. Because this file
# lives here, it also transitively pulls the local cooling.hpp,
# coolingConstructor.hpp, showConfig.hpp and frameBoost.hpp (their guards match
# the framework copies, which are therefore suppressed too).
# ---------------------------------------------------------------------------
target_compile_options(idefix BEFORE PRIVATE
  "SHELL:-include ${CMAKE_CURRENT_LIST_DIR}/fluid.hpp")
