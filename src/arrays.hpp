#ifndef IDEFIX_ARRAYS_HPP_
#define IDEFIX_ARRAYS_HPP_

#include "real_types.h"
#include "idefix.hpp"

template <typename T> using IdefixArray1D = Kokkos::View<T*, Layout, Device>;
template <typename T> using IdefixArray2D = Kokkos::View<T**, Layout, Device>;
template <typename T> using IdefixArray3D = Kokkos::View<T***, Layout, Device>;
template <typename T> using IdefixArray4D = Kokkos::View<T****, Layout, Device>;

/*
template <typename T> using IdefixHostArray1D = Kokkos::View<T*, Layout, Host>;
template <typename T> using IdefixHostArray2D = Kokkos::View<T**, Layout, Host>;
template <typename T> using IdefixHostArray3D = Kokkos::View<T***, Layout, Host>;
template <typename T> using IdefixHostArray4D = Kokkos::View<T****, Layout, Host>;
*/


#endif // IDEFIX_ARRAYS


