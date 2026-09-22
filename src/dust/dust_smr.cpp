//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_smr.cpp
//! \brief Precomputed virtual-grid maps for the experimental smooth-P2 transfer on a
//! single-rank, two-dimensional, static 2:1 mesh.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/mesh.hpp"
#include "mesh/meshblock.hpp"
#include "dust.hpp"

namespace dust {
namespace {

constexpr int kMaxCoarseMapEntries = 4;
constexpr int kMaxFineMapEntries = 32;

int EncodeLeafCell(const int m, const int j, const int i,
                   const int ncells2, const int ncells1) {
  return (m*ncells2 + j)*ncells1 + i;
}

} // namespace

//----------------------------------------------------------------------------------------
//! \fn DustGasDrag::BuildSMRCompositeTransfer
//! \brief Build G_H and G_h as sparse maps from virtual uniform cells to active leaves.
//!
//! G_H restricts fine leaves to a root-level virtual coarse cell by volume averaging.
//! G_h is the identity in the refined region. In the coarse region it reconstructs
//! virtual fine children with an unlimited centered linear slope. Four children of a
//! coarse parent cancel the slope corrections exactly, so both maps conserve the
//! volume integral. Runtime TSC weights compose with these maps and are blended before
//! the resulting active-leaf weights are consumed by gather or transpose deposition.

void DustGasDrag::BuildSMRCompositeTransfer(ParameterInput *pin) {
  Mesh *pmesh = pmy_pack->pmesh;
  auto &indcs = pmesh->mb_indcs;
  auto &mbsize = pmy_pack->pmb->mb_size;
  auto &mblev = pmy_pack->pmb->mb_lev;
  int nmb = pmy_pack->nmb_thispack;
  int ncells1 = indcs.nx1 + 2*indcs.ng;
  int ncells2 = indcs.nx2 + 2*indcs.ng;

  if (pmesh->max_level - pmesh->root_level != 1) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "smooth_p2 currently requires exactly one physical "
              << "level of 2:1 static refinement" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  smr_transfer.enabled = true;
  smr_transfer.ncoarse1 = pmesh->mesh_indcs.nx1;
  smr_transfer.ncoarse2 = pmesh->mesh_indcs.nx2;
  smr_transfer.nfine1 = 2*smr_transfer.ncoarse1;
  smr_transfer.nfine2 = 2*smr_transfer.ncoarse2;
  smr_transfer.x1min = pmesh->mesh_size.x1min;
  smr_transfer.x1max = pmesh->mesh_size.x1max;
  smr_transfer.x2min = pmesh->mesh_size.x2min;
  smr_transfer.x2max = pmesh->mesh_size.x2max;
  Real band_cells = pin->GetOrAddReal(
      "dust", "experimental_smr_blend_width_cells", 2.0);
  if (!(band_cells > 0.0)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "<dust>/experimental_smr_blend_width_cells must be "
              << "positive" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  Real coarse_dx1 = (smr_transfer.x1max - smr_transfer.x1min)/
                    static_cast<Real>(smr_transfer.ncoarse1);
  Real coarse_dx2 = (smr_transfer.x2max - smr_transfer.x2min)/
                    static_cast<Real>(smr_transfer.ncoarse2);
  smr_transfer.transition_width = band_cells*std::max(coarse_dx1, coarse_dx2);
  smr_transfer.measure_correction = pin->GetOrAddBoolean(
      "dust", "experimental_smr_measure_correction", true);

  smr_transfer.coarse_count = DvceArray2D<int>(
      "dust_smr_coarse_count", smr_transfer.ncoarse2, smr_transfer.ncoarse1);
  smr_transfer.coarse_target = DvceArray3D<int>(
      "dust_smr_coarse_target", smr_transfer.ncoarse2, smr_transfer.ncoarse1,
      kMaxCoarseMapEntries);
  smr_transfer.coarse_weight = DvceArray3D<Real>(
      "dust_smr_coarse_weight", smr_transfer.ncoarse2, smr_transfer.ncoarse1,
      kMaxCoarseMapEntries);
  auto coarse_count = Kokkos::create_mirror_view(smr_transfer.coarse_count);
  auto coarse_target = Kokkos::create_mirror_view(smr_transfer.coarse_target);
  auto coarse_weight = Kokkos::create_mirror_view(smr_transfer.coarse_weight);
  Kokkos::deep_copy(coarse_count, 0);
  Kokkos::deep_copy(coarse_target, -1);
  Kokkos::deep_copy(coarse_weight, 0.0);

  auto append_coarse = [&](const int vj, const int vi, const int target,
                           const Real weight) {
    if (std::abs(weight) < 1.0e-15) return;
    int count = coarse_count(vj,vi);
    for (int entry=0; entry<count; ++entry) {
      if (coarse_target(vj,vi,entry) == target) {
        coarse_weight(vj,vi,entry) += weight;
        return;
      }
    }
    if (count >= kMaxCoarseMapEntries) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "virtual coarse cell maps to too many active leaves"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    coarse_target(vj,vi,count) = target;
    coarse_weight(vj,vi,count) = weight;
    coarse_count(vj,vi) = count + 1;
  };

  // Geometrically restrict every virtual root cell onto overlapping active leaves.
  for (int vj=0; vj<smr_transfer.ncoarse2; ++vj) {
    Real virtual_ylo = smr_transfer.x2min + vj*coarse_dx2;
    Real virtual_yhi = virtual_ylo + coarse_dx2;
    for (int vi=0; vi<smr_transfer.ncoarse1; ++vi) {
      Real virtual_xlo = smr_transfer.x1min + vi*coarse_dx1;
      Real virtual_xhi = virtual_xlo + coarse_dx1;
      Real virtual_volume = coarse_dx1*coarse_dx2;
      for (int m=0; m<nmb; ++m) {
        const auto &size = mbsize.h_view(m);
        if (virtual_xhi <= size.x1min || virtual_xlo >= size.x1max ||
            virtual_yhi <= size.x2min || virtual_ylo >= size.x2max) continue;
        for (int j=0; j<indcs.nx2; ++j) {
          Real cell_ylo = size.x2min + j*size.dx2;
          Real cell_yhi = cell_ylo + size.dx2;
          Real overlap2 = std::max(0.0, std::min(virtual_yhi,cell_yhi) -
                                        std::max(virtual_ylo,cell_ylo));
          if (overlap2 == 0.0) continue;
          for (int i=0; i<indcs.nx1; ++i) {
            Real cell_xlo = size.x1min + i*size.dx1;
            Real cell_xhi = cell_xlo + size.dx1;
            Real overlap1 = std::max(0.0, std::min(virtual_xhi,cell_xhi) -
                                          std::max(virtual_xlo,cell_xlo));
            Real overlap = overlap1*overlap2;
            if (overlap <= 1.0e-13*std::min(virtual_volume,size.dx1*size.dx2)) continue;
            int target = EncodeLeafCell(m, j + indcs.js, i + indcs.is,
                                        ncells2, ncells1);
            append_coarse(vj, vi, target, overlap/virtual_volume);
          }
        }
      }
    }
  }

  smr_transfer.fine_count = DvceArray2D<int>(
      "dust_smr_fine_count", smr_transfer.nfine2, smr_transfer.nfine1);
  smr_transfer.fine_target = DvceArray3D<int>(
      "dust_smr_fine_target", smr_transfer.nfine2, smr_transfer.nfine1,
      kMaxFineMapEntries);
  smr_transfer.fine_weight = DvceArray3D<Real>(
      "dust_smr_fine_weight", smr_transfer.nfine2, smr_transfer.nfine1,
      kMaxFineMapEntries);
  auto fine_count = Kokkos::create_mirror_view(smr_transfer.fine_count);
  auto fine_target = Kokkos::create_mirror_view(smr_transfer.fine_target);
  auto fine_weight = Kokkos::create_mirror_view(smr_transfer.fine_weight);
  Kokkos::deep_copy(fine_count, 0);
  Kokkos::deep_copy(fine_target, -1);
  Kokkos::deep_copy(fine_weight, 0.0);

  auto append_fine = [&](const int vj, const int vi, const int target,
                         const Real weight) {
    if (std::abs(weight) < 1.0e-15) return;
    int count = fine_count(vj,vi);
    for (int entry=0; entry<count; ++entry) {
      if (fine_target(vj,vi,entry) == target) {
        fine_weight(vj,vi,entry) += weight;
        return;
      }
    }
    if (count >= kMaxFineMapEntries) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "virtual fine cell maps to too many active leaves"
                << std::endl;
      std::exit(EXIT_FAILURE);
    }
    fine_target(vj,vi,count) = target;
    fine_weight(vj,vi,count) = weight;
    fine_count(vj,vi) = count + 1;
  };

  auto add_coarse_row_to_fine = [&](const int fine_j, const int fine_i,
                                    const int coarse_j, const int coarse_i,
                                    const Real factor) {
    int wrapped_j = SMRCompositeTransfer::WrapIndex(
        coarse_j, smr_transfer.ncoarse2);
    int wrapped_i = SMRCompositeTransfer::WrapIndex(
        coarse_i, smr_transfer.ncoarse1);
    int count = coarse_count(wrapped_j,wrapped_i);
    for (int entry=0; entry<count; ++entry) {
      append_fine(fine_j, fine_i, coarse_target(wrapped_j,wrapped_i,entry),
                  factor*coarse_weight(wrapped_j,wrapped_i,entry));
    }
  };

  Real fine_dx1 = 0.5*coarse_dx1;
  Real fine_dx2 = 0.5*coarse_dx2;
  for (int fj=0; fj<smr_transfer.nfine2; ++fj) {
    Real y = smr_transfer.x2min + (fj + 0.5)*fine_dx2;
    for (int fi=0; fi<smr_transfer.nfine1; ++fi) {
      Real x = smr_transfer.x1min + (fi + 0.5)*fine_dx1;
      int owner = -1;
      int owner_i = -1, owner_j = -1;
      for (int m=0; m<nmb && owner < 0; ++m) {
        const auto &size = mbsize.h_view(m);
        if (!(x >= size.x1min && x < size.x1max &&
              y >= size.x2min && y < size.x2max)) continue;
        owner = m;
        owner_i = std::min(indcs.nx1 - 1,
            static_cast<int>((x - size.x1min)/size.dx1));
        owner_j = std::min(indcs.nx2 - 1,
            static_cast<int>((y - size.x2min)/size.dx2));
      }
      if (owner < 0) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "virtual fine cell has no active leaf owner"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }

      if (mblev.h_view(owner) == pmesh->max_level) {
        append_fine(fj, fi,
            EncodeLeafCell(owner, owner_j + indcs.js, owner_i + indcs.is,
                           ncells2, ncells1), 1.0);
      } else {
        int parent_i = fi/2;
        int parent_j = fj/2;
        Real sign1 = ((fi % 2) == 0) ? -1.0 : 1.0;
        Real sign2 = ((fj % 2) == 0) ? -1.0 : 1.0;
        add_coarse_row_to_fine(fj,fi,parent_j,  parent_i, 1.0);
        add_coarse_row_to_fine(fj,fi,parent_j,  parent_i+1, sign1/8.0);
        add_coarse_row_to_fine(fj,fi,parent_j,  parent_i-1,-sign1/8.0);
        add_coarse_row_to_fine(fj,fi,parent_j+1,parent_i,   sign2/8.0);
        add_coarse_row_to_fine(fj,fi,parent_j-1,parent_i,  -sign2/8.0);
      }
    }
  }

  // Both virtual maps must preserve constants row-wise and physical volume column-wise.
  int encoded_size = nmb*ncells2*ncells1;
  std::vector<Real> coarse_integral(encoded_size, 0.0);
  std::vector<Real> fine_integral(encoded_size, 0.0);
  for (int j=0; j<smr_transfer.ncoarse2; ++j) {
    for (int i=0; i<smr_transfer.ncoarse1; ++i) {
      Real row_sum = 0.0;
      for (int entry=0; entry<coarse_count(j,i); ++entry) {
        int target = coarse_target(j,i,entry);
        Real weight = coarse_weight(j,i,entry);
        row_sum += weight;
        coarse_integral[target] += coarse_dx1*coarse_dx2*weight;
      }
      if (std::abs(row_sum - 1.0) > 2.0e-12) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "virtual coarse map does not preserve constants"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  for (int j=0; j<smr_transfer.nfine2; ++j) {
    for (int i=0; i<smr_transfer.nfine1; ++i) {
      Real row_sum = 0.0;
      for (int entry=0; entry<fine_count(j,i); ++entry) {
        int target = fine_target(j,i,entry);
        Real weight = fine_weight(j,i,entry);
        row_sum += weight;
        fine_integral[target] += fine_dx1*fine_dx2*weight;
      }
      if (std::abs(row_sum - 1.0) > 2.0e-12) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "virtual fine map does not preserve constants"
                  << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
  for (int m=0; m<nmb; ++m) {
    Real leaf_volume = mbsize.h_view(m).dx1*mbsize.h_view(m).dx2;
    for (int j=indcs.js; j<=indcs.je; ++j) {
      for (int i=indcs.is; i<=indcs.ie; ++i) {
        int target = EncodeLeafCell(m,j,i,ncells2,ncells1);
        Real coarse_error = std::abs(coarse_integral[target] - leaf_volume);
        Real fine_error = std::abs(fine_integral[target] - leaf_volume);
        if (coarse_error > 2.0e-11*leaf_volume ||
            fine_error > 2.0e-11*leaf_volume) {
          std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                    << std::endl << "virtual SMR map does not preserve leaf volume"
                    << std::endl;
          std::exit(EXIT_FAILURE);
        }
      }
    }
  }

  int fine_blocks = 0;
  for (int m=0; m<nmb; ++m) {
    if (mblev.h_view(m) == pmesh->max_level) ++fine_blocks;
  }
  smr_transfer.nfine_blocks = fine_blocks;
  smr_transfer.fine_block_bounds = DvceArray2D<Real>(
      "dust_smr_fine_bounds", std::max(1,fine_blocks), 4);
  auto fine_bounds = Kokkos::create_mirror_view(smr_transfer.fine_block_bounds);
  int block = 0;
  for (int m=0; m<nmb; ++m) {
    if (mblev.h_view(m) != pmesh->max_level) continue;
    fine_bounds(block,0) = mbsize.h_view(m).x1min;
    fine_bounds(block,1) = mbsize.h_view(m).x1max;
    fine_bounds(block,2) = mbsize.h_view(m).x2min;
    fine_bounds(block,3) = mbsize.h_view(m).x2max;
    ++block;
  }

  // The first integrated target is deliberately narrower than arbitrary static-SMR:
  // one rectangular fine strip spanning x2, with two periodic x1 interfaces. This is
  // the half-domain linA layout. Keeping this fail-closed avoids silently applying the
  // correction profiles to corners or disconnected patches they do not yet represent.
  Real patch_xmin = smr_transfer.x1max;
  Real patch_xmax = smr_transfer.x1min;
  Real patch_ymin = smr_transfer.x2max;
  Real patch_ymax = smr_transfer.x2min;
  Real fine_block_area = 0.0;
  for (int entry=0; entry<fine_blocks; ++entry) {
    patch_xmin = std::min(patch_xmin, fine_bounds(entry,0));
    patch_xmax = std::max(patch_xmax, fine_bounds(entry,1));
    patch_ymin = std::min(patch_ymin, fine_bounds(entry,2));
    patch_ymax = std::max(patch_ymax, fine_bounds(entry,3));
    fine_block_area += (fine_bounds(entry,1) - fine_bounds(entry,0))*
                       (fine_bounds(entry,3) - fine_bounds(entry,2));
  }
  Real mesh_area = (smr_transfer.x1max - smr_transfer.x1min)*
                   (smr_transfer.x2max - smr_transfer.x2min);
  Real patch_area = (patch_xmax - patch_xmin)*(patch_ymax - patch_ymin);
  if (fine_blocks == 0 ||
      std::abs(patch_ymin - smr_transfer.x2min) > 1.0e-12*std::sqrt(mesh_area) ||
      std::abs(patch_ymax - smr_transfer.x2max) > 1.0e-12*std::sqrt(mesh_area) ||
      std::abs(fine_block_area - patch_area) > 1.0e-12*mesh_area ||
      !(patch_xmax > patch_xmin) ||
      !((patch_xmax - patch_xmin) < (smr_transfer.x1max - smr_transfer.x1min))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "smooth_p2 currently requires one rectangular static "
              << "fine strip spanning the full x2 domain" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  smr_transfer.interface_x1[0] = patch_xmin;
  smr_transfer.interface_x1[1] = patch_xmax;
  smr_transfer.interface_coarse_sign[0] = -1.0;
  smr_transfer.interface_coarse_sign[1] = 1.0;

  // Integrate the column defect of the raw position-dependent blend with four-point
  // Gauss quadrature in every virtual fine cell. All breakpoints of TSC, the aligned
  // patch, and the 2H transition band coincide with virtual-cell faces, so this rule
  // exactly integrates the degree-seven blend/TSC products up to roundoff.
  smr_transfer.correction_tiles = 2*smr_transfer.ncoarse2;
  std::vector<Real> correction_dense(
      static_cast<std::size_t>(smr_transfer.correction_tiles)*encoded_size, 0.0);
  const Real gauss_node[4] = {
      -0.8611363115940525752, -0.3399810435848562648,
       0.3399810435848562648,  0.8611363115940525752};
  const Real gauss_weight[4] = {
      0.3478548451374538574, 0.6521451548625461426,
      0.6521451548625461426, 0.3478548451374538574};
  Real domain_length1 = smr_transfer.x1max - smr_transfer.x1min;

  auto periodic_offset = [](Real value, const Real period) {
    while (value < -0.5*period) value += period;
    while (value >= 0.5*period) value -= period;
    return value;
  };
  auto host_fine_fraction = [&](const Real x) {
    Real left_offset = periodic_offset(x - patch_xmin, domain_length1);
    Real right_offset = periodic_offset(x - patch_xmax, domain_length1);
    bool inside = (x >= patch_xmin && x <= patch_xmax);
    if (inside) return 1.0;
    Real distance = std::min(std::abs(left_offset), std::abs(right_offset));
    Real s = std::max(0.0, 1.0 - distance/smr_transfer.transition_width);
    return s*s*s*(10.0 + s*(-15.0 + 6.0*s));
  };
  auto add_virtual_row = [&](const bool fine_map, const int vj, const int vi,
                             const int tile, const Real factor) {
    int count = fine_map ? fine_count(vj,vi) : coarse_count(vj,vi);
    for (int entry=0; entry<count; ++entry) {
      int target = fine_map ? fine_target(vj,vi,entry)
                            : coarse_target(vj,vi,entry);
      Real map_weight = fine_map ? fine_weight(vj,vi,entry)
                                 : coarse_weight(vj,vi,entry);
      correction_dense[static_cast<std::size_t>(tile)*encoded_size + target] +=
          factor*map_weight;
    }
  };

  for (int cell_j=0; cell_j<smr_transfer.nfine2; ++cell_j) {
    Real cell_ycenter = smr_transfer.x2min + (cell_j + 0.5)*fine_dx2;
    for (int cell_i=0; cell_i<smr_transfer.nfine1; ++cell_i) {
      Real cell_xcenter = smr_transfer.x1min + (cell_i + 0.5)*fine_dx1;
      for (int qj=0; qj<4; ++qj) {
        Real y = cell_ycenter + 0.5*fine_dx2*gauss_node[qj];
        int tile_y = std::min(smr_transfer.ncoarse2 - 1,
            static_cast<int>((y - smr_transfer.x2min)/coarse_dx2));
        for (int qi=0; qi<4; ++qi) {
          Real x = cell_xcenter + 0.5*fine_dx1*gauss_node[qi];
          Real distance0 = std::abs(periodic_offset(
              x - smr_transfer.interface_x1[0], domain_length1));
          Real distance1 = std::abs(periodic_offset(
              x - smr_transfer.interface_x1[1], domain_length1));
          int interface = (distance0 <= distance1) ? 0 : 1;
          int tile = interface*smr_transfer.ncoarse2 + tile_y;
          Real alpha = host_fine_fraction(x);
          Real quadrature_weight = 0.25*fine_dx1*fine_dx2*
                                   gauss_weight[qi]*gauss_weight[qj];
          int coarse_i[3], coarse_j[3], fine_i[3], fine_j[3];
          Real coarse_w1[3], coarse_w2[3], fine_w1[3], fine_w2[3];
          SMRCompositeTransfer::VirtualTSC(
              x,smr_transfer.x1min,smr_transfer.x1max,smr_transfer.ncoarse1,
              coarse_i,coarse_w1);
          SMRCompositeTransfer::VirtualTSC(
              y,smr_transfer.x2min,smr_transfer.x2max,smr_transfer.ncoarse2,
              coarse_j,coarse_w2);
          SMRCompositeTransfer::VirtualTSC(
              x,smr_transfer.x1min,smr_transfer.x1max,smr_transfer.nfine1,
              fine_i,fine_w1);
          SMRCompositeTransfer::VirtualTSC(
              y,smr_transfer.x2min,smr_transfer.x2max,smr_transfer.nfine2,
              fine_j,fine_w2);
          for (int b=0; b<3; ++b) {
            for (int a=0; a<3; ++a) {
              add_virtual_row(false,coarse_j[b],coarse_i[a],tile,
                  -quadrature_weight*alpha*coarse_w2[b]*coarse_w1[a]);
              add_virtual_row(true,fine_j[b],fine_i[a],tile,
                  quadrature_weight*alpha*fine_w2[b]*fine_w1[a]);
            }
          }
        }
      }
    }
  }

  std::vector<Real> target_x1(encoded_size, 0.0);
  std::vector<Real> target_x2(encoded_size, 0.0);
  std::vector<Real> target_volume(encoded_size, 0.0);
  for (int m=0; m<nmb; ++m) {
    const auto &size = mbsize.h_view(m);
    for (int j=indcs.js; j<=indcs.je; ++j) {
      for (int i=indcs.is; i<=indcs.ie; ++i) {
        int target = EncodeLeafCell(m,j,i,ncells2,ncells1);
        target_x1[target] = size.x1min + (i - indcs.is + 0.5)*size.dx1;
        target_x2[target] = size.x2min + (j - indcs.js + 0.5)*size.dx2;
        target_volume[target] = size.dx1*size.dx2;
      }
    }
  }

  Real correction_tolerance = 2.0e-13*coarse_dx1*coarse_dx2;
  int maximum_correction_entries = 0;
  std::vector<int> correction_counts(smr_transfer.correction_tiles, 0);
  std::vector<Real> total_defect(encoded_size, 0.0);
  for (int tile=0; tile<smr_transfer.correction_tiles; ++tile) {
    int interface = tile/smr_transfer.ncoarse2;
    int tile_y = tile % smr_transfer.ncoarse2;
    Real tile_ycenter = smr_transfer.x2min + (tile_y + 0.5)*coarse_dx2;
    Real sum = 0.0, moment1 = 0.0, moment2 = 0.0;
    for (int target=0; target<encoded_size; ++target) {
      Real defect = correction_dense[static_cast<std::size_t>(tile)*encoded_size +
                                     target];
      total_defect[target] += defect;
      sum += defect;
      moment1 += defect*periodic_offset(
          target_x1[target] - smr_transfer.interface_x1[interface], domain_length1);
      moment2 += defect*periodic_offset(
          target_x2[target] - tile_ycenter, smr_transfer.x2max-smr_transfer.x2min);
      if (std::abs(defect) > correction_tolerance) ++correction_counts[tile];
    }
    maximum_correction_entries = std::max(
        maximum_correction_entries, correction_counts[tile]);
    if (std::abs(sum) > 5.0e-11*coarse_dx1*coarse_dx2 ||
        std::abs(moment1) > 5.0e-11*coarse_dx1*coarse_dx2*coarse_dx1 ||
        std::abs(moment2) > 5.0e-11*coarse_dx1*coarse_dx2*coarse_dx2) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                << std::endl << "smooth-P2 correction tile violates a zeroth or first "
                << "moment constraint" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  smr_transfer.correction_count = DvceArray1D<int>(
      "dust_smr_correction_count", smr_transfer.correction_tiles);
  smr_transfer.correction_target = DvceArray2D<int>(
      "dust_smr_correction_target", smr_transfer.correction_tiles,
      std::max(1,maximum_correction_entries));
  smr_transfer.correction_weight = DvceArray2D<Real>(
      "dust_smr_correction_weight", smr_transfer.correction_tiles,
      std::max(1,maximum_correction_entries));
  auto correction_count = Kokkos::create_mirror_view(smr_transfer.correction_count);
  auto correction_target = Kokkos::create_mirror_view(smr_transfer.correction_target);
  auto correction_weight = Kokkos::create_mirror_view(smr_transfer.correction_weight);
  Kokkos::deep_copy(correction_count, 0);
  Kokkos::deep_copy(correction_target, -1);
  Kokkos::deep_copy(correction_weight, 0.0);
  Real maximum_relative_defect = 0.0;
  Real maximum_corrected_relative_defect = 0.0;
  for (int target=0; target<encoded_size; ++target) {
    if (target_volume[target] > 0.0) {
      maximum_relative_defect = std::max(maximum_relative_defect,
          std::abs(total_defect[target])/target_volume[target]);
    }
  }
  for (int tile=0; tile<smr_transfer.correction_tiles; ++tile) {
    int entry = 0;
    for (int target=0; target<encoded_size; ++target) {
      Real defect = correction_dense[static_cast<std::size_t>(tile)*encoded_size +
                                     target];
      if (std::abs(defect) <= correction_tolerance) continue;
      correction_target(tile,entry) = target;
      correction_weight(tile,entry) = defect;
      ++entry;
    }
    correction_count(tile) = entry;
  }

  // The bump profile has unit integral, so the corrected column defect is the raw
  // defect minus the sparse tile entries retained above. Audit the truncation rather
  // than relying only on the analytical normalization of the profile.
  for (int target=0; target<encoded_size; ++target) {
    if (target_volume[target] <= 0.0) continue;
    Real corrected_defect = total_defect[target];
    for (int tile=0; tile<smr_transfer.correction_tiles; ++tile) {
      for (int entry=0; entry<correction_count(tile); ++entry) {
        if (correction_target(tile,entry) == target) {
          corrected_defect -= correction_weight(tile,entry);
        }
      }
    }
    maximum_corrected_relative_defect = std::max(
        maximum_corrected_relative_defect,
        std::abs(corrected_defect)/target_volume[target]);
  }
  if (smr_transfer.measure_correction && maximum_corrected_relative_defect > 2.0e-10) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "smooth-P2 measure correction leaves a relative "
              << "uniform-loading defect of " << maximum_corrected_relative_defect
              << std::endl;
    std::exit(EXIT_FAILURE);
  }

  if (global_variable::my_rank == 0) {
    std::cout << "# smooth_p2 map: blend width=" << band_cells
              << " coarse cells, measure correction="
              << (smr_transfer.measure_correction ? "on" : "off")
              << ", correction entries/tile<="
              << maximum_correction_entries << ", raw uniform-loading defect="
              << maximum_relative_defect << ", corrected defect="
              << maximum_corrected_relative_defect << std::endl;
  }

  Kokkos::deep_copy(smr_transfer.coarse_count, coarse_count);
  Kokkos::deep_copy(smr_transfer.coarse_target, coarse_target);
  Kokkos::deep_copy(smr_transfer.coarse_weight, coarse_weight);
  Kokkos::deep_copy(smr_transfer.fine_count, fine_count);
  Kokkos::deep_copy(smr_transfer.fine_target, fine_target);
  Kokkos::deep_copy(smr_transfer.fine_weight, fine_weight);
  Kokkos::deep_copy(smr_transfer.fine_block_bounds, fine_bounds);
  Kokkos::deep_copy(smr_transfer.correction_count, correction_count);
  Kokkos::deep_copy(smr_transfer.correction_target, correction_target);
  Kokkos::deep_copy(smr_transfer.correction_weight, correction_weight);
}

} // namespace dust
