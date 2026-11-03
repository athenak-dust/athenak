//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_part.cpp
//! \brief

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
#include <algorithm>
#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"
#include "mesh/nghbr_index.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "bvals.hpp"
#include "dust/dust.hpp"

namespace particles {
namespace {
// ATHENA_PART_SENDLIST_OLD=1: the send list is reallocated to exactly the particle count and
// cleared (host and device) whenever that count passes its high-water mark (the former path)
bool SendlistOld() {
  static const bool on = [] {
    const char *e = std::getenv("ATHENA_PART_SENDLIST_OLD");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}
#if MPI_PARALLEL_ENABLED
// Post the count elements at buf (a message of the particle exchange) as consecutive MPI
// messages of at most chunk elements each (MPI counts are int), all with the same tag: MPI's
// non-overtaking order matches them to the receives that the other rank posts in the same
// order with the same sizes.  A message of at most chunk elements is one MPI message, as
// before.  Returns the number of MPI messages.
template <typename T>
int PostPrtclMessage(const bool send, T *buf, const std::size_t count, const std::size_t chunk,
                     MPI_Datatype type, const int peer, const int tag, MPI_Comm comm,
                     std::vector<MPI_Request> &req, bool &no_errors) {
  std::size_t done = 0;
  int nmpi = 0;
  do {
    const std::size_t n = std::min(chunk, count - done);
    req.push_back(MPI_REQUEST_NULL);
    const int ierr = send ?
        MPI_Isend(buf + done, static_cast<int>(n), type, peer, tag, comm, &req.back()) :
        MPI_Irecv(buf + done, static_cast<int>(n), type, peer, tag, comm, &req.back());
    if (ierr != MPI_SUCCESS) {no_errors = false;}
    done += n;
    ++nmpi;
  } while (done < count);
  return nmpi;
}
#endif

// Capacity of a particle message buffer (send or receive, reals or ints) for `required`
// values: half again above the new high-water mark, at least 4096 particles' worth.  The exact
// size reallocated a buffer whenever a message set a new maximum
// (with 32 ranks and four buffers each, about once per cycle on some rank after 100 cycles);
// a new device buffer must be registered with the network/IPC layers at its first message,
// which held the receiver for 5-30 ms while all others waited in the MPI_Allgather that ends
// RecvAndUnpackPrtcls.  Only valid prefixes are packed and passed
// to MPI, so the capacity does not change any result.
template <typename View>
void ReserveMessageBuffer(View &buf, const std::size_t required, const int nvals) {
  if (buf.extent(0) >= required) return;
  const std::size_t cap = std::max(required + required/2, static_cast<std::size_t>(4096)*nvals);
  Kokkos::realloc(buf, cap);
}
}  // namespace

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ReserveSendlist(int npart)
//! \brief capacity of the send list for npart particles.  The capacity grows to an eighth above
//! the new high-water mark and is not cleared (only the valid prefix is ever written and read):
//! with the exact size, a rank whose particle count keeps growing (splits, inflow) reallocated
//! and cleared the whole list, 12 bytes per particle on the host and on the device, at nearly
//! every migration (on a deep adaptive run almost every cycle, a delay that the other ranks
//! waited for).  ATHENA_PART_SENDLIST_OLD=1 restores the exact size with
//! initialization.

void ParticlesBoundaryValues::ReserveSendlist(int npart) {
  const int required_capacity = std::max(npart, 1);
  if (sendlist.extent_int(0) >= required_capacity) return;
  if (SendlistOld()) {
    Kokkos::realloc(sendlist, required_capacity);
  } else {
    const std::size_t cap = std::min<std::size_t>(
        static_cast<std::size_t>(required_capacity) + required_capacity/8,
        static_cast<std::size_t>(std::numeric_limits<int>::max()));
    Kokkos::realloc(Kokkos::WithoutInitializing, sendlist, cap);
  }
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::UpdateGID()
//! \brief Updates GID of particles that cross boundary of their parent MeshBlock.  If
//! the new GID is on a different rank, then store in sendlist_buf DvceArray: (1) index of
//! particle in prtcl array, (2) destination GID, and (3) destination rank.

KOKKOS_INLINE_FUNCTION
void UpdateGID(int &newgid, NeighborBlock nghbr, int myrank, int *pcounter,
               DvceArray1D<ParticleLocationData> slist, int p) {
  newgid = nghbr.gid;
#if MPI_PARALLEL_ENABLED
  if (nghbr.rank != myrank) {
    int index = Kokkos::atomic_fetch_add(pcounter,1);
    slist(index).prtcl_indx = p;
    slist(index).dest_gid   = nghbr.gid;
    slist(index).dest_rank  = nghbr.rank;
  }
#endif
  return;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::SetNewGID()
//! \brief

TaskStatus ParticlesBoundaryValues::SetNewPrtclGID() {
  // create local references for variables in kernel
  auto gids = pmy_part->pmy_pack->gids;
  auto &pr = pmy_part->prtcl_rdata;
  auto &pi = pmy_part->prtcl_idata;
  int npart = pmy_part->nprtcl_thispack;
  auto &mbsize = pmy_part->pmy_pack->pmb->mb_size;
  auto &mblev = pmy_part->pmy_pack->pmb->mb_lev;
  auto &meshsize = pmy_part->pmy_pack->pmesh->mesh_size;
  auto myrank = global_variable::my_rank;
  auto &nghbr = pmy_part->pmy_pack->pmb->nghbr;
  int *pcounter = nullptr;
  bool &multi_d = pmy_part->pmy_pack->pmesh->multi_d;
  bool &three_d = pmy_part->pmy_pack->pmesh->three_d;
  // dust particles carry RK position registers (IPX1/IPY1/IPZ1) that must be shifted by
  // the same box length as the positions when a particle wraps at a periodic boundary
  const bool has_reg = (pmy_part->particle_type == ParticleType::dust);

  // shear-periodic x1 boundaries: particles crossing the radial mesh boundaries are
  // shifted azimuthally by the (folded) shear offset (positions only; velocities are
  // shear-relative), and routed to the boundary MeshBlock found from the shear maps
  const bool shear_x1 = shear_periodic_x1;
  Real ysh = 0.0;
  if (shear_x1) {
    Real lx = (meshsize.x1max - meshsize.x1min);
    Real ly = (meshsize.x2max - meshsize.x2min);
    ysh = fmod(qshear_sp*omega0_sp*lx*(pmy_part->pmy_pack->pmesh->time), ly);
  }
  const int nmbx2 = nmb_sp_x2, nmbx3 = nmb_sp_x3;
  auto &sgid = sgid_map;
  auto &srnk = srank_map;

#if MPI_PARALLEL_ENABLED
  // Keep a worst-case-capacity send list, but allocate only when the local particle
  // high-water mark grows. nprtcl_send below is the logical valid-prefix length.  The
  // persistent allocation avoids the previous grow-to-npart/shrink-to-nsend churn.
  ReserveSendlist(npart);
  Kokkos::deep_copy(send_count.d_view, 0);
  pcounter = send_count.d_view.data();
#endif
  // Capture only the CUDA-accessible view in the kernel. Capturing the DualView (or
  // accessing the sendlist member through `this`) leaves a host pointer in a CUDA
  // lambda and fails as soon as an inter-rank particle is appended or packed.
  auto psendl = sendlist.d_view;
  // the physical x3 faces: reflect or outflow (see bvals.hpp)
  const int x3bc = x3bc_dust;
  constexpr int kLobe3Shift = SMRCompositeTransfer::kCloudLobeShift3;   // dust.hpp
  auto x3rem = x3_removed.d_view;
  const Real z3a = meshsize.x3min, z3b = meshsize.x3max;
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    int m = pi(PGID,p) - gids;
    int mylevel = mblev.d_view(m);
    if (x3bc != 0 && (pr(IPZ,p) < z3a || pr(IPZ,p) >= z3b)) {
      const bool lo = (pr(IPZ,p) < z3a);
      if (x3bc == 1) {   // reflect: mirror the position (and the RK register) and the velocity
        const Real zf = lo ? z3a : z3b;
        Real z = 2.0*zf - pr(IPZ,p);
        if (z >= z3b) z = z3b - 1.0e-12*(z3b - z3a);
        if (z < z3a) z = z3a;
        pr(IPZ,p) = z;
        pr(IPVZ,p) = -pr(IPVZ,p);
        if (has_reg) {
          pr(IPZ1,p) = 2.0*zf - pr(IPZ1,p); pr(IPVZ1,p) = -pr(IPVZ1,p);
          // a refinable cloud's x3 lobe mirrors too: lobe 1 <-> 2 (bits
          // 9-10 of PLEV, dust::SMRCompositeTransfer::kCloudLobeShift3; 0 and 3 are symmetric;
          // point particles have no lobe bits)
          const int l3 = (pi(PLEV,p) >> kLobe3Shift) & 3;
          if (l3 == 1 || l3 == 2) pi(PLEV,p) ^= (3 << kLobe3Shift);
        }
      } else {           // outflow: record the mass, park the (now massless) particle at rest
        Kokkos::atomic_add(&x3rem(0), pr(IPM,p));
        Kokkos::atomic_add(&x3rem(1), 1.0);
        pr(IPM,p) = 0.0;
        const Real dz = mbsize.d_view(m).dx3;
        const Real z = lo ? (mbsize.d_view(m).x3min + 0.5*dz) : (mbsize.d_view(m).x3max - 0.5*dz);
        pr(IPZ,p) = z;
        pr(IPVX,p) = 0.0; pr(IPVY,p) = 0.0; pr(IPVZ,p) = 0.0;
        if (has_reg) {
          pr(IPX1,p) = pr(IPX,p); pr(IPY1,p) = pr(IPY,p); pr(IPZ1,p) = z;
          pr(IPVX1,p) = 0.0; pr(IPVY1,p) = 0.0; pr(IPVZ1,p) = 0.0;
        }
      }
    }
    Real x1 = pr(IPX,p);
    Real x2 = pr(IPY,p);
    Real x3 = pr(IPZ,p);

    // length of MeshBlock in each direction
    Real lx = (mbsize.d_view(m).x1max - mbsize.d_view(m).x1min);
    Real ly = (mbsize.d_view(m).x2max - mbsize.d_view(m).x2min);
    Real lz = (mbsize.d_view(m).x3max - mbsize.d_view(m).x3min);

    // integer offset of particle relative to center of MeshBlock (-1,0,+1)
    int ix = static_cast<int>((x1 - mbsize.d_view(m).x1min + lx)/lx) - 1;
    int iy = static_cast<int>((x2 - mbsize.d_view(m).x2min + ly)/ly) - 1;
    int iz = static_cast<int>((x3 - mbsize.d_view(m).x3min + lz)/lz) - 1;

    // A particle the offset test keeps in its block (offset 0) may
    // still lie outside the block's half-open box by round-off: just below a face at coordinate 0
    // (|x| far below the precision of x - xmin + l), or at xmax after a rounded x - xmin.  Put it
    // onto the box (the largest value below xmax at the upper face), so that every particle lies
    // in [xmin, xmax) of its block; the RK register moves with the position.
    auto snap = [&](const int off, Real &x, const Real xa, const Real xb, const int ip,
                    const int ir) {
      if (off != 0) return;
      Real xs = x;
      if (x < xa) {
        xs = xa;
      } else if (x >= xb) {
        xs = Kokkos::nextafter(xb, xa);
      }
      if (xs != x) {
        if (has_reg) {pr(ir,p) += xs - x;}
        pr(ip,p) = xs;
        x = xs;
      }
    };
    snap(ix, x1, mbsize.d_view(m).x1min, mbsize.d_view(m).x1max, IPX, IPX1);
    if (multi_d) {snap(iy, x2, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max, IPY, IPY1);}
    if (three_d) {snap(iz, x3, mbsize.d_view(m).x3min, mbsize.d_view(m).x3max, IPZ, IPZ1);}

    // sublock indices for faces and edges with S/AMR
    int fx = (x1 < 0.5*(mbsize.d_view(m).x1min + mbsize.d_view(m).x1max))? 0 : 1;
    int fy = (x2 < 0.5*(mbsize.d_view(m).x2min + mbsize.d_view(m).x2max))? 0 : 1;
    int fz = (x3 < 0.5*(mbsize.d_view(m).x3min + mbsize.d_view(m).x3max))? 0 : 1;
    fy = multi_d ? fy : 0;
    fz = three_d ? fz : 0;

    // a particle moves less than one cell per step, so it can only have left for an adjacent
    // MeshBlock; anything else means it was routed to a wrong MeshBlock before
    if (abs(ix) > 1 || abs(iy) > 1 || abs(iz) > 1) {
      Kokkos::printf("Particle routing: p=%d gid=%d x=(%.8e %.8e %.8e) lies more than one "
                     "MeshBlock away from its MeshBlock x1=[%.6e,%.6e] x2=[%.6e,%.6e] "
                     "x3=[%.6e,%.6e]\n", p, pi(PGID,p), x1, x2, x3, mbsize.d_view(m).x1min,
                     mbsize.d_view(m).x1max, mbsize.d_view(m).x2min, mbsize.d_view(m).x2max,
                     mbsize.d_view(m).x3min, mbsize.d_view(m).x3max);
      Kokkos::abort("particle outside the MeshBlocks adjacent to its own");
    }

    // only update particle GID if it has crossed MeshBlock boundary
    if ((abs(ix) + abs(iy) + abs(iz)) != 0) {
      // The particle crossed a Mesh boundary when its block offset points out of
      // the Mesh (a block on that edge), the same rounding as the destination block above.  A
      // position test (x < xmin) disagrees with it within round-off of the boundary: a particle
      // at y = -1e-18 has iy = 0 (it stays in its row), but crossing in x only it was wrapped to
      // y = ymax while routed to the neighbor in its own row (the next step then aborted)
      const bool lo1 = (ix < 0) && (mbsize.d_view(m).x1min < meshsize.x1min + 0.5*lx);
      const bool hi1 = (ix > 0) && (mbsize.d_view(m).x1max > meshsize.x1max - 0.5*lx);
      const bool lo2 = (iy < 0) && (mbsize.d_view(m).x2min < meshsize.x2min + 0.5*ly);
      const bool hi2 = (iy > 0) && (mbsize.d_view(m).x2max > meshsize.x2max - 0.5*ly);
      const bool lo3 = (iz < 0) && (mbsize.d_view(m).x3min < meshsize.x3min + 0.5*lz);
      const bool hi3 = (iz > 0) && (mbsize.d_view(m).x3max > meshsize.x3max - 0.5*lz);
      bool cross_in  = shear_x1 && lo1;
      bool cross_out = shear_x1 && hi1;
      if (cross_in || cross_out) {
        // shear-periodic radial crossing: wrap x, shift y azimuthally by the (folded)
        // shear offset with no velocity change (velocities are shear-relative), wrap
        // y/z periodically, then route to the boundary MeshBlock at the new (y,z).
        // The RK position registers are shifted identically.
        Real lxm = (meshsize.x1max - meshsize.x1min);
        pr(IPX,p) += cross_in ? lxm : -lxm;
        if (has_reg) {pr(IPX1,p) += cross_in ? lxm : -lxm;}
        Real lym = (meshsize.x2max - meshsize.x2min);
        Real ynew = pr(IPY,p) + (cross_in ? -ysh : ysh);
        if (ynew < meshsize.x2min) {ynew += lym;}
        if (ynew >= meshsize.x2max) {ynew -= lym;}
        Real dy = ynew - pr(IPY,p);
        pr(IPY,p) = ynew;
        if (has_reg) {pr(IPY1,p) += dy;}
        Real lzm = (meshsize.x3max - meshsize.x3min);
        if (lo3) {
          pr(IPZ,p) += lzm;
          if (has_reg) {pr(IPZ1,p) += lzm;}
        } else if (hi3) {
          pr(IPZ,p) -= lzm;
          if (has_reg) {pr(IPZ1,p) -= lzm;}
        }
        // destination MeshBlock on the opposite radial side at the new (y,z)
        int b2 = static_cast<int>(((pr(IPY,p) - meshsize.x2min)/lym)
                                  *static_cast<Real>(nmbx2));
        b2 = (b2 < 0) ? 0 : ((b2 > nmbx2-1) ? (nmbx2-1) : b2);
        int b3 = 0;
        if (three_d) {
          b3 = static_cast<int>(((pr(IPZ,p) - meshsize.x3min)/lzm)
                                *static_cast<Real>(nmbx3));
          b3 = (b3 < 0) ? 0 : ((b3 > nmbx3-1) ? (nmbx3-1) : b3);
        }
        NeighborBlock nb;
        nb.gid = sgid.d_view((cross_in ? 1 : 0), b3, b2);
        nb.lev = mylevel;
        nb.rank = srnk.d_view((cross_in ? 1 : 0), b3, b2);
        nb.dest = 0;
        UpdateGID(pi(PGID,p), nb, myrank, pcounter, psendl, p);
      } else {
      // first non-empty slot of direction (jx,jy,jz) from slot k on; the search stays among the
      // subslots of that direction (faces 4, edges 2, corners 1) and returns -1 if all are empty
      auto filled = [&](int jx, int jy, int jz, int k) {
        int nd = abs(jx) + abs(jy) + abs(jz);
        int end = NeighborIndex(jx,jy,jz,0,0) + ((nd == 1) ? 4 : ((nd == 2) ? 2 : 1));
        for (; k < end; ++k) {
          if (nghbr.d_view(m,k).gid >= 0) {return k;}
        }
        return -1;
      };
      // An empty edge or corner slot: the region is covered by a coarser MeshBlock
      // that is stored only once, under the lower-dimensional direction from which it is
      // reached (e.g. the x1 face neighbour also covers the x1x2 edge when the x2 crossing
      // stays inside this MeshBlock's parent).  That direction keeps the components of the
      // crossing that leave the parent; the parity of the logical location tells which half
      // of its parent this MeshBlock is.
      auto coarse_cover = [&](int jx, int jy, int jz) {
        int l1 = static_cast<int>(floor((mbsize.d_view(m).x1min - meshsize.x1min)/lx + 0.5));
        int l2 = static_cast<int>(floor((mbsize.d_view(m).x2min - meshsize.x2min)/ly + 0.5));
        int l3 = static_cast<int>(floor((mbsize.d_view(m).x3min - meshsize.x3min)/lz + 0.5));
        int kx = (jx == 0 || ((l1 & 1) == ((jx > 0) ? 1 : 0))) ? jx : 0;
        int ky = (jy == 0 || ((l2 & 1) == ((jy > 0) ? 1 : 0))) ? jy : 0;
        int kz = (jz == 0 || ((l3 & 1) == ((jz > 0) ? 1 : 0))) ? jz : 0;
        if ((abs(kx) + abs(ky) + abs(kz)) == 0 || (kx == jx && ky == jy && kz == jz)) {
          return -1;
        }
        return filled(kx, ky, kz, NeighborIndex(kx,ky,kz,0,0));
      };
      int indx = -1;
      if (iz == 0) {
        if (iy == 0) {
          // x1 face
          indx = NeighborIndex(ix,0,0,0,0);               // neighbor at same level
          if (nghbr.d_view(m,indx).lev > mylevel) {       // neighbor at finer level
            indx = NeighborIndex(ix,0,0,fy,fz);
          }
          indx = filled(ix,0,0,indx);                     // neighbor at coarser level
        } else if (ix == 0) {
          // x2 face
          indx = NeighborIndex(0,iy,0,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(0,iy,0,fx,fz);
          }
          indx = filled(0,iy,0,indx);
        } else {
          // x1x2 edge
          indx = NeighborIndex(ix,iy,0,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(ix,iy,0,fz,0);
          }
          indx = filled(ix,iy,0,indx);
          if (indx < 0) {indx = coarse_cover(ix,iy,0);}
        }
      } else if (iy == 0) {
        if (ix == 0) {
          // x3 face
          indx = NeighborIndex(0,0,iz,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(0,0,iz,fx,fy);
          }
          indx = filled(0,0,iz,indx);
        } else {
          // x3x1 edge
          indx = NeighborIndex(ix,0,iz,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(ix,0,iz,fy,0);
          }
          indx = filled(ix,0,iz,indx);
          if (indx < 0) {indx = coarse_cover(ix,0,iz);}
        }
      } else {
        if (ix == 0) {
          // x2x3 edge
          indx = NeighborIndex(0,iy,iz,0,0);
          if (nghbr.d_view(m,indx).lev > mylevel) {
            indx = NeighborIndex(0,iy,iz,fx,0);
          }
          indx = filled(0,iy,iz,indx);
          if (indx < 0) {indx = coarse_cover(0,iy,iz);}
        } else {
          // corners
          indx = filled(ix,iy,iz,NeighborIndex(ix,iy,iz,0,0));
          if (indx < 0) {indx = coarse_cover(ix,iy,iz);}
        }
      }
      if (indx < 0) {
        Kokkos::printf("Particle routing: p=%d gid=%d x=(%.8e %.8e %.8e) crossed in direction "
                       "(%d %d %d) and no neighbor MeshBlock covers it\n", p, pi(PGID,p),
                       x1, x2, x3, ix, iy, iz);
        Kokkos::abort("no neighbor MeshBlock for a particle");
      }
      UpdateGID(pi(PGID,p), nghbr.d_view(m,indx), myrank, pcounter, psendl, p);

      // reset x,y,z positions if particle crosses Mesh boundary using periodic BCs (decided by
      // the block offsets, lo1/hi1 etc. above, as the destination block)
      // RK position registers must be shifted with the position so the low-storage
      // combination (gam0*x + gam1*x1) stays in a single periodic image
      if (lo1) {
        pr(IPX,p) += (meshsize.x1max - meshsize.x1min);
        if (has_reg) {pr(IPX1,p) += (meshsize.x1max - meshsize.x1min);}
      } else if (hi1) {
        pr(IPX,p) -= (meshsize.x1max - meshsize.x1min);
        if (has_reg) {pr(IPX1,p) -= (meshsize.x1max - meshsize.x1min);}
      }
      if (lo2) {
        pr(IPY,p) += (meshsize.x2max - meshsize.x2min);
        if (has_reg) {pr(IPY1,p) += (meshsize.x2max - meshsize.x2min);}
      } else if (hi2) {
        pr(IPY,p) -= (meshsize.x2max - meshsize.x2min);
        if (has_reg) {pr(IPY1,p) -= (meshsize.x2max - meshsize.x2min);}
      }
      if (lo3) {
        pr(IPZ,p) += (meshsize.x3max - meshsize.x3min);
        if (has_reg) {pr(IPZ1,p) += (meshsize.x3max - meshsize.x3min);}
      } else if (hi3) {
        pr(IPZ,p) -= (meshsize.x3max - meshsize.x3min);
        if (has_reg) {pr(IPZ1,p) -= (meshsize.x3max - meshsize.x3min);}
      }
      }  // end shear/standard crossing branch
    }
  });
#if MPI_PARALLEL_ENABLED
  // one rank: nothing can leave it, so no count, no copy to the host and no MPI calls
  // (they made an MPI build about 13% slower than a serial one on one GPU)
  if (global_variable::nranks == 1) {
    nprtcl_send = 0;
    return TaskStatus::complete;
  }
  // This deep copy also synchronizes completion of the device append kernel.  Copy only
  // the valid prefix to the host; DualView::sync() would copy the full capacity.
  Kokkos::deep_copy(send_count.h_view, send_count.d_view);
  nprtcl_send = send_count.h_view(0);
  if (nprtcl_send < 0 || nprtcl_send > static_cast<int>(sendlist.extent(0))) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "Invalid particle send count " << nprtcl_send
              << " on rank " << global_variable::my_rank << " (capacity "
              << sendlist.extent(0) << ")" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  if (nprtcl_send > 0) {
    auto d_prefix = Kokkos::subview(sendlist.d_view,
                                    std::make_pair(0, nprtcl_send));
    auto h_prefix = Kokkos::subview(sendlist.h_view,
                                    std::make_pair(0, nprtcl_send));
    Kokkos::deep_copy(h_prefix, d_prefix);
    for (int n=0; n<nprtcl_send; ++n) {
      int p = sendlist.h_view(n).prtcl_indx;
      int r = sendlist.h_view(n).dest_rank;
      if (p < 0 || p >= npart || r < 0 || r >= global_variable::nranks) {
        std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
                  << std::endl << "Invalid particle send-list entry " << n
                  << " on rank " << global_variable::my_rank << ": particle=" << p
                  << ", destination rank=" << r << std::endl;
        std::exit(EXIT_FAILURE);
      }
    }
  }
#else
  nprtcl_send = 0;
#endif

  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::CountSendsAndRecvs()
//! \brief

TaskStatus ParticlesBoundaryValues::CountSendsAndRecvs() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {   // one rank: no messages
    sends_thisrank.clear(); recvs_thisrank.clear();
    nsends = 0; nrecvs = 0; nprtcl_send = 0; nprtcl_recv = 0;
    return TaskStatus::complete;
  }
  if (send_sort_device && nprtcl_send > kDeviceSortMin && DevExeSpace().concurrency() > 1) {
    // (destination rank, particle index) on the device (see send_sort_device); the host
    // copy for the message counts below
    auto d_prefix = Kokkos::subview(sendlist.d_view, std::make_pair(0, nprtcl_send));
    auto h_prefix = Kokkos::subview(sendlist.h_view, std::make_pair(0, nprtcl_send));
    Kokkos::sort(DevExeSpace(), d_prefix, ParticleRankIndexLess());
    Kokkos::deep_copy(h_prefix, d_prefix);
  } else {
  // Sort sendlist on host by destrank.
  std::sort(sendlist.h_view.data(), sendlist.h_view.data() + nprtcl_send, SortByRank);
  // Copy only the sorted valid prefix back to the device.
  if (nprtcl_send > 0) {
    auto h_prefix = Kokkos::subview(sendlist.h_view,
                                    std::make_pair(0, nprtcl_send));
    auto d_prefix = Kokkos::subview(sendlist.d_view,
                                    std::make_pair(0, nprtcl_send));
    Kokkos::deep_copy(d_prefix, h_prefix);
  }
  }

  // load STL::vector of ParticleMessageData with <sendrank, recvrank, nprtcls> for sends
  // from this rank. Length will be nsends; initially this length is unknown
  sends_thisrank.clear();
  if (nprtcl_send > 0) {
    int &myrank = global_variable::my_rank;
    int rank = sendlist.h_view(0).dest_rank;
    int nprtcl = 1;

    for (int n=1; n<nprtcl_send; ++n) {
      if (sendlist.h_view(n).dest_rank == rank) {
        ++nprtcl;
      } else {
        sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
        rank = sendlist.h_view(n).dest_rank;
        nprtcl = 1;
      }
    }
    sends_thisrank.emplace_back(ParticleMessageData(myrank,rank,nprtcl));
  }
  nsends = sends_thisrank.size();

  // Share number of ranks to send to amongst all ranks
  nsends_eachrank[global_variable::my_rank] = nsends;
  MPI_Allgather(&nsends, 1, MPI_INT, nsends_eachrank.data(), 1, MPI_INT, mpi_comm_part);

  // Now share ParticleMessageData amongst all ranks
  // First create vector of starting indices in full vector
  std::vector<int> nsends_displ;
  nsends_displ.resize(global_variable::nranks);
  nsends_displ[0] = 0;
  for (int n=1; n<(global_variable::nranks); ++n) {
    nsends_displ[n] = nsends_displ[n-1] + nsends_eachrank[n-1];
  }
  int nsends_allranks = nsends_displ[global_variable::nranks - 1] +
                        nsends_eachrank[global_variable::nranks - 1];
  // Load ParticleMessageData on this rank into full vector
  sends_allranks.resize(nsends_allranks, ParticleMessageData(0,0,0));
  for (int n=0; n<nsends_eachrank[global_variable::my_rank]; ++n) {
    sends_allranks[n + nsends_displ[global_variable::my_rank]] = sends_thisrank[n];
  }

  // Share tuples using MPI derived data type for tuple of 3*int
  MPI_Datatype mpi_ituple;
  MPI_Type_contiguous(3, MPI_INT, &mpi_ituple);
  MPI_Type_commit(&mpi_ituple);
  MPI_Allgatherv(MPI_IN_PLACE, nsends_eachrank[global_variable::my_rank],
                   mpi_ituple, sends_allranks.data(), nsends_eachrank.data(),
                   nsends_displ.data(), mpi_ituple, mpi_comm_part);
  MPI_Type_free(&mpi_ituple);
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::InitPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::InitPrtclRecv() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  // load STL::vector of ParticleMessageData with <sendrank,recvrank,nprtcl_recv> for
  // receives // on this rank. Length will be nrecvs, initially this length is unknown
  recvs_thisrank.clear();

  int nsends_allranks = sends_allranks.size();
  for (int n=0; n<nsends_allranks; ++n) {
    if (sends_allranks[n].recvrank == global_variable::my_rank) {
      recvs_thisrank.emplace_back(sends_allranks[n]);
    }
  }
  nrecvs = recvs_thisrank.size();

  // Figure out how many particles will be received from all ranks (int: per-rank counts and
  // particle indices are int; the sum in 64 bits for the check)
  std::int64_t nrecv64 = 0;
  for (int n=0; n<nrecvs; ++n) {
    nrecv64 += recvs_thisrank[n].nprtcls;
  }
  if (nrecv64 > std::numeric_limits<int>::max()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rank " << global_variable::my_rank << " would receive " << nrecv64
              << " particles in one exchange (more than 2^31 - 1 per rank)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  nprtcl_recv = static_cast<int>(nrecv64);

  // Retain high-water capacities across migration stages.  MPI uses only the valid
  // prefixes described by nprtcl_recv, so no buffer contents need to be preserved.
  std::size_t rrecv_required = static_cast<std::size_t>(pmy_part->nrdata)*nprtcl_recv;
  std::size_t irecv_required = static_cast<std::size_t>(pmy_part->nidata)*nprtcl_recv;
  ReserveMessageBuffer(prtcl_rrecvbuf, rrecv_required, pmy_part->nrdata);
  ReserveMessageBuffer(prtcl_irecvbuf, irecv_required, pmy_part->nidata);

  // Post non-blocking receives.  Offsets and sizes in 64 bits (nrdata Reals per particle pass
  // 2^31 above 126M particles); a message longer than msg_chunk_bytes as several MPI messages
  // (PostPrtclMessage), so the requests are one per MPI message
  bool no_errors=true;
  rrecv_req.clear();
  irecv_req.clear();
  const std::size_t rchunk = std::max<std::size_t>(1, msg_chunk_bytes/sizeof(Real));
  const std::size_t ichunk = std::max<std::size_t>(1, msg_chunk_bytes/sizeof(int));

  // Init receives for Reals
  std::size_t data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed
    const std::size_t data_size =
        static_cast<std::size_t>(pmy_part->nrdata)*(recvs_thisrank[n].nprtcls);
    int drank = recvs_thisrank[n].sendrank;
    int tag = 0; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    nmsg_mpi += PostPrtclMessage(false, prtcl_rrecvbuf.data() + data_start, data_size, rchunk,
                                 MPI_ATHENA_REAL, drank, tag, mpi_comm_part, rrecv_req,
                                 no_errors);
    data_start += data_size;
  }
  // Init receives for ints
  data_start=0;
  for (int n=0; n<nrecvs; ++n) {
    // calculate amount of data to be passed
    const std::size_t data_size =
        static_cast<std::size_t>(pmy_part->nidata)*(recvs_thisrank[n].nprtcls);
    int drank = recvs_thisrank[n].sendrank;
    int tag = 1; // 0 for Reals, 1 for ints

    // Post non-blocking receive
    nmsg_mpi += PostPrtclMessage(false, prtcl_irecvbuf.data() + data_start, data_size, ichunk,
                                 MPI_INT, drank, tag, mpi_comm_part, irecv_req, no_errors);
    data_start += data_size;
  }
  nmsg_prtcl += 2*static_cast<std::uint64_t>(nrecvs);

  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::PackAndSendPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::PackAndSendPrtcls() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  // Figure out how many particles will be sent from this ranks
  nprtcl_send=0;
  for (int n=0; n<nsends; ++n) {
    nprtcl_send += sends_thisrank[n].nprtcls;
  }

  bool no_errors=true;
  if (nprtcl_send > 0) {
    // Retain high-water capacities across migration stages.  Only the current valid
    // prefixes are packed and passed to MPI.
    std::size_t rsend_required = static_cast<std::size_t>(pmy_part->nrdata)*nprtcl_send;
    std::size_t isend_required = static_cast<std::size_t>(pmy_part->nidata)*nprtcl_send;
    ReserveMessageBuffer(prtcl_rsendbuf, rsend_required, pmy_part->nrdata);
    ReserveMessageBuffer(prtcl_isendbuf, isend_required, pmy_part->nidata);

    // sendlist on device is already sorted by destrank in CountSendAndRecvs()
    // Use sendlist on device to load particles into send buffer ordered by dest_rank
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rsendbuf = prtcl_rsendbuf;
    auto &isendbuf = prtcl_isendbuf;
    auto sendlist_d = sendlist.d_view;
    par_for("ppack",DevExeSpace(),0,(nprtcl_send-1), KOKKOS_LAMBDA(const int n) {
      int p = sendlist_d(n).prtcl_indx;
      // the row of particle n in 64 bits (nrdata*n passes 2^31 above 126M particles)
      const std::int64_t ib = static_cast<std::int64_t>(nidata)*n;
      const std::int64_t rb = static_cast<std::int64_t>(nrdata)*n;
      for (int i=0; i<nidata; ++i) {
        isendbuf(ib + i) = pi(i,p);
      }
      for (int i=0; i<nrdata; ++i) {
        rsendbuf(rb + i) = pr(i,p);
      }
    });

    // Post non-blocking sends: offsets and sizes in 64 bits, a message longer than
    // msg_chunk_bytes as several MPI messages (PostPrtclMessage; the receives match)
    Kokkos::fence();
    rsend_req.clear();
    isend_req.clear();
    const std::size_t rchunk = std::max<std::size_t>(1, msg_chunk_bytes/sizeof(Real));
    const std::size_t ichunk = std::max<std::size_t>(1, msg_chunk_bytes/sizeof(int));

    // Send Reals
    std::size_t data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed
      const std::size_t data_size =
          static_cast<std::size_t>(nrdata)*(sends_thisrank[n].nprtcls);
      int drank = sends_thisrank[n].recvrank;
      int tag = 0; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      nmsg_mpi += PostPrtclMessage(true, prtcl_rsendbuf.data() + data_start, data_size, rchunk,
                                   MPI_ATHENA_REAL, drank, tag, mpi_comm_part, rsend_req,
                                   no_errors);
      data_start += data_size;
    }
    // Send ints
    data_start=0;
    for (int n=0; n<nsends; ++n) {
      // calculate amount of data to be passed
      const std::size_t data_size =
          static_cast<std::size_t>(nidata)*(sends_thisrank[n].nprtcls);
      int drank = sends_thisrank[n].recvrank;
      int tag = 1; // 0 for Reals, 1 for ints

      // Post non-blocking sends
      nmsg_mpi += PostPrtclMessage(true, prtcl_isendbuf.data() + data_start, data_size, ichunk,
                                   MPI_INT, drank, tag, mpi_comm_part, isend_req, no_errors);
      data_start += data_size;
    }
    nmsg_prtcl += 2*static_cast<std::uint64_t>(nsends);
  }

  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in posting non-blocking receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::RecvAndUnpackPrtcls()
//! \brief

TaskStatus ParticlesBoundaryValues::RecvAndUnpackPrtcls() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) {   // one rank: the count is unchanged
    pmy_part->pmy_pack->pmesh->nprtcl_thisrank = pmy_part->nprtcl_thispack;
    pmy_part->pmy_pack->pmesh->nprtcl_eachrank[0] = pmy_part->nprtcl_thispack;
    return TaskStatus::complete;
  }
  // check that particle communications have all completed.  This comes first: the caller
  // polls this function until it returns complete, and on MPI backends that need MPI calls
  // to make progress, a host sort on every poll delays the communication (with millions of
  // particles sent at a regrid, the repeated sorts made the exchange take minutes)
  // (one request per MPI message: nrecvs each, unless a message was split)
  bool bflag = false;
  bool no_errors=true;
  const std::size_t nreq = std::max(rrecv_req.size(), irecv_req.size());
  for (std::size_t n=0; n<nreq; ++n) {
    int test;
    int ierr;
    if (n < rrecv_req.size()) {
      ierr = MPI_Test(&(rrecv_req[n]), &test, MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      if (!(static_cast<bool>(test))) {
        bflag = true;
      }
    }
    if (n < irecv_req.size()) {
      ierr = MPI_Test(&(irecv_req[n]), &test, MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
      if (!(static_cast<bool>(test))) {
        bflag = true;
      }
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
              << std::endl << "MPI error in testing non-blocking receives"
              << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // exit if particle communications have not completed
  if (bflag) {return TaskStatus::incomplete;}

  // Sort sendlist by index in particle array.  The indices are unique, so every sort gives
  // the same list: long lists are sorted on the device (the device copy is the rank-sorted
  // list of CountSendsAndRecvs), and their holes are filled on the device below
  const bool device_list = nprtcl_send > kDeviceSortMin && DevExeSpace().concurrency() > 1;
  if (device_list) {
    auto d_prefix = Kokkos::subview(sendlist.d_view, std::make_pair(0, nprtcl_send));
    Kokkos::sort(DevExeSpace(), d_prefix, ParticleIndexLess());
  } else {
    std::sort(sendlist.h_view.data(), sendlist.h_view.data() + nprtcl_send, SortByIndex);
    // Copy only the sorted valid prefix back to the device.
    if (nprtcl_send > 0) {
      auto h_prefix = Kokkos::subview(sendlist.h_view,
                                      std::make_pair(0, nprtcl_send));
      auto d_prefix = Kokkos::subview(sendlist.d_view,
                                      std::make_pair(0, nprtcl_send));
      Kokkos::deep_copy(d_prefix, h_prefix);
    }
  }

  // increase size of particle arrays if needed.  The arrays keep their capacity (their
  // extent may exceed the number of particles) and grow with 25% headroom, so that a net
  // inflow of a few particles does not reallocate them every cycle
  // (the count in 64 bits for the check: per-rank counts and particle indices are int; the
  // headroom stops at 2^31 - 1)
  const std::int64_t new_npart64 =
      static_cast<std::int64_t>(pmy_part->nprtcl_thispack) + (nprtcl_recv - nprtcl_send);
  if (new_npart64 > std::numeric_limits<int>::max()) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "rank " << global_variable::my_rank << " would hold " << new_npart64
              << " particles (more than 2^31 - 1 per rank)" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  int new_npart = static_cast<int>(new_npart64);
  if (new_npart > pmy_part->prtcl_rdata.extent_int(1) ||
      new_npart > pmy_part->prtcl_idata.extent_int(1)) {
    const std::int64_t cap = std::min<std::int64_t>(std::numeric_limits<int>::max(),
                                                    new_npart64 + new_npart64/4);
    Kokkos::resize(pmy_part->prtcl_idata, pmy_part->nidata, cap);
    Kokkos::resize(pmy_part->prtcl_rdata, pmy_part->nrdata, cap);
  }

  // unpack particles into positions of sent particles
  if (nprtcl_recv > 0) {
    int nrdata = pmy_part->nrdata;
    int nidata = pmy_part->nidata;
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    auto &rrecvbuf = prtcl_rrecvbuf;
    auto &irecvbuf = prtcl_irecvbuf;
    auto sendlist_d = sendlist.d_view;
    int nsend = nprtcl_send;
    int &npart = pmy_part->nprtcl_thispack;
    par_for("punpack",DevExeSpace(),0,(nprtcl_recv-1), KOKKOS_LAMBDA(const int n) {
      int p;
      if (n < nsend) {
        p = sendlist_d(n).prtcl_indx; // place particles in holes created by sends
      } else {
        p = npart + (n - nsend);           // place particle at end of arrays
      }
      // the row of particle n in 64 bits (nrdata*n passes 2^31 above 126M particles)
      const std::int64_t ib = static_cast<std::int64_t>(nidata)*n;
      const std::int64_t rb = static_cast<std::int64_t>(nrdata)*n;
      for (int i=0; i<nidata; ++i) {
        pi(i,p) = irecvbuf(ib + i);
      }
      for (int i=0; i<nrdata; ++i) {
        pr(i,p) = rrecvbuf(rb + i);
      }
    });
  }

  // At this point have filled npart_recv holes in particle arrays from sends
  // If (nprtcl_recv < nprtcl_send), have to move particles from end of arrays to fill
  // remaining holes
  int nremain = nprtcl_send - nprtcl_recv;
  if (nremain > 0 && device_list) {
    // the same (hole, last particle) pairs as the host loop below, on the device: the
    // particles of the tail [npart - nremain, npart) that are not holes, from the end, into
    // the holes below the tail in increasing order (sendlist(nrecv), sendlist(nrecv+1), ...;
    // every hole of a received particle lies below the tail)
    const int npart = pmy_part->nprtcl_thispack;
    const int first_tail = npart - nremain, nrecv = nprtcl_recv, nsend = nprtcl_send;
    auto sl = sendlist.d_view;
    DvceArray1D<int> tail_hole("prtcl_tail_hole", nremain);
    par_for("prtcl_tail_holes",DevExeSpace(),nrecv,nsend-1, KOKKOS_LAMBDA(const int k) {
      const int p = sl(k).prtcl_indx;
      if (p >= first_tail) tail_hole(p - first_tail) = 1;
    });
    auto &pr = pmy_part->prtcl_rdata;
    auto &pi = pmy_part->prtcl_idata;
    const int nrdata = pmy_part->nrdata, nidata = pmy_part->nidata;
    Kokkos::parallel_scan("prtcl_fill_holes_scan",
                          Kokkos::RangePolicy<>(DevExeSpace(), 0, nremain),
    KOKKOS_LAMBDA(const int j, int &q, const bool final) {
      const int src = npart - 1 - j;
      if (tail_hole(src - first_tail)) return;
      if (final) {
        const int dst = sl(nrecv + q).prtcl_indx;
        for (int i=0; i<nrdata; ++i) pr(i,dst) = pr(i,src);
        for (int i=0; i<nidata; ++i) pi(i,dst) = pi(i,src);
      }
      ++q;
    });
  } else if (nremain > 0) {
    // the (hole, last particle) pairs as before, found on the host; the copies in one kernel
    // instead of two device copies per pair
    int &npart = pmy_part->nprtcl_thispack;
    int i_last_hole = nprtcl_send-1;
    int i_next_hole = nprtcl_recv;
    std::vector<int> pairs;
    for (int n=1; n<=nremain; ++n) {
      int nend = npart-n;
      if (nend > sendlist.h_view(i_last_hole).prtcl_indx) {
        // copy particle from end into hole
        pairs.push_back(sendlist.h_view(i_next_hole).prtcl_indx);
        pairs.push_back(nend);
        i_next_hole += 1;
      } else {
        // this index contains a hole, so do nothing except find new index of last hole
        i_last_hole -= 1;
      }
    }
    const int npair = static_cast<int>(pairs.size())/2;
    if (npair > 0) {
      Kokkos::View<int*, HostMemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
          hpairs(pairs.data(), pairs.size());
      DvceArray1D<int> dpairs("prtcl_fill_pairs", pairs.size());
      Kokkos::deep_copy(dpairs, hpairs);
      auto &pr = pmy_part->prtcl_rdata;
      auto &pi = pmy_part->prtcl_idata;
      const int nrdata = pmy_part->nrdata, nidata = pmy_part->nidata;
      par_for("prtcl_fill_holes",DevExeSpace(),0,npair-1, KOKKOS_LAMBDA(const int k) {
        const int dst = dpairs(2*k), src = dpairs(2*k + 1);
        for (int i=0; i<nrdata; ++i) pr(i,dst) = pr(i,src);
        for (int i=0; i<nidata; ++i) pi(i,dst) = pi(i,src);
      });
    }
    // the arrays keep their capacity (no shrink)
  }

  // Update nparticles_thisrank.  Update cost array (use npart_thismb[nmb]?)
  pmy_part->nprtcl_thispack = new_npart;
  pmy_part->pmy_pack->pmesh->nprtcl_thisrank = new_npart;
  MPI_Allgather(&new_npart,1,MPI_INT,(pmy_part->pmy_pack->pmesh->nprtcl_eachrank),1,
                MPI_INT,MPI_COMM_WORLD);
#endif
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearPrtclSend()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclSend() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  bool no_errors=true;
  // wait for all non-blocking sends for vars to finish before continuing
  // (one request per MPI message: nsends each, unless a message was split)
  const std::size_t nreq = std::max(rsend_req.size(), isend_req.size());
  for (std::size_t n=0; n<nreq; ++n) {
    int ierr;
    if (n < rsend_req.size()) {
      ierr = MPI_Wait(&(rsend_req[n]), MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
    }
    if (n < isend_req.size()) {
      ierr = MPI_Wait(&(isend_req[n]), MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing sends" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rsend_req.clear();
  isend_req.clear();
#endif
  nsends=0;
  return TaskStatus::complete;
}

//----------------------------------------------------------------------------------------
//! \fn void ParticlesBoundaryValues::ClearPrtclRecv()
//! \brief

TaskStatus ParticlesBoundaryValues::ClearPrtclRecv() {
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks == 1) return TaskStatus::complete;
  bool no_errors=true;
  // wait for all non-blocking receives to finish before continuing
  // (one request per MPI message: nrecvs each, unless a message was split)
  const std::size_t nreq = std::max(rrecv_req.size(), irecv_req.size());
  for (std::size_t n=0; n<nreq; ++n) {
    int ierr;
    if (n < rrecv_req.size()) {
      ierr = MPI_Wait(&(rrecv_req[n]), MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
    }
    if (n < irecv_req.size()) {
      ierr = MPI_Wait(&(irecv_req[n]), MPI_STATUS_IGNORE);
      if (ierr != MPI_SUCCESS) {no_errors=false;}
    }
  }
  // Quit if MPI error detected
  if (!(no_errors)) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
       << std::endl << "MPI error in clearing receives" << std::endl;
    std::exit(EXIT_FAILURE);
  }
  rrecv_req.clear();
  irecv_req.clear();
#endif
  nrecvs=0;
  return TaskStatus::complete;
}

} // namespace particles
