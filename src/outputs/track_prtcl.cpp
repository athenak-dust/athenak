//========================================================================================
// AthenaK astrophysical fluid dynamics code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file track_prtcl.cpp
//! \brief writes data for tracked particles in unformatted binary
//!
//! File trk/<basename>.trk (<outputN>/file_type = trk, nparticles = N): the particles with
//! tags 0 ... N-1.  Every output appends one block: the text header (newline, the line
//!   "# AthenaK tracked particle data at time= <t>  nranks= <n>  cycle=<c>  ntracked_prtcls=<N>",
//! newline, space, newline), followed by N records in tag order, record k for tag k:
//! 6 float32 (x, y, z, vx, vy, vz; the stored particle velocity), 24 bytes, native byte
//! order; all six NaN for a tag that no particle carries at that time.  A block is exactly
//! the header plus 24 N bytes.  Particle tags are unique (AthenaK convention); a repeated
//! tag stops the run only when a rank finds more than N tracked particles, otherwise the
//! later record of that tag takes its slot.

#include <sys/stat.h>  // mkdir
#include <vector>

#include <algorithm>
#include <cstdio>      // fwrite(), fclose(), fopen(), fnprintf(), snprintf()
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#include "athena.hpp"
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "particles/particles.hpp"
#include "outputs.hpp"

//----------------------------------------------------------------------------------------
// ctor: also calls BaseTypeOutput base class constructor

TrackedParticleOutput::TrackedParticleOutput(ParameterInput *pin, Mesh *pm,
                                             OutputParameters op) :
  BaseTypeOutput(pin, pm, op) {
  // create new directory for this output. Comments in binary.cpp constructor explain why
  mkdir("trk",0775);
  // allocate arrays
  npout_eachrank.resize(global_variable::nranks);
  ntrack = pin->GetInteger(op.block_name,"nparticles");
  // (rank 0 gathers 6 floats per tracked particle with int MPI counts)
  if (ntrack < 0 || ntrack > std::numeric_limits<int>::max()/6) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<" << op.block_name << ">/nparticles must be in [0, "
              << std::numeric_limits<int>::max()/6 << "]" << std::endl;
    exit(EXIT_FAILURE);
  }
  // TODO(@user) improve guess below?
  ntrack_thisrank = ntrack;
}

//----------------------------------------------------------------------------------------
// TrackedParticleOutput::LoadOutputData()
// Copies data for tracked particles on this rank to host outpart array

void TrackedParticleOutput::LoadOutputData(Mesh *pm) {
  // Load data for tracked particles on this rank into new device array
  DualArray1D<TrackedParticleData> tracked_prtcl("d_trked",ntrack_thisrank);
  int npart = pm->nprtcl_thisrank;
  auto &pr = pm->pmb_pack->ppart->prtcl_rdata;
  auto &pi = pm->pmb_pack->ppart->prtcl_idata;
  // the counter in device memory (the kernel runs on the device: a host stack address is
  // invalid there); the list order does not matter, WriteOutputFile places the records by tag
  DvceArray1D<int> counter("trk_counter",1);
  auto trk = tracked_prtcl.d_view;
  const int cap = ntrack_thisrank;
  // the 64-bit tag of dust particles (words PTAG, PTAGH); cosmic rays: PTAG
  const std::uint64_t ntrack_ = (ntrack > 0) ? static_cast<std::uint64_t>(ntrack) : 0;
  const bool hiword = (pm->pmb_pack->ppart->nidata > PTAGH);
  par_for("part_update",DevExeSpace(),0,(npart-1), KOKKOS_LAMBDA(const int p) {
    const std::uint64_t tag = hiword ? GetTag(pi,p)
                                     : static_cast<std::uint64_t>(static_cast<std::uint32_t>(pi(PTAG,p)));
    if (tag < ntrack_) {
      int index = Kokkos::atomic_fetch_add(&counter(0),1);
      if (index < cap) {
        trk(index).tag = tag;
        trk(index).x   = pr(IPX,p);
        trk(index).y   = pr(IPY,p);
        trk(index).z   = pr(IPZ,p);
        trk(index).vx  = pr(IPVX,p);
        trk(index).vy  = pr(IPVY,p);
        trk(index).vz  = pr(IPVZ,p);
      }
    }
  });
  auto hcounter = Kokkos::create_mirror_view_and_copy(HostMemSpace(), counter);
  npout = hcounter(0);
  if (npout > cap) {   // only with repeated tags (fewer repeats go unnoticed)
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << npout << " particles of this rank carry a tag below nparticles = " << ntrack
              << " (tags repeat)" << std::endl;
    exit(EXIT_FAILURE);
  }
  // share number of tracked particles to be output across all ranks
  npout_eachrank[global_variable::my_rank] = npout;
#if MPI_PARALLEL_ENABLED
  MPI_Allgather(&npout, 1, MPI_INT, npout_eachrank.data(), 1, MPI_INT, MPI_COMM_WORLD);
#endif
  tracked_prtcl.resize(npout);
  // sync tracked particle device array with host
  tracked_prtcl.template modify<DevExeSpace>();
  tracked_prtcl.template sync<HostMemSpace>();

  // copy host view into host outpart array
  Kokkos::realloc(outpart, npout);
  Kokkos::deep_copy(outpart, tracked_prtcl.h_view);
}

//----------------------------------------------------------------------------------------
//! \fn void TrackedParticleOutput:::WriteOutputFile(Mesh *pm)
//! \brief Cycles over all tracked particles on this rank and writes ouput data
//! With MPI, all particles are written to the same file.

void TrackedParticleOutput::WriteOutputFile(Mesh *pm, ParameterInput *pin) {
  // create filename: "trk/file_basename".trk
  std::string fname;
  fname.assign("trk/");
  fname.append(out_params.file_basename);
  fname.append(".trk");

  // the records of this rank's tracked particles (6 floats each) and their tags
  std::vector<float> data(6*static_cast<std::size_t>(npout));
  std::vector<std::uint64_t> tags(npout);
  for (int p=0; p<npout; ++p) {
    tags[p] = outpart(p).tag;
    data[ 6*p   ] = static_cast<float>(outpart(p).x);
    data[(6*p)+1] = static_cast<float>(outpart(p).y);
    data[(6*p)+2] = static_cast<float>(outpart(p).z);
    data[(6*p)+3] = static_cast<float>(outpart(p).vx);
    data[(6*p)+4] = static_cast<float>(outpart(p).vy);
    data[(6*p)+5] = static_cast<float>(outpart(p).vz);
  }
  // gathered on rank 0 (npout_eachrank from LoadOutputData), which writes the whole block:
  // one writer, every record at its tag's slot, no overlapping writes
#if MPI_PARALLEL_ENABLED
  if (global_variable::nranks > 1) {
    const int nranks = global_variable::nranks;
    std::vector<int> cnt(nranks), dsp(nranks), cnt6(nranks), dsp6(nranks);
    std::int64_t ntot = 0;
    for (int n=0; n<nranks; ++n) {
      cnt[n] = npout_eachrank[n];
      cnt6[n] = 6*npout_eachrank[n];
      dsp[n] = static_cast<int>(ntot);
      dsp6[n] = static_cast<int>(6*ntot);
      ntot += npout_eachrank[n];
    }
    if (6*ntot > std::numeric_limits<int>::max()) {   // only with tags repeated across ranks
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << ntot << " tracked particles on all ranks (tags repeat)" << std::endl;
      exit(EXIT_FAILURE);
    }
    std::vector<std::uint64_t> alltags;
    std::vector<float> alldata;
    if (global_variable::my_rank == 0) {
      alltags.resize(ntot);
      alldata.resize(6*ntot);
    }
    MPI_Gatherv(tags.data(), npout, MPI_UINT64_T, alltags.data(), cnt.data(), dsp.data(),
                MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Gatherv(data.data(), 6*npout, MPI_FLOAT, alldata.data(), cnt6.data(), dsp6.data(),
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    tags.swap(alltags);
    data.swap(alldata);
  }
#endif

  // rank 0 appends the header and the N records in tag order (NaN where no particle has the tag)
  if (global_variable::my_rank == 0) {
    std::vector<float> block(6*static_cast<std::size_t>(ntrack),
                             std::numeric_limits<float>::quiet_NaN());
    for (std::size_t j=0; j<tags.size(); ++j) {
      for (int k=0; k<6; ++k) block[6*tags[j] + k] = data[6*j + k];
    }
    std::stringstream msg;
    msg << std::endl << "# AthenaK tracked particle data at time= " << pm->time
        << "  nranks= " << global_variable::nranks
        << "  cycle=" << pm->ncycle
        << "  ntracked_prtcls=" << ntrack << std::endl;
    FILE *pfile;
    if ((pfile = std::fopen(fname.c_str(),"a")) == nullptr) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
        << std::endl << "Output file '" << fname << "' could not be opened" <<std::endl;
      exit(EXIT_FAILURE);
    }
    // every stdio result is checked: buffered records can still fail at fclose (disk full,
    // quota), and a silent failure would leave the block short and advance the output time
    bool ok = (std::fprintf(pfile,"%s \n",msg.str().c_str()) >= 0);
    ok = ok && (std::fwrite(block.data(), sizeof(float), block.size(), pfile) == block.size());
    ok = (std::fclose(pfile) == 0) && ok;
    if (!ok) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__
          << std::endl << "particle data not written correctly to tracked particle file '"
          << fname << "'" << std::endl;
      exit(EXIT_FAILURE);
    }
  }

  // increment counters
  if (out_params.last_time < 0.0) {
    out_params.last_time = pm->time;
  } else {
    out_params.last_time += out_params.dt;
  }
  pin->SetReal(out_params.block_name, "last_time", out_params.last_time);
  return;
}
