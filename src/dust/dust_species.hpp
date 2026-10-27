#ifndef DUST_DUST_SPECIES_HPP_
#define DUST_DUST_SPECIES_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_species.hpp
//! \brief The resolved dust species table: one parse of
//! the species keys of <dust> before the particles are allocated, shared by the particle
//! module, DustGasDrag, the problem generators and the outputs.
//!
//! Two independent groups of keys, each either new or legacy (never mixed):
//!   stopping times   new: tau_s_N | stokes_N (Omega: <shearing_box>/omega0 or
//!                    <dust>/stokes_omega_ref) | species_distribution = powerlaw with
//!                    tau_s_min/max (or stokes_min/max);  legacy: taus_N
//!   abundance and    new: epsilon_N | Z_N (with column_normalization = full_profile | domain)
//!   sampling         and ppc_N, or (powerlaw) epsilon | Z and ppc_per_species;
//!                    legacy: <problem>/eps_N or <dust>/dust_to_gas and <particles>/ppc
//! Legacy inputs keep their exact behaviour (to be deprecated).  The input index N is
//! 1-based; species_id = N - 1.

#include <string>
#include <vector>

class ParameterInput;

namespace dust {

struct SpeciesTable {
  int nspecies = 1;
  // stopping times
  bool new_tau = false;              // tau_s_N, stokes_N or the power law
  std::string tau_source = "taus_N (legacy)";
  std::vector<double> tau_s;         // code time
  double omega_ref = 0.0;            // Omega of the Stokes numbers (0: none known)
  // abundance and sampling
  bool new_init = false;             // epsilon_N / Z_N and ppc_N (or the power law)
  std::string abundance = "legacy";  // epsilon | Z | legacy
  std::string column_normalization;  // Z: full_profile | domain
  std::vector<double> value;         // epsilon_N or Z_N (new_init)
  std::vector<int> ppc;              // root ppc per species (new_init)
  int ppc_total = 0;                 // sum of ppc (new_init)
  bool powerlaw = false;
  double mass_slope = 0.0, total = 0.0;   // power law: exponent p and the total epsilon / Z
  bool equal_ppc = false;            // every ppc_N the same (the legacy interleaved layout)
  std::vector<double> hp;            // initial H_p per species (<problem>/H_p_N | H_p |
                                     // H_p_min+H_p_max); empty: none given (legacy dust_hp)

  bool NewAPI() const {return new_tau || new_init;}
  // slot k (0 <= k < ppc_total) of a root cell: species and the index within the species.
  // Equal ppc: interleaved as the legacy layout (k % nspecies); otherwise consecutive blocks.
  void SlotSpecies(const int k, int &s, int &kk) const;
  std::vector<int> Offsets() const;  // nspecies + 1 cumulative ppc (blocks layout)
  // "key = value" lines of the resolved table (pbin header, logs; prefix e.g. "# ")
  std::string Describe(const std::string &prefix) const;
  // the restart record species_table: ASCII, values with 17 digits
  std::string Record() const;
};

// the differences of two species_table records, one line per quantity ("" if they agree;
// numbers to a relative 1e-12, the rest exactly)
std::string CompareSpeciesRecords(const std::string &file, const std::string &input);
// the root ppc of the run: the sum of ppc_N with the new keys, else <particles>/ppc (default 1,
// as the particle module reads it)
double RootPpc(ParameterInput *pin);

// the table of this run (parsed on the first call; the particles' allocation calls it first)
const SpeciesTable &Species(ParameterInput *pin);
// a problem generator that does not implement the new abundance/sampling keys stops on them
void RequireLegacySpeciesInit(ParameterInput *pin, const char *pgen);
// <problem>/H_p_N | H_p | H_p_min + H_p_max (new keys, required with a stratified new_init),
// or the legacy single <problem>/dust_hp (default H_gas); per species
std::vector<double> SpeciesHp(ParameterInput *pin, const SpeciesTable &t, const double hgas,
                              const bool legacy_default);

}  // namespace dust
#endif  // DUST_DUST_SPECIES_HPP_
