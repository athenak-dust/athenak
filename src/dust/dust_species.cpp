//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_species.cpp
//! \brief The resolved dust species table (dust_species.hpp).  All checks happen here,
//! once, before the particles are allocated: complete vectors
//! (every species 1..nspecies, no index beyond nspecies), no mixing of a legacy and a new key
//! for the same quantity, finite positive stopping times, non-negative abundances, integer
//! ppc, a positive abundance only with particles and particles only with a positive abundance.

#include "dust/dust_species.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "athena.hpp"
#include "globals.hpp"
#include "parameter_input.hpp"

namespace dust {

namespace {

[[noreturn]] void Fail(const std::string &msg) {
  std::cout << "### FATAL ERROR in dust species input: " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}

bool Has(ParameterInput *pin, const std::string &block, const std::string &key) {
  return pin->DoesParameterExist(block, key);
}

// the indices N of the keys <block>/<prefix>N (N all digits)
std::set<int> Indexed(ParameterInput *pin, const std::string &block, const std::string &prefix) {
  std::set<int> out;
  for (const auto &b : pin->block) {
    if (b.block_name != block) continue;
    for (const auto &l : b.line) {
      const std::string &k = l.param_name;
      if (k.size() <= prefix.size() || k.compare(0, prefix.size(), prefix) != 0) continue;
      const std::string rest = k.substr(prefix.size());
      if (rest.find_first_not_of("0123456789") != std::string::npos) continue;
      out.insert(std::atoi(rest.c_str()));
    }
  }
  return out;
}

// a vector key family present: it must be complete (1..n) and not go beyond n
void Complete(const std::set<int> &idx, const int n, const std::string &what) {
  for (int i : idx) {
    if (i < 1 || i > n) {
      Fail(what + std::to_string(i) + " is beyond <dust>/nspecies = " + std::to_string(n));
    }
  }
  if (static_cast<int>(idx.size()) != n) {
    std::string miss;
    for (int i=1; i<=n; ++i) if (!idx.count(i)) miss += " " + what + std::to_string(i);
    Fail("incomplete vector " + what + "N (nspecies = " + std::to_string(n) + "), missing:" + miss);
  }
}

double Positive(ParameterInput *pin, const std::string &block, const std::string &key) {
  const double v = pin->GetReal(block, key);
  if (!std::isfinite(v) || !(v > 0.0)) Fail("<" + block + ">/" + key + " must be finite and positive");
  return v;
}
double NonNegative(ParameterInput *pin, const std::string &block, const std::string &key) {
  const double v = pin->GetReal(block, key);
  if (!std::isfinite(v) || v < 0.0) Fail("<" + block + ">/" + key + " must be finite and >= 0");
  return v;
}
int Count(ParameterInput *pin, const std::string &block, const std::string &key) {
  const double v = pin->GetReal(block, key);
  if (!std::isfinite(v) || v < 0.0 || v != std::floor(v) || v > 1.0e9) {
    Fail("<" + block + ">/" + key + " must be a non-negative integer");
  }
  return static_cast<int>(v);
}

// <problem>/H_p_N | H_p | H_p_min + H_p_max; empty when none is given
std::vector<double> ResolveHp(ParameterInput *pin, const SpeciesTable &t) {
  const std::string P = "problem";
  const int n = t.nspecies;
  if (!pin->DoesBlockExist(P)) return {};
  const auto i_hp = Indexed(pin, P, "H_p_");
  const bool common = Has(pin, P, "H_p");
  const bool range = Has(pin, P, "H_p_min") || Has(pin, P, "H_p_max");
  const int kinds = (!i_hp.empty()) + common + range;
  if (kinds == 0) return {};
  if (kinds > 1) Fail("give one of <problem>/H_p_N, H_p, or H_p_min with H_p_max");
  std::vector<double> hp(n, 0.0);
  if (!i_hp.empty()) {
    Complete(i_hp, n, "H_p_");
    for (int s=0; s<n; ++s) hp[s] = Positive(pin, P, "H_p_" + std::to_string(s + 1));
  } else if (common) {
    const double h = Positive(pin, P, "H_p");
    for (auto &v : hp) v = h;
  } else {
    const double hlo = Positive(pin, P, "H_p_min"), hhi = Positive(pin, P, "H_p_max");
    if (hlo > hhi) Fail("H_p_min must not exceed H_p_max");
    double tmin = t.tau_s[0], tmax = t.tau_s[0];
    for (double v : t.tau_s) {tmin = std::min(tmin, v); tmax = std::max(tmax, v);}
    if (tmax == tmin) {
      if (hlo != hhi) Fail("H_p_min / H_p_max with equal stopping times need min = max");
      for (auto &v : hp) v = hlo;
    } else {
      // small t_s the large H_p, large t_s the small one, logarithmic in both (Athena-C)
      for (int s=0; s<n; ++s) {
        const double f = std::log(t.tau_s[s]/tmin)/std::log(tmax/tmin);
        hp[s] = std::exp(std::log(hhi) + f*(std::log(hlo) - std::log(hhi)));
      }
    }
  }
  return hp;
}

SpeciesTable Parse(ParameterInput *pin) {
  SpeciesTable t;
  const std::string D = "dust", P = "problem";
  t.nspecies = pin->GetOrAddInteger(D, "nspecies", 1);
  const int n = t.nspecies;
  if (n < 1) Fail("<dust>/nspecies must be at least 1");
  // Omega of Stokes numbers: the shearing box, or <dust>/stokes_omega_ref
  const bool sbox = Has(pin, "shearing_box", "omega0");
  if (sbox) t.omega_ref = pin->GetReal("shearing_box", "omega0");
  if (Has(pin, D, "stokes_omega_ref")) {
    const double w = Positive(pin, D, "stokes_omega_ref");
    if (sbox && w != t.omega_ref) {
      Fail("<dust>/stokes_omega_ref differs from <shearing_box>/omega0");
    }
    t.omega_ref = w;
  }
  // ---- the form of the table
  std::string dist = "explicit";
  if (Has(pin, D, "species_distribution")) dist = pin->GetString(D, "species_distribution");
  if (dist != "explicit" && dist != "powerlaw") {
    Fail("<dust>/species_distribution must be explicit or powerlaw");
  }
  t.powerlaw = (dist == "powerlaw");
  const auto i_taus = Indexed(pin, D, "taus_"), i_tau = Indexed(pin, D, "tau_s_");
  const auto i_st = Indexed(pin, D, "stokes_"), i_eps = Indexed(pin, D, "epsilon_");
  const auto i_z = Indexed(pin, D, "Z_"), i_ppc = Indexed(pin, D, "ppc_");
  const auto i_peps = Indexed(pin, P, "eps_");
  const bool range_tau = Has(pin, D, "tau_s_min") || Has(pin, D, "tau_s_max");
  const bool range_st = Has(pin, D, "stokes_min") || Has(pin, D, "stokes_max");
  const bool tot_eps = Has(pin, D, "epsilon"), tot_z = Has(pin, D, "Z");
  const bool tot_ppc = Has(pin, D, "ppc_per_species");

  // ---- stopping times
  t.tau_s.assign(n, 0.0);
  if (t.powerlaw) {
    if (!i_taus.empty() || !i_tau.empty() || !i_st.empty()) {
      Fail("species_distribution = powerlaw generates the stopping times: leave out taus_N, "
           "tau_s_N and stokes_N");
    }
    if (range_tau == range_st) {
      Fail("species_distribution = powerlaw needs tau_s_min and tau_s_max, or stokes_min and "
           "stokes_max (one pair)");
    }
    const std::string a = range_tau ? "tau_s_" : "stokes_";
    double lo = Positive(pin, D, a + "min"), hi = Positive(pin, D, a + "max");
    if (range_st) {
      if (!(t.omega_ref > 0.0)) {
        Fail("stokes_min/max need <shearing_box>/omega0 or <dust>/stokes_omega_ref");
      }
      lo /= t.omega_ref; hi /= t.omega_ref;
    }
    if (n == 1 && lo != hi) Fail("species_distribution = powerlaw with one species needs min = max");
    if (hi < lo) Fail("species_distribution = powerlaw: max below min");
    for (int s=0; s<n; ++s) {
      t.tau_s[s] = (n == 1) ? lo : lo*std::pow(hi/lo, static_cast<double>(s)/(n - 1));
    }
    t.new_tau = true;
    t.tau_source = range_tau ? "powerlaw (tau_s_min, tau_s_max)" : "powerlaw (stokes_min, stokes_max)";
  } else {
    if (range_tau || range_st) Fail("tau_s_min/max and stokes_min/max belong to species_distribution = powerlaw");
    const int kinds = (!i_taus.empty()) + (!i_tau.empty()) + (!i_st.empty());
    if (kinds > 1) {
      Fail("stopping times given by more than one of taus_N (legacy), tau_s_N, stokes_N: use one");
    }
    if (kinds == 0) Fail("the stopping times are missing: give tau_s_N (or stokes_N; legacy taus_N)");
    if (!i_taus.empty()) {
      Complete(i_taus, n, "taus_");
      // legacy: read exactly as before (to be deprecated; tau_s_N is the new spelling)
      for (int s=0; s<n; ++s) {
        const double ts = pin->GetReal(D, "taus_" + std::to_string(s + 1));
        if (ts <= 0.0) Fail("Dust stopping times must be positive");
        t.tau_s[s] = ts;
      }
    } else if (!i_tau.empty()) {
      Complete(i_tau, n, "tau_s_");
      for (int s=0; s<n; ++s) t.tau_s[s] = Positive(pin, D, "tau_s_" + std::to_string(s + 1));
      t.new_tau = true;
      t.tau_source = "tau_s_N";
    } else {
      Complete(i_st, n, "stokes_");
      if (!(t.omega_ref > 0.0)) Fail("stokes_N need <shearing_box>/omega0 or <dust>/stokes_omega_ref");
      for (int s=0; s<n; ++s) {
        t.tau_s[s] = Positive(pin, D, "stokes_" + std::to_string(s + 1))/t.omega_ref;
      }
      t.new_tau = true;
      t.tau_source = "stokes_N";
    }
  }

  // ---- abundance and sampling
  t.new_init = t.powerlaw || !i_eps.empty() || !i_z.empty() || !i_ppc.empty() || tot_eps ||
               tot_z || tot_ppc || Has(pin, D, "column_normalization");
  if (!t.new_init) return t;   // legacy: <problem>/eps_N or dust_to_gas, <particles>/ppc
  if (!i_peps.empty()) {
    Fail("the new species keys (epsilon_N / Z_N / ppc_N) and the legacy <problem>/eps_N cannot be "
         "mixed: give the abundances once");
  }
  if (Has(pin, D, "dust_to_gas")) {
    Fail("the new species keys and the legacy <dust>/dust_to_gas cannot be mixed");
  }
  t.value.assign(n, 0.0);
  t.ppc.assign(n, 0);
  if (t.powerlaw) {
    if (!i_eps.empty() || !i_z.empty() || !i_ppc.empty()) {
      Fail("species_distribution = powerlaw: give the total epsilon or Z and ppc_per_species, "
           "not epsilon_N / Z_N / ppc_N");
    }
    if (tot_eps == tot_z) Fail("species_distribution = powerlaw needs epsilon or Z (one of them)");
    if (!tot_ppc) Fail("species_distribution = powerlaw needs ppc_per_species");
    t.abundance = tot_eps ? "epsilon" : "Z";
    t.total = NonNegative(pin, D, tot_eps ? "epsilon" : "Z");
    t.mass_slope = pin->GetReal(D, "mass_slope");
    if (!std::isfinite(t.mass_slope)) Fail("<dust>/mass_slope must be finite");
    const int pps = Count(pin, D, "ppc_per_species");
    double wsum = 0.0;
    std::vector<double> w(n);
    for (int s=0; s<n; ++s) {w[s] = std::pow(t.tau_s[s], t.mass_slope); wsum += w[s];}
    for (int s=0; s<n; ++s) {t.value[s] = t.total*w[s]/wsum; t.ppc[s] = pps;}
  } else {
    if (tot_ppc || Has(pin, D, "mass_slope")) {
      Fail("ppc_per_species and mass_slope belong to species_distribution = powerlaw");
    }
    if (!i_eps.empty() && !i_z.empty()) Fail("give epsilon_N or Z_N, not both");
    if (i_eps.empty() && i_z.empty()) Fail("ppc_N needs the abundances epsilon_N (or Z_N)");
    if (i_ppc.empty()) Fail("epsilon_N / Z_N need the sampling ppc_N");
    t.abundance = i_eps.empty() ? "Z" : "epsilon";
    Complete(t.abundance == "Z" ? i_z : i_eps, n, t.abundance + "_");
    Complete(i_ppc, n, "ppc_");
    for (int s=0; s<n; ++s) {
      t.value[s] = NonNegative(pin, D, t.abundance + "_" + std::to_string(s + 1));
      t.ppc[s] = Count(pin, D, "ppc_" + std::to_string(s + 1));
    }
    // a redundant total is only a consistency check
    double sum = 0.0;
    for (double v : t.value) sum += v;
    t.total = sum;
    if (tot_eps || tot_z) {
      if ((tot_eps && t.abundance != "epsilon") || (tot_z && t.abundance != "Z")) {
        Fail("the total <dust>/epsilon or Z does not match the kind of the vector");
      }
      const double g = pin->GetReal(D, t.abundance);
      if (std::fabs(g - sum) > 1.0e-12*std::max(1.0, std::fabs(sum))) {
        Fail("<dust>/" + t.abundance + " differs from the sum of " + t.abundance + "_N");
      }
    }
  }
  if (t.abundance == "Z") {
    // redraw (as in Athena-C) is the default; full_profile and domain
    // are explicit alternatives
    t.column_normalization = Has(pin, D, "column_normalization") ?
        pin->GetString(D, "column_normalization") : std::string("redraw");
    if (t.column_normalization != "redraw" && t.column_normalization != "full_profile" &&
        t.column_normalization != "domain") {
      Fail("<dust>/column_normalization must be redraw (default), full_profile or domain");
    }
  } else if (Has(pin, D, "column_normalization")) {
    Fail("column_normalization belongs to Z (column ratios), not to epsilon");
  }
  t.ppc_total = 0;
  for (int s=0; s<n; ++s) {
    if (t.value[s] > 0.0 && t.ppc[s] == 0) {
      Fail("species " + std::to_string(s + 1) + " has a positive abundance and no particles (ppc_" +
           std::to_string(s + 1) + " = 0)");
    }
    if (t.value[s] == 0.0 && t.ppc[s] > 0) {
      Fail("species " + std::to_string(s + 1) + " has particles but zero abundance (test particles "
           "need an explicit test-particle mode)");
    }
    t.ppc_total += t.ppc[s];
  }
  if (t.ppc_total < 1) Fail("no particles: every ppc_N is 0");
  t.equal_ppc = true;
  for (int s=0; s<n; ++s) if (t.ppc[s] != t.ppc[0]) t.equal_ppc = false;
  // the sampling is given once; <particles>/ppc is the legacy key
  // and is not allowed next to ppc_N / ppc_per_species, not even as their sum.  The particle
  // module takes the total from this table (RootPpc).
  if (Has(pin, "particles", "ppc")) {
    Fail("<particles>/ppc = " + pin->GetString("particles", "ppc") + " cannot be given with " +
         (t.powerlaw ? std::string("ppc_per_species") : std::string("ppc_N")) + " (the new "
         "sampling keys; total " + std::to_string(t.ppc_total) + " per root cell): remove "
         "<particles>/ppc");
  }
  t.hp = ResolveHp(pin, t);
  return t;
}

}  // namespace

void SpeciesTable::SlotSpecies(const int k, int &s, int &kk) const {
  if (!new_init || equal_ppc) {
    s = k % nspecies;
    kk = k/nspecies;
    return;
  }
  int off = 0;
  for (s=0; s<nspecies; ++s) {
    if (k < off + ppc[s]) {kk = k - off; return;}
    off += ppc[s];
  }
  s = nspecies - 1; kk = 0;
}

std::vector<int> SpeciesTable::Offsets() const {
  std::vector<int> o(nspecies + 1, 0);
  for (int s=0; s<nspecies; ++s) o[s + 1] = o[s] + (new_init ? ppc[s] : 0);
  return o;
}

std::string SpeciesTable::Describe(const std::string &pre) const {
  std::ostringstream h;
  h << std::setprecision(17);
  h << pre << "nspecies = " << nspecies << "\n";
  h << pre << "stopping_times = " << tau_source << "\n";
  if (new_init) {
    h << pre << "abundance = " << abundance;
    if (abundance == "Z") h << " (column ratio, column_normalization = " << column_normalization << ")";
    else h << " (initial reference ratio rho_d/rho_g; the midplane ratio if stratified)";
    h << (powerlaw ? ", power law" : ", explicit") << "\n";
    if (powerlaw) h << pre << "mass_slope = " << mass_slope << "\n";
    h << pre << abundance << "_total = " << total << "\n";
    h << pre << "sampling = ppc_N per root cell, total " << ppc_total << " ("
      << (equal_ppc ? "equal: species interleaved as the legacy layout" : "species in blocks")
      << ")\n";
  } else {
    h << pre << "abundance = legacy (<problem>/eps_N or <dust>/dust_to_gas, as the problem "
      << "generator reads them; to be deprecated)\n";
    h << pre << "sampling = legacy <particles>/ppc shared by the species\n";
  }
  for (int s=0; s<nspecies; ++s) {
    h << pre << "species." << s << " = input_index " << (s + 1) << ", tau_s " << tau_s[s];
    if (omega_ref > 0.0) h << ", stokes " << tau_s[s]*omega_ref;
    if (new_init) h << ", " << abundance << " " << value[s] << ", ppc " << ppc[s];
    if (!hp.empty()) h << ", H_p " << hp[s];
    h << "\n";
  }
  return h.str();
}

// the restart record species_table: ASCII lines, values with 17 digits
//   species_table 1
//   nspecies N
//   stopping_times <source>          (tau_s_N | stokes_N | powerlaw (...) | taus_N (legacy))
//   omega_ref <Omega of the Stokes numbers, 0: none>
//   abundance epsilon | Z | legacy
//   column_normalization redraw | full_profile | domain | -
//   species <input index> tau_s <t> stokes <St | -> <epsilon | Z | value> <v | -> ppc <n | -> H_p <h | ->
std::string SpeciesTable::Record() const {
  std::ostringstream r;
  r << std::setprecision(17);
  r << "species_table 1\n";
  r << "nspecies " << nspecies << "\n";
  r << "stopping_times " << tau_source << "\n";
  r << "omega_ref " << omega_ref << "\n";
  r << "abundance " << abundance << "\n";
  r << "column_normalization " << (column_normalization.empty() ? "-" : column_normalization) << "\n";
  const std::string vk = new_init ? abundance : "value";
  for (int s=0; s<nspecies; ++s) {
    r << "species " << (s + 1) << " tau_s " << tau_s[s] << " stokes ";
    if (omega_ref > 0.0) {r << tau_s[s]*omega_ref;} else {r << "-";}
    r << " " << vk << " ";
    if (new_init) {r << value[s];} else {r << "-";}
    r << " ppc ";
    if (new_init) {r << ppc[s];} else {r << "-";}
    r << " H_p ";
    if (!hp.empty()) {r << hp[s];} else {r << "-";}
    r << "\n";
  }
  return r.str();
}

namespace {
struct RecordFields {
  std::vector<std::pair<std::string, std::string>> head;               // in file order
  std::vector<std::vector<std::pair<std::string, std::string>>> sp;    // per species
};
RecordFields ParseRecord(const std::string &text) {
  RecordFields f;
  std::istringstream in(text);
  for (std::string line; std::getline(in, line);) {
    std::istringstream l(line);
    std::string k;
    if (!(l >> k)) continue;
    if (k == "species") {
      int n = 0;
      l >> n;
      std::vector<std::pair<std::string, std::string>> row;
      for (std::string a, b; l >> a >> b;) row.emplace_back(a, b);
      f.sp.push_back(row);
    } else {
      std::string rest;
      std::getline(l, rest);
      const auto b = rest.find_first_not_of(' ');
      f.head.emplace_back(k, b == std::string::npos ? "" : rest.substr(b));
    }
  }
  return f;
}
std::string Field(const std::vector<std::pair<std::string, std::string>> &v, const std::string &k) {
  for (const auto &p : v) if (p.first == k) return p.second;
  return "(none)";
}
bool SameValue(const std::string &a, const std::string &b) {
  if (a == b) return true;
  char *ea = nullptr, *eb = nullptr;
  const double x = std::strtod(a.c_str(), &ea), y = std::strtod(b.c_str(), &eb);
  if (ea == a.c_str() || eb == b.c_str() || *ea != '\0' || *eb != '\0') return false;
  return std::fabs(x - y) <= 1.0e-12*std::max(std::fabs(x), std::fabs(y));
}
}  // namespace

std::string CompareSpeciesRecords(const std::string &file, const std::string &input) {
  const RecordFields a = ParseRecord(file), b = ParseRecord(input);
  std::ostringstream d;
  for (const char *k : {"nspecies", "abundance", "column_normalization", "omega_ref"}) {
    const std::string x = Field(a.head, k), y = Field(b.head, k);
    if (!SameValue(x, y)) {
      d << "  " << k << ": restart file " << x << ", input " << y << "\n";
    }
  }
  if (a.sp.size() != b.sp.size()) return d.str();
  for (std::size_t s=0; s<a.sp.size(); ++s) {
    const auto &ra = a.sp[s], &rb = b.sp[s];
    for (std::size_t q=0; q<std::max(ra.size(), rb.size()); ++q) {
      const std::string k = (q < ra.size()) ? ra[q].first : rb[q].first;
      const std::string x = Field(ra, k), y = Field(rb, k);
      if (!SameValue(x, y)) {
        d << "  species " << (s + 1) << " (species_id " << s << "): " << k << ": restart file " << x
          << ", input " << y << "\n";
      }
    }
  }
  return d.str();
}

double RootPpc(ParameterInput *pin) {
  if (pin->DoesBlockExist("dust")) {
    const SpeciesTable &t = Species(pin);
    if (t.new_init) return static_cast<double>(t.ppc_total);
  }
  return pin->GetOrAddReal("particles", "ppc", 1.0);   // legacy and other particle types
}

const SpeciesTable &Species(ParameterInput *pin) {
  static std::unique_ptr<SpeciesTable> table;
  static ParameterInput *owner = nullptr;
  if (!table || owner != pin) {
    table.reset(new SpeciesTable(Parse(pin)));
    owner = pin;
    if (global_variable::my_rank == 0 && table->NewAPI()) {
      std::cout << "# dust species:\n" << table->Describe("#   ") << std::flush;
    }
  }
  return *table;
}

void RequireLegacySpeciesInit(ParameterInput *pin, const char *pgen) {
  if (Species(pin).new_init) {
    Fail(std::string("problem generator ") + pgen + " does not implement the new abundance and "
         "sampling keys (epsilon_N / Z_N / ppc_N, power law); use its legacy keys");
  }
}

std::vector<double> SpeciesHp(ParameterInput *pin, const SpeciesTable &t, const double hgas,
                              const bool legacy_default) {
  const int n = t.nspecies;
  if (!t.hp.empty()) {
    if (Has(pin, "problem", "dust_hp")) {
      Fail("the legacy <problem>/dust_hp and the new H_p keys cannot be mixed");
    }
    return t.hp;
  }
  if (t.new_init && !legacy_default) {
    Fail("a stratified set-up with the new species keys needs <problem>/H_p_N, H_p or "
         "H_p_min with H_p_max (no default)");
  }
  const double h = pin->GetOrAddReal("problem", "dust_hp", hgas);   // legacy (to be deprecated)
  return std::vector<double>(n, h);
}

}  // namespace dust
