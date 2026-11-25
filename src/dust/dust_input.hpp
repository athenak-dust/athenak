#ifndef DUST_DUST_INPUT_HPP_
#define DUST_DUST_INPUT_HPP_
//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
//! \file dust_input.hpp
//! \brief Input keys of the dust module that the mesh and boundary classes read before the
//! dust module exists: <dust>/amr_pic (how the dust particles live on a refined mesh), the
//! notes for the keys that are no longer read, and the keys of retired options (RetiredKeys,
//! CheckRetiredKeys; retired-compat).

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "globals.hpp"
#include "parameter_input.hpp"

namespace dust {

// <dust>/amr_pic: refinable_cloud (default on a refined mesh) or instant_switch (point particles,
// TSC on the block that owns them, ghost deposits remapped to the leaves).  On a uniform mesh the
// point particles deposit by TSC (none); amr_pic = refinable_cloud selects the cloud path there
// too (exact TSC clouds, a reference with the coupling of the mixed-resolution runs).
enum class AmrPic {none, refinable_cloud, instant_switch};

// The method of <dust>/amr_pic on this mesh, without printing (the dust module prints the
// notes).  The key experimental_smr_transfer of the internal development branches is read as
// amr_pic (smooth_p2 was removed and stops the run).
inline AmrPic ResolveAmrPic(ParameterInput *pin, const bool multilevel) {
  auto fatal = [](const std::string &msg) {
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << msg << std::endl;
    std::exit(EXIT_FAILURE);
  };
  const bool given = pin->DoesParameterExist("dust", "amr_pic");
  const bool legacy = pin->DoesParameterExist("dust", "experimental_smr_transfer");
  std::string v = given ? pin->GetString("dust", "amr_pic") : "";
  // (older input and restart files name instant_switch owner_overlap)
  if (v == "owner_overlap") v = "instant_switch";
  if (given && v != "refinable_cloud" && v != "instant_switch") {
    fatal("<dust>/amr_pic = '" + v + "' not recognized (refinable_cloud or instant_switch)");
  }
  if (legacy) {
    std::string w = pin->GetString("dust", "experimental_smr_transfer");
    if (w == "owner_overlap") w = "instant_switch";
    if (w == "smooth_p2") {
      fatal("<dust>/experimental_smr_transfer = smooth_p2: the smooth_p2 transfer was removed "
            "(use <dust>/amr_pic = refinable_cloud or instant_switch)");
    }
    if (w != "refinable_cloud" && w != "instant_switch" && w != "none") {
      fatal("<dust>/experimental_smr_transfer = '" + w + "' not recognized");
    }
    if (given && w != v && !(w == "none" && !multilevel)) {
      fatal("<dust>/amr_pic = " + v + " and the old key experimental_smr_transfer = " + w +
            " disagree (drop the old key)");
    }
    if (!given) v = w;
  }
  if (v == "refinable_cloud") return AmrPic::refinable_cloud;
  if (v == "instant_switch") return multilevel ? AmrPic::instant_switch : AmrPic::none;
  if (v == "none") return AmrPic::none;
  return multilevel ? AmrPic::refinable_cloud : AmrPic::none;
}

// A key that is no longer read: one NOTE line (rank 0), the run goes on.
inline void NoteDroppedKey(ParameterInput *pin, const char *block, const char *key,
                           const char *why) {
  if (!pin->DoesParameterExist(block, key) || global_variable::my_rank != 0) return;
  std::cout << "# NOTE: <" << block << ">/" << key << " = " << pin->GetString(block, key)
            << " is no longer needed and is ignored (" << why << ")" << std::endl;
}

// retired-compat: the keys of retired options (the 2-D-only reference and diagnostic
// paths).  A key is judged by its value, where the value comes from and what it would
// have done in this configuration:
//  - the kept value (the default of the current code): ignored;
//  - another value that would have had no effect in this configuration in the retired code (a
//    cloud key of a point-particle run, a G key with G off, an R key with R off, a key that acted
//    only together with another retired option): ignored;
//  - another value that would have selected the retired path: FATAL with the replacement, whether
//    given in the input / on the command line or stored by a restart file as the run-time default
//    of the run that wrote it (that run then used the retired path).
// The ignored keys are listed by Finish() in at most two NOTE lines (rank 0).  The origin of a value
// is read from its comment: a restart file stores each run-time default with "# Default value added
// at run time", which a change of that key on the command line keeps.
class RetiredKeys {
 public:
  // date: the day of the retirement, printed in the messages (the default: the 2-D-only
  // options; the later date: the constants and fixed switches retired with them)
  RetiredKeys(ParameterInput *pin, const char *block, const char *date = "2026-11-24") :
      pin_(pin), block_(block), date_(date) {}
  bool Has(const char *key) const {return pin_->DoesParameterExist(block_, key);}
  std::string Get(const char *key) const {return pin_->GetString(block_, key);}
  bool Bool(const char *key) const {return pin_->GetBoolean(block_, key);}
  // the value carries the comment of a run-time default stored by a restart file
  bool StoredDefault(const char *key) const {
    for (auto &blk : pin_->block) {
      if (blk.block_name != block_) continue;
      const InputLine *pl = blk.GetPtrToLine(key);
      return pl != nullptr &&
             pl->param_comment.find("Default value added at run time") != std::string::npos;
    }
    return false;
  }
  // kept: the value is the current code's (the default); effective: the value would have selected
  // the retired path in this configuration; kept_value: the value to give instead; differs: the
  // kept value runs another scheme than the retired one (false: the same scheme)
  void Check(const char *key, const bool kept, const bool effective, const char *what,
             const char *instead, const char *kept_value, const bool differs = true) {
    if (!Has(key)) return;
    if (kept) {Default(key); return;}
    if (!effective) {Ignore(key); return;}
    std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
              << "<" << block_ << ">/" << key << " = " << Get(key) << " ("
              << (StoredDefault(key) ?
                  "stored in the restart file as the run-time default of the run that wrote it" :
                  "given in the input or on the command line, or in the input of the run that "
                  "wrote the restart file")
              << "): " << what << " was retired on " << date_ << ". " << instead << ". (A restart file "
              << "that carries this value continues only with " << block_ << "/" << key << "="
              << kept_value << " on the command line"
              << (differs ? "; the results then differ from the old run.)" : ".)") << std::endl;
    std::exit(EXIT_FAILURE);
  }
  // a key that acted only together with another retired option (which stops the run by itself)
  void Inert(const char *key) {
    if (Has(key)) {if (StoredDefault(key)) Default(key); else Ignore(key);}
  }
  void Finish() const {
    if (global_variable::my_rank != 0) return;
    if (!defaults_.empty()) {
      std::cout << "# NOTE: <" << block_ << "> keys of options retired on " << date_ << " at their "
                << "default values (or stored by the restart file as run-time defaults, which "
                << "a value given again on the command line does not change), ignored:"
                << defaults_ << std::endl;
    }
    if (!noeffect_.empty()) {
      std::cout << "# NOTE: <" << block_ << "> keys of options retired on " << date_ << " that have no "
                << "effect in this configuration (they had none before the retirement either), "
                << "ignored:" << noeffect_ << std::endl;
    }
  }

 private:
  void Default(const char *key) {defaults_ += std::string(" ") + key + "=" + Get(key);}
  void Ignore(const char *key) {
    noeffect_ += std::string(" ") + key + "=" + Get(key) +
                 (StoredDefault(key) ? "(stored default)" : "");
  }
  ParameterInput *pin_;
  const char *block_;
  const char *date_;
  std::string defaults_, noeffect_;
};

// retired-compat: the coupling check, at the start of the dust module's constructor (before the
// coupling is parsed).  coupling = hybrid always ran the hybrid task path; with
// hybrid_force_mode = pc2 that path ran PC2 on every cycle, the scheme of coupling = pc2.
inline void CheckRetiredCoupling(ParameterInput *pin) {
  RetiredKeys rk(pin, "dust");
  if (!rk.Has("coupling") || rk.Get("coupling") != "hybrid") return;
  const bool forced_pc2 = rk.Has("hybrid_force_mode") && rk.Get("hybrid_force_mode") == "pc2";
  rk.Check("coupling", false, true, forced_pc2 ?
           "the hybrid coupling (coupling = hybrid; here with hybrid_force_mode = pc2, which ran "
           "PC2 on every cycle)" :
           "the hybrid coupling with the automatic split backward-Euler (coupling = hybrid)",
           forced_pc2 ? "Continue with dust/coupling=pc2: the same scheme (PC2 on every cycle)" :
           "Use coupling = pc2 (with integrator = rk2) or imex (with imex2+)", "pc2",
           !forced_pc2);
}

// retired-compat: what the run would have used of the retired options (CheckRetiredKeys)
struct RetiredContext {
  bool clouds = false;     // refinable clouds (UsesRefinableClouds)
  bool split = false;      // the clouds' P1 split runs (cloud_split_enable)
  bool g_on = false;       // G runs (refinable clouds, cloud_merge)
  bool r_on = false;       // R runs (refinable clouds, cloud_resample)
  bool stencil = false;    // stage X (cloud_merge_stencil)
  bool budget = false;     // the pass budget (cloud_merge_budget > 0)
  int dim = 0;             // cloud_dim
  bool streaming_linear = false;   // <problem>/pgen_name = streaming_linear (built-in pgens)
  bool merge_by_time = false;      // G on its physical-time schedule (cloud_merge_schedule = time)
  bool resample_by_time = false;   // R on its physical-time schedule
  bool r_device = false;           // R with the device patch solves (cloud_resample_device)
  bool split_resample = false;     // the local R after a split pass (cloud_split_resample)
  bool split_mg = false;           // the split with cloud_split_velocity = mesh_gradient
  bool split_range = false;        // ... and its range bound (test_split_gradient_bound = range)
  bool shear3d = false;            // 3-D clouds with shear-periodic x1 faces
  bool multirank = false;          // more than one rank
  bool lb_weight = false;          // the cost-weighted load balance (load_balance_cloud_weight > 0)
  bool dc1 = false;                // the clouds' dc1 / dc2 (drag_solver)
  int launch_bounds = 0;           // the compiled DUST_CLOUD_LAUNCH_BOUNDS
  bool qr_columns = false;         // the column-QR reference build (DUST_R_QR_COLUMNS = 1)
};

// retired-compat: every other retired key in one place, called once in the dust module's
// constructor after the representation, G, R and the coupling are known.  At the release that
// drops the backward compatibility these functions, the class RetiredKeys and their calls are
// deleted (grep retired-compat).
inline void CheckRetiredKeys(ParameterInput *pin, const RetiredContext &c) {
  {
    RetiredKeys rk(pin, "dust");
    // the host split and merge, and the virtual finest-grid table they read
    rk.Check("cloud_split_device", rk.Has("cloud_split_device") && rk.Bool("cloud_split_device"),
             c.clouds && c.split, "the host P1 split (cloud_split_device = false)",
             "The P1 split runs in Kokkos on every backend: drop the key", "true");
    rk.Check("cloud_merge_device", rk.Has("cloud_merge_device") && rk.Bool("cloud_merge_device"),
             c.g_on, "the host merge pass (cloud_merge_device = false)",
             "G runs in Kokkos on every backend: drop the key", "true");
    // (the old parser accepted only auto, false and true, in every configuration)
    if (rk.Has("cloud_table") && rk.Get("cloud_table") != "auto" &&
        rk.Get("cloud_table") != "false" && rk.Get("cloud_table") != "true") {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dust>/cloud_table = '" << rk.Get("cloud_table") << "' not recognized (auto, "
                << "false or true; the key of the retired finest-grid table: drop "
                << "it)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rk.Check("cloud_table", rk.Has("cloud_table") && rk.Get("cloud_table") != "true", c.clouds,
             "the virtual finest-grid table (cloud_table = true)",
             "The clouds use the actual leaves only: drop the key", "auto");
    // the position order of G and R (diagnostic), the X budget and the X protection (2-D)
    rk.Check("cloud_merge_position_order", rk.Has("cloud_merge_position_order") &&
             !rk.Bool("cloud_merge_position_order"), c.g_on && c.dim == 2,
             "the position order of the G walk (cloud_merge_position_order = true)",
             "G walks the pieces in particle order: drop the key", "false");
    // (cloud_resample_order is a live key, array or tag: only the retired value is judged here)
    if (rk.Has("cloud_resample_order") && rk.Get("cloud_resample_order") == "position")
      rk.Check("cloud_resample_order", false, c.r_on,
             "the position order of the R pool (cloud_resample_order = position)",
             "Drop the key (the array order), or use <dust>/test_canonical_order = resample_tag "
             "(PTAG order, independent of the storage order)", "array");
    rk.Check("cloud_merge_budget_x", rk.Has("cloud_merge_budget_x") &&
             !rk.Bool("cloud_merge_budget_x"), c.g_on && c.stencil && c.budget,
             "stage X under the pass budget (cloud_merge_budget_x = true)",
             "Stage X re-merges every complete family (cloud_merge_stencil, careful_merge): drop "
             "the key", "false");
    rk.Check("cloud_merge_xprotect", rk.Has("cloud_merge_xprotect") &&
             !rk.Bool("cloud_merge_xprotect"), c.g_on && c.stencil,
             "the X protection (cloud_merge_xprotect = true)",
             "Stage X re-merges every complete family (cloud_merge_stencil, careful_merge): drop "
             "the key", "false");
    // cloud_gather_order = 1 (the cell values)
    rk.Check("cloud_gather_order", rk.Has("cloud_gather_order") &&
             pin->GetInteger("dust", "cloud_gather_order") == 2, c.clouds,
             "the piecewise-constant gather (cloud_gather_order = 1)",
             "The clouds gather the linear reconstruction of the leaves (order 2): drop the key",
             "2");
    // the 2-D tent pass of G and its two keys (they acted only in the tent pass and the host
    // merge: ignored whatever their value; first_batch / framework inputs set lobe_wait = 1 with
    // the quantum pass)
    rk.Check("cloud_merge_mode", rk.Has("cloud_merge_mode") &&
             rk.Get("cloud_merge_mode") == "quantum", c.g_on,
             "the tent merge pass (cloud_merge_mode = tent)", "G is the quantum pass: drop the key",
             "quantum");
    rk.Inert("cloud_merge_nc");
    rk.Inert("cloud_merge_lobe_wait");
    // the keys of the hybrid coupling's switch (coupling = hybrid stops in CheckRetiredCoupling):
    // they acted only there (an imex restart file carries hybrid_force_mode = auto and the four
    // thresholds, a pc2 file the thresholds)
    for (const char *k : {"hybrid_force_mode", "hybrid_enter_zeta", "hybrid_enter_chi",
                          "hybrid_exit_zeta", "hybrid_exit_chi"}) {
      rk.Inert(k);
    }
    rk.Finish();
  }
  {
    // the analytic gather of streaming_linear (any representation: the point particles stopped
    // on it before); no other pgen read these keys
    RetiredKeys rp(pin, "problem");
    const bool ag = rp.Has("analytic_gather") && rp.Bool("analytic_gather");
    rp.Check("analytic_gather", !ag, c.streaming_linear,
             "the analytic gather (<problem>/analytic_gather = true)",
             "The clouds gather the gas of the mesh: drop the key", "false");
    rp.Check("analytic_gather_point", rp.Has("analytic_gather_point") &&
             !rp.Bool("analytic_gather_point"), ag && c.streaming_linear,
             "the analytic gather's point variant (<problem>/analytic_gather_point = true)",
             "Drop the key", "false");
    rp.Check("analytic_gather_centroid", rp.Has("analytic_gather_centroid") &&
             !rp.Bool("analytic_gather_centroid"), ag && c.streaming_linear,
             "the analytic gather's centroid variant (<problem>/analytic_gather_centroid = true)",
             "Drop the key", "false");
    rp.Finish();
  }
  {
    // the retired G and R constants (cloud_merge_K 4, cloud_merge_smax 1,
    // cloud_merge_min_mass 0.5, cloud_merge_budget_evals 4, cloud_absorb_floor 0.1,
    // cloud_merge_t0 0, cloud_resample_t0 0, cloud_resample_patch 2, cloud_resample_patch_cells
    // 4, cloud_resample_patch_max 4096, cloud_split_slack 0.02, cloud_split_dispersion_max 0.5,
    // cloud_split_resample_frac 0.05), cloud_transition (2), the middle bucket of the device R
    // (cloud_resample_device_mid, _mid_team) and the column QR (cloud_resample_qr_rows = true),
    // the count rule and the span / margin of the split velocity bound (test_split_gradient_nmin,
    // _count, _range_kind, _range_margin; the value count of test_split_gradient_bound stops in
    // ReadSplitVelocityKeys), the fixed switches (cloud_merge_exact_ts, cloud_merge_budget_thin,
    // cloud_halo_strips, cloud_order_keep, cloud_shear_list_cache, load_balance_octant_count),
    // amr_shear_face_limit, cloud_launch_bounds and test_cloud_dc1_blend.  (<shearing_box>/
    // test_pull_lagged_xslope: ShearingBox; <gravity>/dust_x3_assignment: SelfGravity.)
    RetiredKeys rk(pin, "dust", "2026-11-25");
    // a numeric key at its kept value (compared as numbers: 0.50 == 0.5)
    auto num = [&](const char *key, const double v) {
      return rk.Has(key) && pin->GetReal("dust", key) == static_cast<Real>(v);
    };
    auto boolean = [&](const char *key, const bool v) {return rk.Has(key) && rk.Bool(key) == v;};
    auto str = [&](const char *key, const char *v) {return rk.Has(key) && rk.Get(key) == v;};
    const char *ck = "The constant is used: drop the key";
    // G
    rk.Check("cloud_merge_K", num("cloud_merge_K", 4.0), c.g_on,
             "the number of open G groups per cell walk (cloud_merge_K)",
             "The constant 4 is used: drop the key", "4");
    rk.Check("cloud_merge_smax", num("cloud_merge_smax", 1.0), c.g_on,
             "the position extent of a G group (cloud_merge_smax)",
             "The constant 1 (target cells) is used: drop the key", "1.0");
    rk.Check("cloud_merge_min_mass", num("cloud_merge_min_mass", 0.5), c.g_on,
             "the smallest committed G mass (cloud_merge_min_mass)",
             "The constant 0.5 (of m_ref) is used: drop the key", "0.5");
    rk.Check("cloud_merge_budget_evals", num("cloud_merge_budget_evals", 4.0), c.g_on && c.budget,
             "the evaluation rounds of the G pass budget (cloud_merge_budget_evals)",
             "The constant 4 is used: drop the key", "4");
    rk.Check("cloud_absorb_floor", num("cloud_absorb_floor", 0.1), c.g_on && c.budget,
             "the mass floor of the G pass budget (cloud_absorb_floor)",
             "The constant 0.1 (root clouds per root cell) is used: drop the key", "0.1");
    rk.Check("cloud_merge_t0", num("cloud_merge_t0", 0.0), c.merge_by_time,
             "the time origin of the G schedule (cloud_merge_t0)",
             "The G passes are planned at k cloud_merge_dt (origin 0): drop the key", "0.0");
    rk.Check("cloud_merge_exact_ts", boolean("cloud_merge_exact_ts", true), c.g_on,
             "the mass-weighted ts of merged clouds (cloud_merge_exact_ts = false)",
             "Merged clouds keep their members' common ts: drop the key", "true");
    rk.Check("cloud_merge_budget_thin", boolean("cloud_merge_budget_thin", false),
             c.g_on && c.budget && c.dim == 3,
             "the thinning G budget (cloud_merge_budget_thin = true)",
             "The budget rejects every group on a violating leaf: drop the key", "false");
    // R
    rk.Check("cloud_resample_t0", num("cloud_resample_t0", 0.0), c.resample_by_time,
             "the time origin of the R schedule (cloud_resample_t0)",
             "R is planned at k cloud_resample_dt (origin 0): drop the key", "0.0");
    rk.Check("cloud_resample_patch", num("cloud_resample_patch", 2.0), c.r_on,
             "the R patch width in root cells (cloud_resample_patch)",
             "The constant 2 is used: drop the key", "2");
    rk.Check("cloud_resample_patch_cells", num("cloud_resample_patch_cells", 4.0), c.r_on,
             "the cap of the R patch width in block cells (cloud_resample_patch_cells)",
             "The constant 4 is used: drop the key", "4");
    rk.Check("cloud_resample_patch_max", num("cloud_resample_patch_max", 4096.0),
             c.r_on && c.dim == 3, "the sub-patch size of crowded R patches (cloud_resample_patch_max)",
             "The constant 4096 is used: drop the key", "4096");
    // (the middle bucket existed only for mid > 64 clouds; its team key only then)
    const bool mid = rk.Has("cloud_resample_device_mid") &&
                     pin->GetReal("dust", "cloud_resample_device_mid") > 64.0;
    rk.Check("cloud_resample_device_mid", num("cloud_resample_device_mid", 0.0), c.r_device && mid,
             "the middle bucket of the device R solves (cloud_resample_device_mid)",
             "The device R solves have two buckets (at most 64 clouds, more): drop the key", "0",
             false);
    rk.Check("cloud_resample_device_mid_team", num("cloud_resample_device_mid_team", 128.0),
             c.r_device && mid, "the team size of the middle bucket (cloud_resample_device_mid_team)",
             "The device R solves have two buckets: drop the key", "128", false);
    // (judged by the compiled algorithm: the production build shares the rows (true; false
    // needed the column build), the reference build -DDUST_R_QR_COLUMNS=1 now runs the column QR
    // in every device solve (false; true ran the rows there))
    rk.Check("cloud_resample_qr_rows", boolean("cloud_resample_qr_rows", !c.qr_columns),
             c.r_device, c.qr_columns ?
             "the row-parallel QR in the column reference build (cloud_resample_qr_rows = true)" :
             "the column-by-column QR of the device R solves (cloud_resample_qr_rows = false)",
             c.qr_columns ? "This build runs the column QR in every device solve: drop the key" :
             "The production build shares the rows; the column QR is the reference build "
             "-DDUST_R_QR_COLUMNS=1: drop the key", c.qr_columns ? "false" : "true");
    rk.Check("cloud_split_resample_frac", num("cloud_split_resample_frac", 0.05), c.split_resample,
             "the trigger fraction of the R after a split (cloud_split_resample_frac)",
             "The constant 0.05 is used: drop the key", "0.05");
    // the split and the clouds
    rk.Check("cloud_split_slack", num("cloud_split_slack", 0.02), c.split,
             "the slack of the split's support test (cloud_split_slack)",
             "The constant 0.02 is used: drop the key", "0.02");
    rk.Check("cloud_split_dispersion_max", num("cloud_split_dispersion_max", 0.5), c.split_mg,
             "the single-stream threshold of the mesh_gradient split (cloud_split_dispersion_max)",
             "The constant 0.5 is used: drop the key", "0.5");
    rk.Check("cloud_transition", num("cloud_transition", 2.0), c.clouds,
             "the transition clouds' modes 0 and 1 (cloud_transition)",
             "The transition clouds gather with the lobed weights (mode 2): drop the key", "2");
    // (the count rule acted only with test_split_gradient_bound = count, which stops by itself)
    rk.Check("test_split_gradient_nmin", num("test_split_gradient_nmin", 8.0), false, "", ck, "8");
    rk.Check("test_split_gradient_count", str("test_split_gradient_count", "cell"), false, "", ck,
             "cell");
    rk.Check("test_split_gradient_range_kind", str("test_split_gradient_range_kind", "box"),
             c.split_range, "the span form of the split velocity bound "
             "(test_split_gradient_range_kind = span)",
             "The range bound has the box form: drop the key", "box");
    rk.Check("test_split_gradient_range_margin", num("test_split_gradient_range_margin", 0.5),
             c.split_range, "the margin of the split velocity bound "
             "(test_split_gradient_range_margin)", "The constant 0.5 is used: drop the key", "0.5");
    // fixed switches
    // (cloud_halo_strips and cloud_shear_list_cache = false gave the same results: never
    // effective)
    rk.Check("cloud_halo_strips", boolean("cloud_halo_strips", true), false,
             "the whole-block dust halo (cloud_halo_strips = false)",
             "The halo exchanges the reachable cells: drop the key", "true", false);
    rk.Check("cloud_order_keep", boolean("cloud_order_keep", true), c.clouds,
             "the storage order after a migration (cloud_order_keep = false)",
             "The cloud kernels keep the cell-sorted index: drop the key", "true");
    rk.Check("cloud_shear_list_cache", boolean("cloud_shear_list_cache", true), false,
             "the uncached shear-image lists (cloud_shear_list_cache = false)",
             "The lists are cached (the same results): drop the key", "true", false);
    rk.Check("load_balance_octant_count", boolean("load_balance_octant_count", true),
             c.multirank && c.lb_weight,
             "the per-block cloud count of the load balance (load_balance_octant_count = false)",
             "The clouds are counted per octant: drop the key", "true");
    rk.Check("test_cloud_dc1_blend", boolean("test_cloud_dc1_blend", false), c.dc1,
             "the gather blend of the clouds' dc1 (test_cloud_dc1_blend = true)",
             "dc1 runs without the gather blend: drop the key", "false");
    // (only on existed; off was ignored with a NOTE; another value stopped the run)
    if (rk.Has("amr_shear_face_limit") && !str("amr_shear_face_limit", "on") &&
        !str("amr_shear_face_limit", "off")) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dust>/amr_shear_face_limit = " << rk.Get("amr_shear_face_limit")
                << " not recognized (on or off; the key retired: the shear-periodic "
                << "x1 faces always follow the 2:1 rule, drop it)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rk.Check("amr_shear_face_limit", str("amr_shear_face_limit", "on"), false, "", ck, "on");
    // (a compile-time choice: only the compiled value was accepted, by refinable clouds)
    if (rk.Has("cloud_launch_bounds") && c.clouds &&
        pin->GetInteger("dust", "cloud_launch_bounds") != c.launch_bounds) {
      std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl
                << "<dust>/cloud_launch_bounds = " << rk.Get("cloud_launch_bounds")
                << ", but this binary was built with DUST_CLOUD_LAUNCH_BOUNDS = "
                << c.launch_bounds << " (the launch bound of the cloud kernels is a compile-time "
                << "choice: rebuild with -DDUST_CLOUD_LAUNCH_BOUNDS=<n> in CMAKE_CXX_FLAGS; the "
                << "key retired: drop it)" << std::endl;
      std::exit(EXIT_FAILURE);
    }
    rk.Check("cloud_launch_bounds", rk.Has("cloud_launch_bounds") &&
             pin->GetInteger("dust", "cloud_launch_bounds") == c.launch_bounds, false, "", ck,
             "the compiled value");
    rk.Finish();
  }
}

// <dust>/amr_density_trigger: the dust-density refinement trigger.  Default:
// on when amr_rho_refine or amr_rho_refine_levels is given; false turns it off.  The value is
// added to the parameters, so a restart keeps the value of the run that wrote the file.
inline bool GetAmrDensityTrigger(ParameterInput *pin) {
  return pin->GetOrAddBoolean("dust", "amr_density_trigger",
                              pin->DoesParameterExist("dust", "amr_rho_refine") ||
                              pin->DoesParameterExist("dust", "amr_rho_refine_levels"));
}

// A hidden test_ key: read only when given, never added to the parameters (nor to the restart
// header), so that the parameter dump of the default runs does not list it.
inline bool TestKeyBool(ParameterInput *pin, const char *block, const char *key, bool def) {
  return pin->DoesParameterExist(block, key) ? pin->GetBoolean(block, key) : def;
}
inline int TestKeyInt(ParameterInput *pin, const char *block, const char *key, int def) {
  return pin->DoesParameterExist(block, key) ? pin->GetInteger(block, key) : def;
}
inline Real TestKeyReal(ParameterInput *pin, const char *block, const char *key, Real def) {
  return pin->DoesParameterExist(block, key) ? pin->GetReal(block, key) : def;
}
inline std::string TestKeyString(ParameterInput *pin, const char *block, const char *key,
                                 const std::string &def) {
  return pin->DoesParameterExist(block, key) ? pin->GetString(block, key) : def;
}

[[noreturn]] inline void InputFatal(const std::string &msg) {
  std::cout << "### FATAL ERROR in " << __FILE__ << " at line " << __LINE__ << std::endl << msg
            << std::endl;
  std::exit(EXIT_FAILURE);
}

// a boolean as ParameterInput::GetBoolean reads it: a leading 0 or 1 as an integer, else true /
// false in any case (anything else reads as false)
inline bool AthenaBool(const std::string &v) {
  if (v.compare(0, 1, "0") == 0 || v.compare(0, 1, "1") == 0) return std::atoi(v.c_str()) != 0;
  std::string l = v;
  for (char &c : l) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return l == "true";
}
inline bool LooksBool(const std::string &v) {
  std::string l = v;
  for (char &c : l) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return l == "true" || l == "false" || l == "0" || l == "1";
}

// the shortest of 15 / 17 significant digits that reads back as the same number
inline std::string RealWord(const Real x) {
  for (const int p : {15, 17}) {
    std::ostringstream s;
    s << std::setprecision(p) << x;
    if (static_cast<Real>(std::strtod(s.str().c_str(), nullptr)) == x || p == 17) return s.str();
  }
  return "";
}

inline std::vector<std::string> SplitWords(const std::string &s) {
  std::vector<std::string> w;
  std::istringstream is(s);
  std::string t;
  while (is >> t) w.push_back(t);
  return w;
}

// <dust>/test_resample_after_regrid (hidden): the resampling after the regrids (default off: the
// planned G and R cover them).  Value: off | local | global, then optionally the word derefine
// (also after a regrid that only deleted blocks) and a number dt > 0 (the regrids of dt / omega0
// after the first one are handled together).  local resamples the blocks the regrid made and
// their neighbors, global every block.  Example: test_resample_after_regrid = local derefine 0.05.
// It replaces cloud_resample_after_regrid, cloud_resample_event_local,
// cloud_resample_after_derefine and cloud_event_dt_omega.
struct ResampleAfterRegrid {
  bool on = false, local = true, derefine = false;
  Real dt_omega = 0.0;
  bool operator==(const ResampleAfterRegrid &o) const {
    return on == o.on && (!on || (local == o.local && derefine == o.derefine &&
                                  dt_omega == o.dt_omega));
  }
  std::string Str() const {
    if (!on) return "off";
    std::ostringstream s;
    s << (local ? "local" : "global");
    if (derefine) s << " derefine";
    if (dt_omega > 0.0) s << " " << RealWord(dt_omega);
    return s.str();
  }
  static ResampleAfterRegrid Parse(const std::string &v) {
    ResampleAfterRegrid r;
    const std::vector<std::string> w = SplitWords(v);
    const std::string bad = "<dust>/test_resample_after_regrid = '" + v + "' not recognized "
        "(off, or local / global followed optionally by derefine and a number dt > 0)";
    if (w.empty()) InputFatal(bad);
    if (w[0] == "off" || w[0] == "false") {
      if (w.size() > 1) InputFatal(bad);
      return r;
    }
    if (w[0] != "local" && w[0] != "global") InputFatal(bad);
    r.on = true;
    r.local = (w[0] == "local");
    for (std::size_t i = 1; i < w.size(); ++i) {
      if (w[i] == "derefine") {r.derefine = true; continue;}
      char *end = nullptr;
      const double d = std::strtod(w[i].c_str(), &end);
      if (end == w[i].c_str() || *end != '\0' || !(d > 0.0)) InputFatal(bad);
      r.dt_omega = static_cast<Real>(d);
    }
    return r;
  }
};

// <dust>/test_merge_after_derefine (hidden): G every `interval` cycles during dt / omega0 after
// each regrid that deleted blocks (default off).  Value: off | dt [interval] (interval default 50).
// It replaces cloud_merge_burst_dt_omega and cloud_merge_burst_interval.
struct MergeAfterDerefine {
  Real dt_omega = 0.0;
  int interval = 50;
  bool operator==(const MergeAfterDerefine &o) const {
    return dt_omega == o.dt_omega && (!(dt_omega > 0.0) || interval == o.interval);
  }
  std::string Str() const {
    if (!(dt_omega > 0.0)) return "off";
    std::ostringstream s;
    s << RealWord(dt_omega) << " " << interval;
    return s.str();
  }
  static MergeAfterDerefine Parse(const std::string &v) {
    MergeAfterDerefine m;
    const std::vector<std::string> w = SplitWords(v);
    const std::string bad = "<dust>/test_merge_after_derefine = '" + v + "' not recognized "
        "(off, or dt > 0 followed optionally by an interval >= 1)";
    if (w.empty() || w.size() > 2) InputFatal(bad);
    if (w[0] == "off" || w[0] == "false" || w[0] == "0") {
      if (w.size() > 1) InputFatal(bad);
      return m;
    }
    char *end = nullptr;
    const double d = std::strtod(w[0].c_str(), &end);
    if (end == w[0].c_str() || *end != '\0' || !(d > 0.0)) InputFatal(bad);
    m.dt_omega = static_cast<Real>(d);
    if (w.size() == 2) {
      const long n = std::strtol(w[1].c_str(), &end, 10);
      if (end == w[1].c_str() || *end != '\0' || n < 1 || n > std::numeric_limits<int>::max()) {
        InputFatal(bad);
      }
      m.interval = static_cast<int>(n);
    }
    return m;
  }
};

// <dust>/test_canonical_order (hidden, diagnostic): off (default), or the words resample_tag (the
// R pool in PTAG order, independent of the storage order) and / or sort_random (the cloud sort
// applies a random permutation).  It replaces cloud_resample_order = tag and cloud_sort_order =
// random (the defaults array and linear are fixed).
struct CanonicalOrder {
  bool resample_tag = false, sort_random = false;
  bool operator==(const CanonicalOrder &o) const {
    return resample_tag == o.resample_tag && sort_random == o.sort_random;
  }
  std::string Str() const {
    if (!resample_tag && !sort_random) return "off";
    if (resample_tag && sort_random) return "resample_tag sort_random";
    return resample_tag ? "resample_tag" : "sort_random";
  }
  static CanonicalOrder Parse(const std::string &v) {
    CanonicalOrder c;
    const std::vector<std::string> w = SplitWords(v);
    const std::string bad = "<dust>/test_canonical_order = '" + v + "' not recognized (off, or "
        "resample_tag and / or sort_random)";
    if (w.empty()) InputFatal(bad);
    if (w.size() == 1 && (w[0] == "off" || w[0] == "false")) return c;
    for (const auto &t : w) {
      if (t == "resample_tag") c.resample_tag = true;
      else if (t == "sort_random") c.sort_random = true;
      else InputFatal(bad);
    }
    return c;
  }
};

// retired-compat: the members of the merged groups (old key, value) applied to the new value
inline Real MergedReal(const std::string &k, const std::string &v) {
  char *e = nullptr;
  const double d = std::strtod(v.c_str(), &e);
  if (e == v.c_str() || *e != '\0') InputFatal("<dust>/" + k + " = " + v + " is not a number");
  return static_cast<Real>(d);
}
inline void ApplyResampleAfterRegrid(ResampleAfterRegrid &r, const std::string &k,
                                     const std::string &v) {
  if (k == "cloud_resample_after_regrid") r.on = AthenaBool(v);
  if (k == "cloud_resample_event_local") r.local = AthenaBool(v);
  if (k == "cloud_resample_after_derefine") r.derefine = AthenaBool(v);
  if (k == "cloud_event_dt_omega") r.dt_omega = std::fmax(0.0, MergedReal(k, v));
}
inline void ApplyMergeAfterDerefine(MergeAfterDerefine &m, const std::string &k,
                                    const std::string &v) {
  if (k == "cloud_merge_burst_dt_omega") m.dt_omega = std::fmax(0.0, MergedReal(k, v));
  if (k == "cloud_merge_burst_interval") {
    char *e = nullptr;
    const long long n = std::strtoll(v.c_str(), &e, 10);
    if (e == v.c_str() || *e != '\0' || n < 1 || n > std::numeric_limits<int>::max()) {
      InputFatal("<dust>/cloud_merge_burst_interval = " + v + " must be an integer >= 1");
    }
    m.interval = static_cast<int>(n);
  }
}
inline void ApplyCanonicalOrder(CanonicalOrder &c, const std::string &k, const std::string &v) {
  if (k == "cloud_resample_order") {
    if (v != "array" && v != "tag" && v != "position") {
      InputFatal("<dust>/cloud_resample_order must be array or tag (now <dust>/"
                 "test_canonical_order = resample_tag)");
    }
    c.resample_tag = (v == "tag");
  }
  if (k == "cloud_sort_order") {
    if (v != "linear" && v != "random") {
      InputFatal("<dust>/cloud_sort_order must be linear or random (now <dust>/"
                 "test_canonical_order = sort_random)");
    }
    c.sort_random = (v == "random");
  }
}

// retired-compat: the keys renamed to the hidden test_<old name> (block, old name)
inline const std::vector<std::pair<const char *, const char *>> &RenamedKeyList() {
  static const std::vector<std::pair<const char *, const char *>> list = {
      // research controls of the split and R
      {"dust", "cloud_resample_patch_split"}, {"dust", "cloud_split_overlap"},
      {"dust", "cloud_split_everywhere"}, {"dust", "cloud_virtual_levels"},
      {"dust", "cloud_split_buffer"}, {"dust", "const_accel_phi"},
      {"dust", "allow_cfl_above_third_3d"},
      // diagnostics
      {"dust", "drag_diagnostic_interval"}, {"dust", "cloud_split_velocity_check"},
      {"dust", "cloud_resample_xv_audit"}, {"dust", "amr_profile"}, {"dust", "load_balance_log"},
      {"gravity", "reuse_check"}, {"shearing_box", "report_timing"}, {"hydro", "probe_ncycle"},
      // switches whose default is right (the hidden key turns them off)
      {"dust", "pc2_feedback_damping"}, {"dust", "imex_consistent_gas"},
      {"dust", "cloud_merge_alpha_leaf"}, {"dust", "cloud_split_resample"},
      {"dust", "cloud_split_enable"}, {"dust", "cloud_resample_rank_screen"},
      {"dust", "cloud_resample_xv_rows"}, {"dust", "load_balance_minmax"},
      {"dust", "amr_rho_kernel_above_split_max_level"}};
  return list;
}

// retired-compat: the (block, key) set on the command line (RenameCommandLineKeys), the most
// explicit origin of a value
inline std::set<std::pair<std::string, std::string>> &CommandLineKeys() {
  static std::set<std::pair<std::string, std::string>> s;
  return s;
}

// retired-compat: the keys renamed to hidden test_ keys and the keys merged into one.  Called
// once in main() after the input / restart header and the command line
// are read, before any module reads its keys.  An old name (in an input, on the command line or
// in a restart header) is mapped onto the new key, keeping its value and comment, and listed in one
// NOTE line; old and new name both given with different values stop the run (unless the old one
// is a run-time default stored by a restart file: the new value then wins).  The old lines are
// removed, so that new restart files carry the new names only.
class RenamedKeys {
 public:
  explicit RenamedKeys(ParameterInput *pin) : pin_(pin) {}
  InputLine *Line(const char *block, const char *key) {
    for (auto &blk : pin_->block) {
      if (blk.block_name == block) return blk.GetPtrToLine(key);
    }
    return nullptr;
  }
  bool Stored(const char *block, const char *key) {
    InputLine *pl = Line(block, key);
    return pl != nullptr &&
           pl->param_comment.find("Default value added at run time") != std::string::npos;
  }
  // the parameter dump pads names and values to the longest of the block
  void Widen(const char *block, const std::string &name, const std::string &value) {
    for (auto &blk : pin_->block) {
      if (blk.block_name != block) continue;
      if (blk.max_len_parname < name.length()) blk.max_len_parname = name.length();
      if (blk.max_len_parvalue < value.length()) blk.max_len_parvalue = value.length();
    }
  }
  void Erase(const char *block, const char *key) {
    for (auto &blk : pin_->block) {
      if (blk.block_name != block) continue;
      blk.line.remove_if([key](const InputLine &l) {return l.param_name == key;});
    }
  }
  static bool Same(const std::string &a, const std::string &b) {
    if (a == b) return true;
    auto as_bool = [](const std::string &s, int &v) {
      if (!LooksBool(s)) return false;
      v = AthenaBool(s) ? 1 : 0;
      return true;
    };
    int ia, ib;
    if (as_bool(a, ia) && as_bool(b, ib)) return ia == ib;
    char *ea = nullptr, *eb = nullptr;
    const double da = std::strtod(a.c_str(), &ea), db = std::strtod(b.c_str(), &eb);
    if (ea != a.c_str() && *ea == '\0' && eb != b.c_str() && *eb == '\0') return da == db;
    return SplitWords(a) == SplitWords(b);
  }
  [[noreturn]] void Conflict(const char *block, const std::string &olds, const char *neu) {
    InputFatal(std::string("<") + block + ">/" + olds + " and <" + block + ">/" + neu + " = " +
               pin_->GetString(block, neu) + " are both given with different values: the old "
               "key(s) were renamed to the hidden key " + neu + " (same meaning). "
               "Give " + neu + " only (to change the value of a restart file that carries the old "
               "key, give the old key on the command line).");
  }
  // how explicit a value is: 2 set on the command line, 1 given in an input (or the input of the
  // run that wrote the restart file), 0 a run-time default stored by a restart file
  int Rank(const char *block, const char *key) {
    if (CommandLineKeys().count({block, key}) != 0) return 2;
    return Stored(block, key) ? 0 : 1;
  }
  // one key renamed: the old line takes the new name (value and comment kept).  Both present with
  // different values: the more explicit one wins; equally explicit (and not stored defaults):
  // FATAL
  void Rename(const char *block, const char *old, const char *neu) {
    InputLine *po = Line(block, old);
    if (po == nullptr) return;
    if (pin_->DoesParameterExist(block, neu)) {
      const std::string vo = po->param_value;
      if (!Same(vo, pin_->GetString(block, neu))) {
        const int ro = Rank(block, old), rn = Rank(block, neu);
        if (ro == rn && ro > 0) Conflict(block, std::string(old) + " = " + vo, neu);
        if (ro > rn) {
          InputLine *pn = Line(block, neu);
          pn->param_value = vo;
          pn->param_comment = po->param_comment;
          Widen(block, neu, vo);
        }
      }
      Erase(block, old);
    } else {
      po->param_name = neu;
      Widen(block, neu, po->param_value);
      if (CommandLineKeys().count({block, old}) != 0) CommandLineKeys().insert({block, neu});
    }
    note_ += std::string(" <") + block + ">/" + old + "->" + neu;
  }
  // a group merged into one key.  apply(T &, old key, value) sets the member of an old key.  The
  // new value starts from the given new key (or the default); each old key present overrides its
  // member when it is more explicit than the new key, agrees, or is less explicit (then the new
  // key's member wins); an old key as explicit as the new one with another member value: FATAL.
  // The default is not written.
  template <class T, class F>
  void Merge(const char *block, const std::vector<const char *> &olds, const char *neu,
             const F &apply) {
    std::string list;
    for (const char *k : olds) {
      if (!pin_->DoesParameterExist(block, k)) continue;
      list += std::string(list.empty() ? "" : ", ") + k + " = " + pin_->GetString(block, k);
    }
    if (list.empty()) return;
    const bool given = pin_->DoesParameterExist(block, neu);
    const int rn = given ? Rank(block, neu) : -1;
    T t = given ? T::Parse(pin_->GetString(block, neu)) : T();
    for (const char *k : olds) {
      if (!pin_->DoesParameterExist(block, k)) continue;
      T probe = t;
      apply(probe, k, pin_->GetString(block, k));
      if (probe == t) continue;
      const int ro = Rank(block, k);
      if (ro > rn) {
        t = probe;
      } else if (ro == rn && ro > 0) {
        Conflict(block, std::string(k) + " = " + pin_->GetString(block, k), neu);
      }
    }
    if (given) {
      if (!(T::Parse(pin_->GetString(block, neu)) == t)) pin_->SetString(block, neu, t.Str());
    } else if (!(t == T())) {
      pin_->SetString(block, neu, t.Str());
    }
    for (const char *k : olds) Erase(block, k);
    note_ += std::string(" <") + block + ">/{" + list + "}->" + neu + "=" + t.Str();
  }
  void Finish() const {
    if (global_variable::my_rank != 0 || note_.empty()) return;
    std::cout << "# NOTE: keys renamed (hidden test_ keys, same meaning; the restart "
              << "files written from now on carry the new names), read as:" << note_ << std::endl;
  }

 private:
  ParameterInput *pin_;
  std::string note_;
};

// retired-compat: the ATHENA_* environment switches retired, judged by their value
// as the old code read it.  The former paths (*_OLD and the like) are gone: a NOTE (rank 0) if
// one is set.  The others became hidden input keys: a value that switched something on stops the
// run with the key (so that a job script does not lose its setting silently); a value that left
// the default draws the NOTE.  Called in main() once MPI is up, for every run.
inline void CheckRetiredEnvironment() {
  static const char *const old_paths[] = {
      "ATHENA_DEP_PREFETCH_OLD", "ATHENA_DUST_GATHER_ALLOC", "ATHENA_DUST_GATHER_SNAP_OLD",
      "ATHENA_IMEX_STAGE3_OLD", "ATHENA_PART_SENDLIST_OLD", "ATHENA_SG_PHASE_ALLOC_OLD",
      "ATHENA_SG_PHASE_FORCE_OLD", "ATHENA_SG_MG_SYNC_COPIES", "ATHENA_SG_GHOST_MAP_OLD",
      "ATHENA_SG_PHASE_COLLECTIVES"};
  // kind, as the old code read the variable (set = getenv non-null):
  //   0 on when its first character is 1 (the dep remap, template check, level stats, device
  //     MPI, project fuse, its probe test, halo Alltoallv)
  //   1 ATHENA_SG_MEMSTATS: on when atoi != 0
  //   2 ATHENA_SG_LEAN: any non-empty value forced the mask max(0, atoi) (also -1: mask 0)
  //   3 ATHENA_SG_SPMV_ROWS: any value (even empty: 0) overrode the backend default
  //   4 ATHENA_PART_MSG_CHUNK: a non-empty value v > 0 below the 1 GiB default split the messages
  //   5 ATHENA_SG_DEPOSIT_ATOMIC: empty or warp was the default; any other value acted (or
  //     stopped the run)
  struct EnvKey {const char *var; const char *key; int kind;};
  static const EnvKey to_keys[] = {
      {"ATHENA_DEP_WEIGHTS_HOST", "<dust>/test_host_reference = deposit", 0},
      {"ATHENA_DEP_VERIFY", "<dust>/test_dep_verify = true", 0},
      {"ATHENA_DEP_PROFILE", "<dust>/test_dep_profile = true", 0},
      {"ATHENA_SG_DEPOSIT_ATOMIC",
       "<gravity>/test_deposit_atomic = warp | plain | cycle | compare", 5},
      {"ATHENA_PART_MSG_CHUNK", "<particles>/test_msg_chunk_bytes = <bytes>", 4},
      {"ATHENA_SG_TEMPLATE_CHECK", "<gravity>/test_sg_report = template_check", 0},
      {"ATHENA_SG_MEMSTATS", "<gravity>/test_sg_report = mem", 1},
      {"ATHENA_SG_LEVEL_STATS", "<gravity>/test_sg_report = level", 0},
      {"ATHENA_SG_LEAN", "<gravity>/test_lean_bits = <bit mask> (a negative value was mask 0)",
       2},
      {"ATHENA_SG_DEVICE_MPI", "<gravity>/test_device_mpi = true", 0},
      {"ATHENA_SG_SPMV_ROWS", "<gravity>/test_spmv_rows = <n> (an empty value was 0)", 3},
      {"ATHENA_SG_PROJECT_FUSE", "<gravity>/test_project_fuse = true", 0},
      {"ATHENA_SG_PROJECT_FUSE_PROBE_TEST", "<gravity>/test_project_fuse = probe_test", 0},
      {"ATHENA_SG_HALO_ALLTOALLV", "<gravity>/test_halo_alltoallv = true", 0}};
  auto value = [](const char *v) {const char *e = std::getenv(v); return e ? std::string(e) : "";};
  std::string off;
  for (const auto &ek : to_keys) {
    const char *ep = std::getenv(ek.var);
    if (ep == nullptr) continue;
    const std::string e = ep;
    bool active = false;
    switch (ek.kind) {
      case 0: active = !e.empty() && e[0] == '1'; break;
      case 1: active = std::atoi(e.c_str()) != 0; break;
      case 2: active = !e.empty(); break;
      case 3: active = true; break;
      case 4: {
        const unsigned long long v = std::strtoull(e.c_str(), nullptr, 10);
        active = !e.empty() && v > 0 && v < (1ULL << 30);
        break;
      }
      default: active = !e.empty() && e != "warp"; break;
    }
    if (!active) {off += std::string(" ") + ek.var + "=" + e; continue;}
    InputFatal(std::string("the environment variable ") + ek.var + " = " + e + " is no longer "
               "read (retired): give the hidden input key " + ek.key + " instead "
               "(in the input file, or with -i on a restart), and unset the variable");
  }
  std::string old;
  for (const char *v : old_paths) {
    if (!value(v).empty()) old += std::string(" ") + v + "=" + value(v);
  }
  if (global_variable::my_rank == 0 && !(old.empty() && off.empty())) {
    std::cout << "# NOTE: retired environment switches are no longer read and are "
              << "ignored (former code paths, or values that left the default):" << old << off
              << std::endl;
  }
}

// retired-compat: the command line of a merged group.  args: (argv index, key, value) of this
// group's keys given on the command line.  The new key and an old member given there with
// different meanings stop the run.  An old member that the file does not carry is migrated: it is
// applied to the new key's value (the file's, or the default) and the argument becomes the new key;
// the new key given there while the file does not carry it is added (the more explicit value, see
// RenamedKeys::Rank, then wins over the file's old members).
template <class T, class F>
void CommandLineGroup(ParameterInput *pin, char *argv[], std::list<std::string> &keep,
                      const char *block, const std::vector<const char *> &olds,
                      const char *neu, const F &apply,
                      const std::vector<std::tuple<int, std::string, std::string>> &args) {
  if (args.empty()) return;
  const std::tuple<int, std::string, std::string> *given = nullptr;
  for (const auto &a : args) if (std::get<1>(a) == neu) given = &a;
  if (given != nullptr) {
    const T n = T::Parse(std::get<2>(*given));
    for (const auto &a : args) {
      if (std::get<1>(a) == neu) continue;
      T probe = n;
      apply(probe, std::get<1>(a), std::get<2>(a));
      if (!(probe == n)) {
        InputFatal(std::string("the command line gives ") + block + "/" + std::get<1>(a) + "=" +
                   std::get<2>(a) + " and " + block + "/" + neu + "=" + std::get<2>(*given) +
                   ": the old key was merged into the hidden key " + neu + " (same "
                   "meaning), with a different value here. Give " + neu + " only.");
      }
    }
    if (!pin->DoesParameterExist(block, neu)) pin->SetString(block, neu, std::get<2>(*given));
    CommandLineKeys().insert({block, neu});
  }
  T t = pin->DoesParameterExist(block, neu) ? T::Parse(pin->GetString(block, neu)) : T();
  std::vector<int> moved;
  for (const auto &a : args) {
    const std::string &k = std::get<1>(a);
    if (k == neu || pin->DoesParameterExist(block, k)) continue;   // (the file carries it)
    apply(t, k, std::get<2>(a));
    moved.push_back(std::get<0>(a));
  }
  if (moved.empty()) return;
  pin->SetString(block, neu, t.Str());
  CommandLineKeys().insert({block, neu});
  for (const int i : moved) {
    keep.push_back(std::string(block) + "/" + neu + "=" + t.Str());
    argv[i] = &keep.back()[0];
  }
  (void)olds;
}

// retired-compat: called in main() before the command line is applied.  The command line only
// changes keys present in the input or restart file.  For the renamed or merged keys it
// accepts either name: a renamed key given under the name the file does not carry is
// read under the one it carries (old name with a restart file written by this version, new name
// with an older one), or added under the new name if the file carries neither (a hidden key not
// written by the run); an old member of a merged group is migrated into the new key
// (CommandLineGroup).  The same key given twice (old and new name) with different values stops the
// run.  <outputN>/precision and test_pbin_compress of a pbin block may be added too.  Every key set
// on the command line is recorded for ApplyRenamedKeys.
inline void RenameCommandLineKeys(ParameterInput *pin, int argc, char *argv[]) {
  static std::list<std::string> keep;   // (argv points into these)
  std::map<std::string, std::pair<std::string, std::string>> seen;   // new name -> (arg, value)
  std::vector<std::tuple<int, std::string, std::string>> g_rar, g_mad, g_co;
  const std::vector<const char *> o_rar = {"cloud_resample_after_regrid",
      "cloud_resample_event_local", "cloud_resample_after_derefine", "cloud_event_dt_omega"};
  const std::vector<const char *> o_mad = {"cloud_merge_burst_dt_omega",
                                           "cloud_merge_burst_interval"};
  const std::vector<const char *> o_co = {"cloud_resample_order", "cloud_sort_order"};
  auto in = [](const std::vector<const char *> &l, const std::string &k) {
    for (const char *x : l) if (k == x) return true;
    return false;
  };
  for (int i = 1; i < argc; ++i) {
    const std::string t = argv[i];
    const std::size_t sl = t.find('/'), eq = t.find('=');
    if (sl == std::string::npos || eq == std::string::npos || eq < sl) continue;
    const std::string block = t.substr(0, sl), name = t.substr(sl + 1, eq - sl - 1);
    const std::string val = t.substr(eq + 1);
    if (!pin->DoesBlockExist(block)) continue;
    if (block == "dust") {
      if (name == "test_resample_after_regrid" || in(o_rar, name)) {
        g_rar.emplace_back(i, name, val); continue;
      }
      if (name == "test_merge_after_derefine" || in(o_mad, name)) {
        g_mad.emplace_back(i, name, val); continue;
      }
      // (cloud_resample_order = position, retired: left to CheckRetiredKeys)
      if (name == "test_canonical_order" || (in(o_co, name) && val != "position")) {
        g_co.emplace_back(i, name, val); continue;
      }
    }
    if (block.compare(0, 6, "output") == 0 && (name == "precision" ||
        name == "test_pbin_compress") && !pin->DoesParameterExist(block, name) &&
        pin->DoesParameterExist(block, "file_type") &&
        pin->GetString(block, "file_type") == "pbin") {
      pin->SetString(block, name, val);
    }
    std::string to = name;
    for (const auto &bk : RenamedKeyList()) {
      if (block != bk.first) continue;
      const std::string neu = std::string("test_") + bk.second;
      if (name != bk.second && name != neu) continue;
      const std::string id = block + "/" + neu;
      auto it = seen.find(id);
      if (it != seen.end() && !RenamedKeys::Same(it->second.second, val)) {
        InputFatal("the command line gives " + it->second.first + " and " + t + ": the same key "
                   "(" + bk.second + " was renamed to the hidden key " + neu + ") "
                   "with different values. Give it once.");
      }
      seen[id] = {t, val};
      if (pin->DoesParameterExist(block, name)) continue;
      if (pin->DoesParameterExist(block, neu)) {
        to = neu;                                                   // old -> new
      } else if (pin->DoesParameterExist(block, bk.second)) {
        to = bk.second;                                             // new -> old
      } else {
        to = neu;                                                   // neither: added
        pin->SetString(block, neu, val);
      }
    }
    CommandLineKeys().insert({block, to});
    if (to != name) {
      keep.push_back(block + "/" + to + t.substr(eq));
      argv[i] = &keep.back()[0];
    }
  }
  CommandLineGroup<ResampleAfterRegrid>(pin, argv, keep, "dust", o_rar,
      "test_resample_after_regrid", ApplyResampleAfterRegrid, g_rar);
  CommandLineGroup<MergeAfterDerefine>(pin, argv, keep, "dust", o_mad,
      "test_merge_after_derefine", ApplyMergeAfterDerefine, g_mad);
  CommandLineGroup<CanonicalOrder>(pin, argv, keep, "dust", o_co, "test_canonical_order",
      ApplyCanonicalOrder, g_co);
  // (old members the file carries are changed in place by the command line and judged by
  // ApplyRenamedKeys with the command-line rank)
  for (const auto *g : {&g_rar, &g_mad, &g_co}) {
    for (const auto &a : *g) {
      const std::string k = std::get<1>(a);
      if (pin->DoesParameterExist("dust", k)) CommandLineKeys().insert({"dust", k});
    }
  }
}

// retired-compat: the pbin blocks of a restart file without precision / test_pbin_compress (a
// file of an older code), recorded right after the restart header is read (before -i)
inline std::vector<std::pair<std::string, std::string>> &RestartPbinGaps() {
  static std::vector<std::pair<std::string, std::string>> g;
  return g;
}
inline void RecordRestartPbinGaps(ParameterInput *pin) {
  for (auto &blk : pin->block) {
    if (blk.block_name.compare(0, 6, "output") != 0) continue;
    const InputLine *ft = blk.GetPtrToLine("file_type");
    if (ft == nullptr || ft->param_value != "pbin") continue;
    for (const char *k : {"precision", "test_pbin_compress"}) {
      if (blk.GetPtrToLine(k) == nullptr) RestartPbinGaps().emplace_back(blk.block_name, k);
    }
  }
}
// The pbin defaults are float32 and compressed, and the values are added
// to the parameters.  A recorded gap of a block that is still a pbin output after -i and the
// command line, and still without the key, takes the former default (float64, uncompressed); a
// pbin block added by -i takes the new defaults.  Called by ApplyRenamedKeys.
inline void KeepRestartPbinFormat(ParameterInput *pin) {
  std::string pb;
  for (const auto &g : RestartPbinGaps()) {
    const std::string &b = g.first, &k = g.second;
    if (!pin->DoesParameterExist(b, "file_type") || pin->GetString(b, "file_type") != "pbin" ||
        pin->DoesParameterExist(b, k)) continue;
    const std::string v = (k == "precision") ? "double" : "none";
    pin->SetString(b, k, v);
    pb += " <" + b + ">/" + k + "=" + v;
  }
  if (!pb.empty() && global_variable::my_rank == 0) {
    std::cout << "# NOTE: pbin output blocks of the restart file without precision / "
              << "test_pbin_compress (as written by earlier versions) keep the former "
              << "format (the new defaults are float32 and compressed; give the keys in an input "
              << "file with -i or on the command line to change it):" << pb << std::endl;
  }
}

// retired-compat: see RenamedKeys
inline void ApplyRenamedKeys(ParameterInput *pin) {
  RenamedKeys rn(pin);
  for (const auto &bk : RenamedKeyList()) {
    const std::string neu = std::string("test_") + bk.second;
    rn.Rename(bk.first, bk.second, neu.c_str());
  }
  rn.Merge<ResampleAfterRegrid>("dust", {"cloud_resample_after_regrid",
      "cloud_resample_event_local", "cloud_resample_after_derefine", "cloud_event_dt_omega"},
      "test_resample_after_regrid", ApplyResampleAfterRegrid);
  rn.Merge<MergeAfterDerefine>("dust", {"cloud_merge_burst_dt_omega",
      "cloud_merge_burst_interval"}, "test_merge_after_derefine", ApplyMergeAfterDerefine);
  // (cloud_resample_order = position, retired, is left in place for
  // CheckRetiredKeys)
  if (TestKeyString(pin, "dust", "cloud_resample_order", "array") == "position") {
    rn.Merge<CanonicalOrder>("dust", {"cloud_sort_order"}, "test_canonical_order",
                             ApplyCanonicalOrder);
  } else {
    rn.Merge<CanonicalOrder>("dust", {"cloud_resample_order", "cloud_sort_order"},
                             "test_canonical_order", ApplyCanonicalOrder);
  }
  KeepRestartPbinFormat(pin);
  rn.Finish();
}

}  // namespace dust

#endif  // DUST_DUST_INPUT_HPP_
