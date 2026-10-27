//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
#include "gravity/vacuum_mesh.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace gravity {
namespace {
constexpr int kCells = 8;
constexpr double kPi = 3.141592653589793238462643383279502884;
using Point = std::array<double,3>;

bool PowerTwo(int n) { return n >= kCells && (n & (n-1)) == 0; }
GlobalIndex Count(const GravityPatch &p) {
  GlobalIndex n = 1;
  for (int d=0; d<3; ++d) {
    if (p.cells[d]<=0 || n>std::numeric_limits<GlobalIndex>::max()/p.cells[d])
      throw std::overflow_error("Gravity vacuum patch cell count overflows");
    n *= p.cells[d];
  }
  return n;
}
GlobalIndex AddCount(GlobalIndex a,GlobalIndex b) {
  if (b>std::numeric_limits<GlobalIndex>::max()-a)
    throw std::overflow_error("Gravity vacuum global unknown count overflows");
  return a+b;
}
struct Box { Point lo{},hi{}; };
Box Bounds(const GravityPatch &p) {
  Box b;
  for (int d=0; d<3; ++d) { b.lo[d]=p.lo[d]; b.hi[d]=p.hi[d]; }
  return b;
}
bool Intersects(const Box &a,const Box &b,double eps) {
  for (int d=0; d<3; ++d)
    if (a.hi[d]<b.lo[d]-eps || b.hi[d]<a.lo[d]-eps) return false;
  return true;
}

// The tree indexes PATCH boxes only. Every query returns metadata IDs, never
// cells; storage is linear in physical+virtual patch count.
class BoxTree {
 public:
  explicit BoxTree(std::vector<Box> boxes):boxes_(std::move(boxes)) {
    if (boxes_.size()>static_cast<std::size_t>(std::numeric_limits<int>::max()/2))
      throw std::overflow_error("Gravity vacuum patch tree exceeds int metadata indices");
    order_.resize(boxes_.size()); std::iota(order_.begin(),order_.end(),0);
    if (!order_.empty()) Build(0,order_.size());
  }
  void Query(const Box &b,double eps,std::vector<int> &out) const {
    out.clear(); if (!nodes_.empty()) QueryNode(0,b,eps,out);
  }
 private:
  struct Node { Box box; int begin=0,end=0,left=-1,right=-1; };
  std::vector<Box> boxes_;
  std::vector<int> order_;
  std::vector<Node> nodes_;
  int Build(int begin,int end) {
    Node n; n.begin=begin; n.end=end;
    n.box.lo.fill(std::numeric_limits<double>::max());
    n.box.hi.fill(-std::numeric_limits<double>::max());
    for (int i=begin; i<end; ++i) for (int d=0; d<3; ++d) {
      n.box.lo[d]=std::min(n.box.lo[d],boxes_[order_[i]].lo[d]);
      n.box.hi[d]=std::max(n.box.hi[d],boxes_[order_[i]].hi[d]);
    }
    const int node=nodes_.size(); nodes_.push_back(n);
    if (end-begin<=8) return node;
    int axis=0;
    for (int d=1; d<3; ++d)
      if (n.box.hi[d]-n.box.lo[d]>n.box.hi[axis]-n.box.lo[axis]) axis=d;
    const int mid=begin+(end-begin)/2;
    std::nth_element(order_.begin()+begin,order_.begin()+mid,order_.begin()+end,
        [&](int a,int b) { return boxes_[a].lo[axis]+boxes_[a].hi[axis]<
                                boxes_[b].lo[axis]+boxes_[b].hi[axis]; });
    const int left=Build(begin,mid),right=Build(mid,end);
    nodes_[node].left=left; nodes_[node].right=right; return node;
  }
  void QueryNode(int node,const Box &b,double eps,std::vector<int> &out) const {
    const auto &n=nodes_[node]; if (!Intersects(n.box,b,eps)) return;
    if (n.left<0) {
      for (int i=n.begin; i<n.end; ++i)
        if (Intersects(boxes_[order_[i]],b,eps)) out.push_back(order_[i]);
    } else { QueryNode(n.left,b,eps,out); QueryNode(n.right,b,eps,out); }
  }
};

struct Leaf { GravityPatch patch; int level=0,original=-1; };
std::array<Leaf,8> Split(const Leaf &parent) {
  std::array<Leaf,8> children;
  for (int bits=0; bits<8; ++bits) {
    auto &c=children[bits]; c=parent; ++c.level;
    for (int d=0; d<3; ++d) {
      const Real mid=(parent.patch.lo[d]+parent.patch.hi[d])/2;
      if (bits&(1<<d)) c.patch.lo[d]=mid; else c.patch.hi[d]=mid;
    }
  }
  return children;
}

struct Boundary {
  std::vector<Box> boxes;
  std::vector<int> levels;
  BoxTree tree;
  Boundary(std::vector<Box> b,std::vector<int> l)
      :boxes(std::move(b)),levels(std::move(l)),tree(boxes) {}
  int Finest(const GravityPatch &p,double eps) const {
    Box b=Bounds(p); b.lo[2]=b.hi[2]=0;
    std::vector<int> ids; tree.Query(b,eps,ids); int finest=-1;
    for (int i:ids) {
      bool area=true;
      for (int d=0; d<2; ++d)
        area=area && std::min(b.hi[d],boxes[i].hi[d])-
                           std::max(b.lo[d],boxes[i].lo[d])>eps;
      if (area) finest=std::max(finest,levels[i]);
    }
    if (finest<0) throw std::runtime_error("Gravity vacuum uncovered boundary footprint");
    return finest;
  }
};

void RefineBound(const Leaf &leaf,const Boundary &boundary,int side,
                 const CompositeDomain &physical,double kmin,double eps,
                 std::vector<Leaf> &out) {
  const double distance=side<0 ? physical.lo[2]-leaf.patch.hi[2]
                              : leaf.patch.lo[2]-physical.hi[2];
  const int finest=boundary.Finest(leaf.patch,eps);
  const double allowed=kmin*std::max(0.,distance)/(2*std::log(2.));
  if (finest-leaf.level>allowed+1.e-10) {
    for (const auto &child:Split(leaf))
      RefineBound(child,boundary,side,physical,kmin,eps,out);
  } else out.push_back(leaf);
}

int Balance(std::vector<Leaf> &leaves,const CompositeDomain &domain,double eps) {
  const double lx=domain.hi[0]-domain.lo[0],ly=domain.hi[1]-domain.lo[1];
  // Deliberately do not use domain.shear_offset: exterior topology must remain
  // fixed as the shearing phase evolves. Shifted-seam arbitrary dyadic overlap
  // is a composite-operator responsibility.
  std::vector<int> neighbors;
  for (int pass=0; pass<128; ++pass) {
    std::vector<Box> boxes; boxes.reserve(leaves.size());
    for (const auto &p:leaves) boxes.push_back(Bounds(p.patch));
    BoxTree tree(boxes); std::vector<unsigned char> split(leaves.size(),0);
    for (std::size_t i=0; i<leaves.size(); ++i) {
      for (int ix=-1; ix<=1; ++ix) for (int iy=-1; iy<=1; ++iy) {
        const Point image{{ix*lx,iy*ly,0.}}; Box query=boxes[i];
        for (int d=0; d<3; ++d) { query.lo[d]-=image[d]; query.hi[d]-=image[d]; }
        tree.Query(query,eps,neighbors);
        for (int j:neighbors) {
          if (i==static_cast<std::size_t>(j) && ix==0 && iy==0) continue;
          bool touch=false,overlap=true;
          for (int d=0; d<3; ++d) {
            const double common=std::min(query.hi[d],boxes[j].hi[d])-
                                std::max(query.lo[d],boxes[j].lo[d]);
            overlap=overlap && common>eps; touch=touch || std::abs(common)<=eps;
          }
          if (overlap) throw std::invalid_argument("Gravity vacuum input patches overlap");
          if (!touch || std::abs(leaves[i].level-leaves[j].level)<=1) continue;
          const std::size_t coarse=leaves[i].level<leaves[j].level ? i : j;
          if (leaves[coarse].original>=0)
            throw std::invalid_argument("Gravity vacuum would require physical refinement "
                                        "to enforce2:1 in unshifted periodic geometry");
          split[coarse]=1;
        }
      }
    }
    if (std::find(split.begin(),split.end(),1)==split.end()) return pass;
    std::vector<Leaf> balanced; balanced.reserve(leaves.size());
    for (std::size_t i=0; i<leaves.size(); ++i) {
      if (split[i]) for (const auto &c:Split(leaves[i])) balanced.push_back(c);
      else balanced.push_back(leaves[i]);
    }
    leaves.swap(balanced);
  }
  throw std::runtime_error("Gravity vacuum2:1 balance exceeded bounded128 passes");
}
}  // namespace

VacuumMesh BuildVacuumMesh(const std::vector<GravityPatch> &physical,
                          const CompositeDomain &domain,int nranks) {
  if (physical.empty() || nranks<=0 || domain.ndim!=3 || !domain.isolated_z)
    throw std::invalid_argument("Gravity vacuum requires3D isolated-z patches and ranks");
  Point length{},root_width{}; double physical_volume=1.; int minimum_root=0;
  for (int d=0; d<3; ++d) {
    length[d]=domain.hi[d]-domain.lo[d];
    if (!(length[d]>0.) || !std::isfinite(length[d]) || !PowerTwo(domain.root_counts[d]))
      throw std::invalid_argument("Gravity vacuum root counts must be powers-of-two>=8");
    root_width[d]=length[d]/domain.root_counts[d]; physical_volume*=length[d];
    minimum_root=d==0 ? domain.root_counts[d] : std::min(minimum_root,domain.root_counts[d]);
  }
  if (!std::isfinite(static_cast<double>(domain.shear_offset)))
    throw std::invalid_argument("Gravity vacuum nonfinite shear offset");
  const double eps=*std::max_element(length.begin(),length.end())*1.e-10;
  VacuumMesh result; result.domain=domain;
  result.physical_patch_count=physical.size();
  result.physical_cells_per_rank.assign(nranks,0);
  result.total_cells_per_rank.assign(nranks,0);
  result.input_to_output_patch.assign(physical.size(),-1);
  std::vector<Leaf> leaves; leaves.reserve(physical.size());
  std::array<std::vector<Box>,2> boundary_boxes;
  std::array<std::vector<int>,2> boundary_levels;
  std::array<double,2> area{{0.,0.}}; double covered=0.;
  for (std::size_t i=0; i<physical.size(); ++i) {
    const auto &p=physical[i];
    if (p.owner<0 || p.owner>=nranks)
      throw std::invalid_argument("Gravity vacuum invalid physical owner");
    int level=-1; double volume=1.;
    for (int d=0; d<3; ++d) {
      const double extent=p.hi[d]-p.lo[d],width=extent/p.cells[d];
      if (!PowerTwo(p.cells[d]) || !(extent>0.) || !std::isfinite(extent) ||
          p.lo[d]<domain.lo[d]-eps || p.hi[d]>domain.hi[d]+eps)
        throw std::invalid_argument("Gravity vacuum invalid physical box/count>=8");
      const double loglevel=std::log2(root_width[d]/width);
      const int ld=std::llround(loglevel);
      if (ld<0 || ld>40 || std::abs(loglevel-ld)>1.e-7 || (level>=0 && level!=ld))
        throw std::invalid_argument("Gravity vacuum requires isotropic dyadic root-aspect refinement");
      level=ld;
      const double aligned=(p.lo[d]-domain.lo[d])/(kCells*width);
      if (std::abs(aligned-std::round(aligned))>1.e-6)
        throw std::invalid_argument("Gravity vacuum physical bounds require8-cell tile alignment");
      volume*=extent;
    }
    covered+=volume; leaves.push_back({p,level,static_cast<int>(i)});
    const GlobalIndex n=Count(p); result.physical_unknowns=AddCount(result.physical_unknowns,n);
    result.physical_cells_per_rank[p.owner]+=n;
    for (int side=0; side<2; ++side) {
      const double face=side==0 ? p.lo[2] : p.hi[2];
      const double wall=side==0 ? domain.lo[2] : domain.hi[2];
      if (std::abs(face-wall)>eps) continue;
      Box b=Bounds(p); b.lo[2]=b.hi[2]=0;
      boundary_boxes[side].push_back(b); boundary_levels[side].push_back(level);
      area[side]+=(p.hi[0]-p.lo[0])*(p.hi[1]-p.lo[1]);
    }
  }
  if (std::abs(covered-physical_volume)>physical_volume*1.e-8 ||
      std::abs(area[0]-length[0]*length[1])>length[0]*length[1]*1.e-8 ||
      std::abs(area[1]-length[0]*length[1])>length[0]*length[1]*1.e-8)
    throw std::invalid_argument("Gravity vacuum physical volume/boundary coverage mismatch");

  int coarse_shift=0;
  for (int n=minimum_root/kCells; n>1; n/=2) ++coarse_shift;
  Point tile{};
  for (int d=0; d<3; ++d) tile[d]=kCells*std::ldexp(root_width[d],coarse_shift);
  const double requested=2*std::max(length[0],length[1]);
  const long long layers=static_cast<long long>(std::ceil(requested/tile[2]));
  result.padding=layers*tile[2];
  result.domain.lo[2]-=result.padding; result.domain.hi[2]+=result.padding;
  result.domain.c1_z_interfaces.push_back(domain.lo[2]);
  result.domain.c1_z_interfaces.push_back(domain.hi[2]);
  std::sort(result.domain.c1_z_interfaces.begin(),result.domain.c1_z_interfaces.end());
  result.domain.c1_z_interfaces.erase(std::unique(result.domain.c1_z_interfaces.begin(),
      result.domain.c1_z_interfaces.end()),result.domain.c1_z_interfaces.end());
  const long long added=std::llround(2*result.padding/root_width[2]);
  if (added>std::numeric_limits<int>::max()-domain.root_counts[2])
    throw std::overflow_error("Gravity vacuum extended root count exceeds int");
  result.domain.root_counts[2]+=static_cast<int>(added);
  const int nx=std::llround(length[0]/tile[0]),ny=std::llround(length[1]/tile[1]);
  const double kmin=2*kPi/std::max(length[0],length[1]);
  for (int side=0; side<2; ++side) {
    Boundary boundary(std::move(boundary_boxes[side]),std::move(boundary_levels[side]));
    for (long long iz=0; iz<layers; ++iz) for (int iy=0; iy<ny; ++iy)
      for (int ix=0; ix<nx; ++ix) {
        Leaf root; root.level=-coarse_shift; root.patch.cells={{kCells,kCells,kCells}};
        root.patch.lo[0]=domain.lo[0]+ix*tile[0]; root.patch.hi[0]=root.patch.lo[0]+tile[0];
        root.patch.lo[1]=domain.lo[1]+iy*tile[1]; root.patch.hi[1]=root.patch.lo[1]+tile[1];
        if (side==0) {
          root.patch.hi[2]=domain.lo[2]-iz*tile[2]; root.patch.lo[2]=root.patch.hi[2]-tile[2];
        } else {
          root.patch.lo[2]=domain.hi[2]+iz*tile[2]; root.patch.hi[2]=root.patch.lo[2]+tile[2];
        }
        RefineBound(root,boundary,side==0 ? -1:1,domain,kmin,eps,leaves);
      }
  }
  result.balance_passes=Balance(leaves,result.domain,eps);

  const GlobalIndex per_vacuum=kCells*kCells*kCells;
  result.vacuum_patch_count=leaves.size()-physical.size();
  result.vacuum_unknowns=static_cast<GlobalIndex>(result.vacuum_patch_count)*per_vacuum;
  const GlobalIndex total=AddCount(result.physical_unknowns,result.vacuum_unknowns);
  // Contiguous octree-order chunks avoid round-robin scattering of neighboring
  // vacuum patches. Account for existing physical ownership when assigning work.
  std::vector<std::vector<std::size_t>> physical_owned(nranks),vacuum_owned(nranks);
  for (std::size_t i=0; i<physical.size(); ++i) physical_owned[physical[i].owner].push_back(i);
  using Load=std::pair<GlobalIndex,int>;
  std::priority_queue<Load,std::vector<Load>,std::greater<Load>> loads;
  for (int r=0; r<nranks; ++r) loads.push({result.physical_cells_per_rank[r],r});
  std::vector<std::size_t> quota(nranks,0);
  for (std::size_t i=0; i<result.vacuum_patch_count; ++i) {
    auto least=loads.top(); loads.pop(); ++quota[least.second];
    least.first+=per_vacuum; loads.push(least);
  }
  std::size_t next=physical.size();
  for (int r=0; r<nranks; ++r) {
    for (std::size_t j=0; j<quota[r]; ++j) vacuum_owned[r].push_back(next++);
  }
  result.row_partition.assign(nranks+1,0); result.patches.reserve(leaves.size());
  result.original_patch.reserve(leaves.size());
  for (int r=0; r<nranks; ++r) {
    GlobalIndex cursor=result.row_partition[r];
    for (std::size_t i:physical_owned[r]) {
      GravityPatch patch=physical[i]; patch.row_begin=cursor; cursor+=Count(patch);
      result.input_to_output_patch[i]=result.patches.size();
      result.patches.push_back(patch); result.original_patch.push_back(i);
    }
    for (std::size_t i:vacuum_owned[r]) {
      GravityPatch patch=leaves[i].patch; patch.owner=r; patch.gid=-1; patch.row_begin=cursor;
      cursor+=per_vacuum; result.patches.push_back(patch); result.original_patch.push_back(-1);
    }
    result.row_partition[r+1]=cursor;
    result.total_cells_per_rank[r]=cursor-result.row_partition[r];
  }
  result.unknown_overhead=static_cast<Real>(total)/result.physical_unknowns;
  result.metadata_bytes=result.patches.size()*sizeof(GravityPatch)+
      result.original_patch.size()*sizeof(int)+result.input_to_output_patch.size()*sizeof(int)+
      (result.row_partition.size()+result.physical_cells_per_rank.size()+
       result.total_cells_per_rank.size())*sizeof(GlobalIndex);
  return result;
}
}  // namespace gravity
