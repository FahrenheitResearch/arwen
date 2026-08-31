//! A 3-D k-d tree whose nearest-neighbour answer is DEFINED, not
//! whatever a traversal order happened to reach first.
//!
//! This replaces `scipy.spatial.cKDTree.query(k=1,
//! distance_upper_bound=...)` on the observation battery's remap plan.
//! Two halves of scipy's behaviour are reproduced exactly because they
//! are specified by the arithmetic, and one is deliberately NOT
//! reproduced because scipy leaves it unspecified:
//!
//! **Reproduced — the metric.**  scipy accumulates the squared distance
//! as `dx*dx + dy*dy + dz*dz` in that dimension order and returns
//! `sqrt` of it.  Measured bit-identical over 200 000 random unit-sphere
//! pairs (`golden/gen_regrid_goldens.py`, probe `distance_formula`).
//!
//! **Reproduced — the bound.**  scipy squares `distance_upper_bound` and
//! accepts a candidate on the STRICT predicate `d2 < bound*bound`.  Both
//! halves matter: comparing `d < bound` instead disagrees with scipy on
//! points within one ULP of the bound (measured: 2060 disagreements in
//! 12 000 boundary trials, scipy agreeing with the squared form every
//! time), and using `<=` disagrees on the exact-equality case (measured:
//! 300 of 300 exact-equality trials rejected by scipy).
//!
//! **NOT reproduced — the tie.**  When two source points are exactly
//! equidistant, scipy returns whichever the tree traversal reached
//! first.  Measured over 400 duplicate-source trials it returned the
//! lowest flat index 232 times and some other tied index 168 times, and
//! over 400 mirror-symmetric trials 268 / 132.  There is no rule there
//! to be bit-exact to.  This tree defines the answer instead: **among
//! exactly tied candidates the lowest flat source index wins.**  The
//! concrete breakage that rule prevents is a plan whose integer mapping
//! is not a function of the two grids -- the battery's whole reason for
//! computing the plan once and reusing it across arms is that no score
//! may differ because a neighbour search broke a tie differently, and a
//! tie broken by traversal order is exactly that, one rebuild away.
//!
//! Pruning descends into a box whose lower-bound distance EQUALS the
//! best so far, rather than cutting on `>=`, because a tied candidate in
//! that box may carry a lower index.  Ties are rare enough that the cost
//! is not measurable and the rule is worth more than the cycles.

/// The best candidate found so far: `None` until something inside the
/// bound is seen.
#[derive(Clone, Copy, Debug)]
pub struct Neighbour {
    pub index: usize,
    /// Squared chord distance, the value scipy compares and square-roots.
    pub distance_squared: f64,
}

/// A balanced k-d tree over 3-D points, built once and queried many.
pub struct KdTree {
    points: Vec<[f64; 3]>,
    /// Point indices in tree order; each node owns a contiguous slice.
    order: Vec<u32>,
    nodes: Vec<Node>,
    root: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
struct Node {
    /// Half-open range into `order`.
    start: u32,
    end: u32,
    /// Axis-aligned bounding box of the points in this node.
    lower: [f64; 3],
    upper: [f64; 3],
    left: Option<u32>,
    right: Option<u32>,
}

/// Leaves below this many points are scanned linearly.  Same order as
/// scipy's default `leafsize=16`; the value changes speed only, never
/// the answer, because the answer is defined by the arithmetic above.
const LEAF_SIZE: usize = 16;

impl KdTree {
    pub fn build(points: Vec<[f64; 3]>) -> Self {
        let count = points.len();
        let mut tree = KdTree {
            points,
            order: (0..count as u32).collect(),
            nodes: Vec::new(),
            root: None,
        };
        if count > 0 {
            tree.root = Some(tree.build_node(0, count));
        }
        tree
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    fn build_node(&mut self, start: usize, end: usize) -> u32 {
        let (lower, upper) = self.bounds(start, end);
        let handle = self.nodes.len() as u32;
        self.nodes.push(Node {
            start: start as u32,
            end: end as u32,
            lower,
            upper,
            left: None,
            right: None,
        });
        if end - start <= LEAF_SIZE {
            return handle;
        }
        // Split the widest dimension at its median, which keeps the tree
        // balanced for the elongated point clouds an observation swath
        // makes.  A degenerate spread (every point identical on the
        // widest axis) cannot be split usefully, so it stays a leaf.
        let axis = (0..3)
            .max_by(|&a, &b| {
                (upper[a] - lower[a])
                    .partial_cmp(&(upper[b] - lower[b]))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(0);
        if !(upper[axis] > lower[axis]) {
            return handle;
        }
        let middle = start + (end - start) / 2;
        let points = &self.points;
        self.order[start..end].select_nth_unstable_by(middle - start, |&a, &b| {
            let left = points[a as usize][axis];
            let right = points[b as usize][axis];
            left.partial_cmp(&right)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
        let left = self.build_node(start, middle);
        let right = self.build_node(middle, end);
        self.nodes[handle as usize].left = Some(left);
        self.nodes[handle as usize].right = Some(right);
        handle
    }

    fn bounds(&self, start: usize, end: usize) -> ([f64; 3], [f64; 3]) {
        let mut lower = [f64::INFINITY; 3];
        let mut upper = [f64::NEG_INFINITY; 3];
        for &slot in &self.order[start..end] {
            let point = self.points[slot as usize];
            for axis in 0..3 {
                if point[axis] < lower[axis] {
                    lower[axis] = point[axis];
                }
                if point[axis] > upper[axis] {
                    upper[axis] = point[axis];
                }
            }
        }
        (lower, upper)
    }

    /// Nearest point to `query` with squared distance strictly below
    /// `bound_squared`, ties broken by lowest index.
    pub fn nearest(&self, query: [f64; 3], bound_squared: f64) -> Option<Neighbour> {
        let root = self.root?;
        let mut best: Option<Neighbour> = None;
        // `radius` is the exclusive ceiling: a candidate must be
        // strictly below it, which is scipy's squared-bound predicate
        // before anything is found and the tie window after.
        let mut radius = bound_squared;
        self.search(root, query, &mut best, &mut radius);
        best
    }

    fn search(
        &self,
        handle: u32,
        query: [f64; 3],
        best: &mut Option<Neighbour>,
        radius: &mut f64,
    ) {
        let node = self.nodes[handle as usize];
        match (node.left, node.right) {
            (Some(left), Some(right)) => {
                // Visit the nearer child first so the radius shrinks
                // early; correctness does not depend on the order.
                let left_distance = self.box_distance_squared(left, query);
                let right_distance = self.box_distance_squared(right, query);
                let (first, first_distance, second, second_distance) =
                    if left_distance <= right_distance {
                        (left, left_distance, right, right_distance)
                    } else {
                        (right, right_distance, left, left_distance)
                    };
                if first_distance <= *radius {
                    self.search(first, query, best, radius);
                }
                if second_distance <= *radius {
                    self.search(second, query, best, radius);
                }
            }
            _ => {
                for &slot in &self.order[node.start as usize..node.end as usize] {
                    let index = slot as usize;
                    let point = self.points[index];
                    let dx = point[0] - query[0];
                    let dy = point[1] - query[1];
                    let dz = point[2] - query[2];
                    let distance_squared = dx * dx + dy * dy + dz * dz;
                    let better = match best {
                        None => distance_squared < *radius,
                        Some(current) => {
                            distance_squared < current.distance_squared
                                || (distance_squared == current.distance_squared
                                    && index < current.index)
                        }
                    };
                    if better {
                        *best = Some(Neighbour {
                            index,
                            distance_squared,
                        });
                        // The window stays INCLUSIVE of the best
                        // distance so a tied point with a lower index in
                        // an unvisited box is still reachable.
                        *radius = distance_squared;
                    }
                }
            }
        }
    }

    /// Squared distance from `query` to the node's bounding box: zero
    /// inside, and the usual per-axis shortfall outside.
    fn box_distance_squared(&self, handle: u32, query: [f64; 3]) -> f64 {
        let node = self.nodes[handle as usize];
        let mut total = 0.0;
        for axis in 0..3 {
            let delta = if query[axis] < node.lower[axis] {
                node.lower[axis] - query[axis]
            } else if query[axis] > node.upper[axis] {
                query[axis] - node.upper[axis]
            } else {
                0.0
            };
            total += delta * delta;
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(points: &[[f64; 3]], query: [f64; 3], bound_squared: f64) -> Option<Neighbour> {
        let mut best: Option<Neighbour> = None;
        for (index, point) in points.iter().enumerate() {
            let dx = point[0] - query[0];
            let dy = point[1] - query[1];
            let dz = point[2] - query[2];
            let distance_squared = dx * dx + dy * dy + dz * dz;
            if distance_squared >= bound_squared {
                continue;
            }
            let better = match best {
                None => true,
                Some(current) => {
                    distance_squared < current.distance_squared
                        || (distance_squared == current.distance_squared && index < current.index)
                }
            };
            if better {
                best = Some(Neighbour {
                    index,
                    distance_squared,
                });
            }
        }
        best
    }

    /// A deterministic xorshift so the property test is reproducible
    /// without a dependency.
    struct Rng(u64);

    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 11) as f64 / (1u64 << 53) as f64
        }

        fn unit(&mut self) -> [f64; 3] {
            let z = self.next_f64() * 2.0 - 1.0;
            let phi = self.next_f64() * std::f64::consts::TAU;
            let r = (1.0 - z * z).max(0.0).sqrt();
            [r * phi.cos(), r * phi.sin(), z]
        }
    }

    #[test]
    fn the_tree_agrees_with_brute_force_on_index_and_distance() {
        let mut rng = Rng(0x2545F4914F6CDD1D);
        for trial in 0..40 {
            let count = 1 + (trial * 17) % 300;
            let points: Vec<[f64; 3]> = (0..count).map(|_| rng.unit()).collect();
            let tree = KdTree::build(points.clone());
            for _ in 0..25 {
                let query = rng.unit();
                for bound in [0.05, 0.3, 1.0, 4.1] {
                    let expected = brute_force(&points, query, bound * bound);
                    let got = tree.nearest(query, bound * bound);
                    match (expected, got) {
                        (None, None) => {}
                        (Some(a), Some(b)) => {
                            assert_eq!(a.index, b.index, "index disagreed at bound {bound}");
                            assert_eq!(
                                a.distance_squared.to_bits(),
                                b.distance_squared.to_bits(),
                                "distance disagreed at bound {bound}"
                            );
                        }
                        (a, b) => panic!("reachability disagreed: {a:?} vs {b:?}"),
                    }
                }
            }
        }
    }

    #[test]
    fn exact_ties_resolve_to_the_lowest_index() {
        // Six copies of one point scattered through the array, plus
        // decoys further away.  Every build order must answer 1, the
        // lowest tied index -- this is the property scipy does not have.
        let target = [0.0, 0.0, 1.0];
        let decoy = [1.0, 0.0, 0.0];
        let points = vec![
            decoy, target, decoy, target, decoy, target, decoy, target, decoy, target, decoy,
            target, decoy, decoy, decoy, decoy, decoy, decoy, decoy, decoy, decoy, decoy, decoy,
            decoy, decoy, decoy, decoy, decoy, decoy, decoy, decoy, decoy, decoy,
        ];
        let tree = KdTree::build(points);
        let found = tree.nearest(target, 4.1 * 4.1).unwrap();
        assert_eq!(found.index, 1);
        assert_eq!(found.distance_squared, 0.0);
    }

    #[test]
    fn mirror_symmetric_ties_resolve_to_the_lowest_index() {
        let mut points = Vec::new();
        for step in 0..40 {
            let angle = step as f64 * std::f64::consts::TAU / 40.0;
            points.push([angle.cos(), angle.sin(), 0.0]);
        }
        // The query sits on the axis, exactly equidistant from every
        // point on the ring.
        let tree = KdTree::build(points);
        let found = tree.nearest([0.0, 0.0, 1.0], 4.1 * 4.1).unwrap();
        assert_eq!(found.index, 0);
    }

    #[test]
    fn the_bound_is_strict_on_the_squared_predicate() {
        let points = vec![[0.0, 0.0, 0.0]];
        let tree = KdTree::build(points);
        let query = [0.3, 0.0, 0.0];
        let distance_squared = 0.3 * 0.3;
        assert!(tree.nearest(query, distance_squared).is_none());
        assert!(
            tree.nearest(query, f64::from_bits(distance_squared.to_bits() + 1))
                .is_some()
        );
    }

    #[test]
    fn an_empty_tree_finds_nothing_rather_than_panicking() {
        let tree = KdTree::build(Vec::new());
        assert!(tree.is_empty());
        assert!(tree.nearest([0.0, 0.0, 0.0], 1.0).is_none());
    }

    #[test]
    fn a_degenerate_cloud_of_identical_points_still_builds_and_answers() {
        let points = vec![[0.5, 0.5, 0.5]; 500];
        let tree = KdTree::build(points);
        let found = tree.nearest([0.5, 0.5, 0.5], 1.0).unwrap();
        assert_eq!(found.index, 0);
    }

    /// The tree is a TREE, and its root box is the cloud.
    ///
    /// Every test above this one compares ANSWERS, and the answers are
    /// the same whether the search prunes or scans every point: break
    /// the bounding boxes, or the rule that decides when a node stops
    /// splitting, and `nearest` still returns exactly the right index
    /// and the right bits -- it just walks the whole cloud to get there.
    /// So a change that collapses this to one leaf node is invisible to
    /// the parity suite, and it is not a small thing: a regrid plan
    /// resolves one query per destination cell against every observation
    /// in the swath, and linear search turns a few seconds into hours.
    ///
    /// This is the structural claim the answers cannot carry: the root
    /// box is exactly the cloud's extent on each axis, and four thousand
    /// points do not fit in one leaf of sixteen.
    #[test]
    fn the_root_box_is_the_cloud_and_the_tree_really_splits() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        let count = 4000;
        let points: Vec<[f64; 3]> = (0..count).map(|_| rng.unit()).collect();
        let tree = KdTree::build(points.clone());
        let root = tree.root.expect("a non-empty cloud has a root");
        let node = tree.nodes[root as usize];
        for axis in 0..3 {
            let mut lower = f64::INFINITY;
            let mut upper = f64::NEG_INFINITY;
            for point in &points {
                if point[axis] < lower {
                    lower = point[axis];
                }
                if point[axis] > upper {
                    upper = point[axis];
                }
            }
            assert_eq!(
                node.lower[axis].to_bits(),
                lower.to_bits(),
                "root box lower bound on axis {axis}"
            );
            assert_eq!(
                node.upper[axis].to_bits(),
                upper.to_bits(),
                "root box upper bound on axis {axis}"
            );
        }
        let least = count / LEAF_SIZE;
        assert!(
            tree.nodes.len() >= least,
            "{count} points in {} node(s); a leaf holds {LEAF_SIZE}, so a tree \
             that split has at least {least}.  One node means every query \
             scans the whole cloud.",
            tree.nodes.len()
        );
    }

    /// The box distance is the thing that decides what is skipped.
    ///
    /// Return zero from it and nothing is ever pruned; return a negative
    /// number and nothing is ever pruned either; accumulate the axis
    /// shortfalls with the wrong sign and the same.  All three answer
    /// every query correctly and none of them is a k-d tree any more,
    /// which is why this asserts the function rather than the search.
    #[test]
    fn the_box_distance_is_zero_inside_and_the_axis_shortfall_outside() {
        // Two points, so the root is a leaf whose box is [0,0,0]..[1,2,3].
        let tree = KdTree::build(vec![[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]);
        let root = tree.root.expect("a non-empty cloud has a root");
        assert_eq!(
            tree.box_distance_squared(root, [0.5, 1.0, 1.5]),
            0.0,
            "a query inside the box is at no distance from it"
        );
        assert_eq!(
            tree.box_distance_squared(root, [3.0, 1.0, 1.5]),
            4.0,
            "two beyond the upper x bound, and nothing on the other axes"
        );
        assert_eq!(
            tree.box_distance_squared(root, [-1.0, 1.0, -2.0]),
            5.0,
            "one below the lower x bound and two below the lower z bound"
        );
    }
}
