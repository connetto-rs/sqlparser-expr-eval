//! How two operands of one total order may stand to each other.

use crate::scalar::Collation;

/// The total order a comparison between two operands runs under.
///
/// Relations between operands compose only within one order, since a pair of
/// text operands under one collation says nothing about the same values under
/// another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Order {
    /// Integers and `numeric`, `NaN` above every other number.
    Number,
    /// `boolean`, `false` below `true`.
    Boolean,
    /// `date` and `timestamp` on one time line.
    Time,
    /// Text under the deterministic collation the comparison resolves to.
    Text(Collation),
}

/// Which of less, equal and greater the left operand may stand in to the
/// right one on the rows where both hold a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Relation {
    /// Whether the left operand may be below the right one.
    pub less: bool,
    /// Whether the two may be equal.
    pub equal: bool,
    /// Whether the left operand may be above the right one.
    pub greater: bool,
}

impl Relation {
    pub(crate) const NONE: Self = Self { less: false, equal: false, greater: false };
    pub(crate) const LESS: Self = Self { less: true, equal: false, greater: false };
    pub(crate) const EQUAL: Self = Self { less: false, equal: true, greater: false };
    pub(crate) const GREATER: Self = Self { less: false, equal: false, greater: true };
    pub(crate) const ANY: Self = Self { less: true, equal: true, greater: true };

    /// The orders in either relation.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            less: self.less || other.less,
            equal: self.equal || other.equal,
            greater: self.greater || other.greater,
        }
    }

    /// The orders in both relations.
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        Self {
            less: self.less && other.less,
            equal: self.equal && other.equal,
            greater: self.greater && other.greater,
        }
    }

    /// The orders outside the relation, what a comparison failing leaves.
    #[must_use]
    pub fn complement(self) -> Self {
        Self { less: !self.less, equal: !self.equal, greater: !self.greater }
    }

    /// The relation read from the right operand to the left one.
    #[must_use]
    pub fn mirrored(self) -> Self {
        Self { less: self.greater, equal: self.equal, greater: self.less }
    }

    /// Whether no order is left, which no row with both operands holding a
    /// value satisfies.
    #[must_use]
    pub fn is_empty(self) -> bool {
        !(self.less || self.equal || self.greater)
    }

    /// Whether every order of the relation is one of `other`'s.
    #[must_use]
    pub fn is_subset(self, other: Self) -> bool {
        self.intersection(other) == self
    }

    /// The relation of `a` to `c` when `a` stands in `self` to `b` and `b`
    /// stands in `other` to `c`, by the point algebra's composition table.
    #[must_use]
    pub fn composed(self, other: Self) -> Self {
        let (first, second) = (!self.is_empty(), !other.is_empty());
        Self {
            less: (self.less && second) || (other.less && first),
            equal: (self.equal && other.equal)
                || (self.less && other.greater)
                || (self.greater && other.less),
            greater: (self.greater && second) || (other.greater && first),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Relation;

    const ALL: [Relation; 8] = {
        let mut all = [Relation::NONE; 8];
        let mut bits = 0;
        while bits < 8 {
            all[bits] =
                Relation { less: bits & 1 != 0, equal: bits & 2 != 0, greater: bits & 4 != 0 };
            bits += 1;
        }
        all
    };

    /// The order of `left` to `right`, as a single relation.
    fn order_of(left: i32, right: i32) -> Relation {
        match left.cmp(&right) {
            core::cmp::Ordering::Less => Relation::LESS,
            core::cmp::Ordering::Equal => Relation::EQUAL,
            core::cmp::Ordering::Greater => Relation::GREATER,
        }
    }

    #[test]
    fn composition_is_exactly_what_some_three_points_realise() {
        let points = [0, 1, 2];
        for first in ALL {
            for second in ALL {
                let mut realised = Relation::NONE;
                for a in points {
                    for b in points {
                        for c in points {
                            if first.intersection(order_of(a, b)) != Relation::NONE
                                && second.intersection(order_of(b, c)) != Relation::NONE
                            {
                                realised = realised.union(order_of(a, c));
                            }
                        }
                    }
                }
                assert_eq!(first.composed(second), realised, "{first:?} then {second:?}");
            }
        }
    }

    #[test]
    fn mirroring_and_complement_swap_the_expected_orders() {
        let less_or_equal = Relation::LESS.union(Relation::EQUAL);
        assert_eq!(less_or_equal.mirrored(), Relation::GREATER.union(Relation::EQUAL));
        assert_eq!(less_or_equal.complement(), Relation::GREATER);
        assert!(Relation::LESS.is_subset(less_or_equal));
        assert!(!less_or_equal.is_subset(Relation::LESS));
        assert!(Relation::NONE.is_empty());
        assert!(!Relation::EQUAL.is_empty());
    }
}
