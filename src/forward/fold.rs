//! The conservative comparison reading and the type arithmetic the default
//! dialect answers with.

use core::ops::Bound;

use sql_values::{PgNumeric, PgValue};
use sqlparser::ast::BinaryOperator;

use crate::{
    domain::{Evaluation, Interval, Values},
    scalar::ScalarType,
    support,
};

/// The category two operands compare in.
pub(crate) enum Category {
    Number,
    Boolean,
}

/// The kinds of answer one non-`NULL` left value may give against the whole
/// set of non-`NULL` right values, each a possibility. No kind at all means
/// no pair of values meets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowKinds {
    /// The left value may satisfy the operator against every right value.
    pub all_true: bool,
    /// The left value may fail the operator against every right value.
    pub all_false: bool,
    /// The left value may satisfy the operator against some right values and
    /// fail it against others.
    pub mixed: bool,
}

impl RowKinds {
    /// No pair of values meets.
    pub const NONE: Self = Self { all_true: false, all_false: false, mixed: false };
    /// Every kind stays possible.
    pub const ANY: Self = Self { all_true: true, all_false: true, mixed: true };

    /// Some pair may satisfy the operator.
    #[must_use]
    pub fn may_be_true(self) -> bool {
        self.all_true || self.mixed
    }

    /// Some pair may fail the operator.
    #[must_use]
    pub fn may_be_false(self) -> bool {
        self.all_false || self.mixed
    }

    /// These kinds with a mixed row wherever a pair may satisfy, or may
    /// fail, outside the rows that may do so against every right value.
    #[must_use]
    fn with_pairs(self, may_satisfy: bool, may_fail: bool) -> Self {
        Self {
            mixed: self.mixed || (may_satisfy && !self.all_true) || (may_fail && !self.all_false),
            ..self
        }
    }
}

/// A dialect's reading of the comparison of two evaluated operands.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Comparison {
    /// The kinds of answer a left value may give against the right values.
    pub rows: RowKinds,
    /// Evaluating the operands or applying the operator may raise.
    pub error: bool,
}

impl Comparison {
    /// The reading with `rows` and the raise possibility `error`.
    #[must_use]
    pub fn new(rows: RowKinds, error: bool) -> Self {
        Self { rows, error }
    }
}

/// The conservative reading of a comparison: only boolean and exact-number
/// literals compare by value, because the standard dialects type untyped
/// literals with their own affinity and compare text under their own
/// collation.
pub(crate) fn standard_comparison(
    left: &Evaluation,
    op: &BinaryOperator,
    right: &Evaluation,
    error: bool,
) -> Comparison {
    let decidable = match (left.constant(), right.constant()) {
        (Some(PgValue::Boolean(_)), Some(PgValue::Boolean(_))) => {
            matches!(op, BinaryOperator::Eq | BinaryOperator::NotEq)
        }
        (Some(_), Some(_)) => left.ty().is_number() && right.ty().is_number(),
        _ => false,
    };
    if !decidable {
        return Comparison::new(RowKinds::ANY, error);
    }
    let category = if matches!(left.constant(), Some(PgValue::Boolean(_))) {
        Category::Boolean
    } else {
        Category::Number
    };
    Comparison::new(pair_analysis(left, right, &category, op), error)
}

/// The kinds of answer a left value may give against the right values of two
/// operands in a resolved category.
pub(crate) fn pair_analysis(
    left: &Evaluation,
    right: &Evaluation,
    category: &Category,
    op: &BinaryOperator,
) -> RowKinds {
    // With no value on a side, no pair of values meets.
    if left.values().is_empty() || right.values().is_empty() {
        return RowKinds::NONE;
    }
    match category {
        Category::Number => number_pairs(left.values(), right.values(), op),
        Category::Boolean => {
            match op {
                BinaryOperator::Eq => boolean_pairs(left, right, true),
                BinaryOperator::NotEq => boolean_pairs(left, right, false),
                _ => RowKinds::NONE.with_pairs(true, true),
            }
        }
    }
}

/// The numeric value set and its bounds.
struct NumberSet<'a> {
    values: &'a Values,
}

impl<'a> NumberSet<'a> {
    /// The interval the set lies in, unbounded when it holds no number.
    fn bounds(&self) -> Interval<PgNumeric> {
        support::number_bounds(self.values)
    }

    /// The one value the set holds, if it holds exactly one.
    fn single(&self) -> Option<&'a PgValue> {
        match self.values {
            Values::Listed(listed) if listed.len() == 1 => listed.first(),
            _ => None,
        }
    }

    /// Whether no value of the set could be a value of `other`.
    fn disjoint(&self, other: &NumberSet<'a>) -> bool {
        match (self.values, other.values) {
            (Values::Listed(l), Values::Listed(r)) => l.iter().all(|value| !r.contains(value)),
            (Values::Listed(l), Values::Unlisted(u)) | (Values::Unlisted(u), Values::Listed(l)) => {
                l.iter().all(|value| !u.admits(value))
            }
            (Values::Unlisted(a), Values::Unlisted(b)) => {
                support::intervals_disjoint(&a.numbers, &b.numbers)
            }
        }
    }
}

/// Whether every value at or below the upper bound `high` is below every
/// value at or above the lower bound `low`, strictly when `strict` is set.
fn below(high: &Bound<PgNumeric>, low: &Bound<PgNumeric>, strict: bool) -> bool {
    match (high, low) {
        (Bound::Included(high), Bound::Included(low)) if strict => high < low,
        (
            Bound::Included(high) | Bound::Excluded(high),
            Bound::Included(low) | Bound::Excluded(low),
        ) => high <= low,
        _ => false,
    }
}

/// The kinds of answer a left value of one numeric set may give against
/// the other set under `op`.
fn number_pairs(left: &Values, right: &Values, op: &BinaryOperator) -> RowKinds {
    let left = NumberSet { values: left };
    let right = NumberSet { values: right };
    let Interval { low: llo, high: lhi } = left.bounds();
    let Interval { low: rlo, high: rhi } = right.bounds();
    let same = matches!((left.single(), right.single()), (Some(l), Some(r)) if l == r);
    // A row's comparison is certain against every member once it holds
    // strictly past the open bound, or at the closed one.
    let below_open = match &rlo {
        Bound::Unbounded => false,
        Bound::Included(value) => has_below(left.values, value, true),
        Bound::Excluded(value) => has_below(left.values, value, false),
    };
    let below_any = bound_value(&rlo).is_some_and(|value| has_below(left.values, value, false));
    let above_open = match &rhi {
        Bound::Unbounded => false,
        Bound::Included(value) => has_above(left.values, value, true),
        Bound::Excluded(value) => has_above(left.values, value, false),
    };
    let above_any = bound_value(&rhi).is_some_and(|value| has_above(left.values, value, false));
    let eq_match = number_eq_match(left.values, right.values);
    let eq_absent = number_eq_absent(left.values, right.values);
    let (every_satisfies, every_fails, row_satisfies, row_fails) = match op {
        BinaryOperator::Eq => (same, left.disjoint(&right), eq_match, eq_absent),
        BinaryOperator::NotEq => (left.disjoint(&right), same, eq_absent, eq_match),
        BinaryOperator::Lt => {
            (below(&lhi, &rlo, true), below(&rhi, &llo, false), below_open, above_any)
        }
        BinaryOperator::LtEq => {
            (below(&lhi, &rlo, false), below(&rhi, &llo, true), below_any, above_open)
        }
        BinaryOperator::Gt => {
            (below(&rhi, &llo, true), below(&lhi, &rlo, false), above_open, below_any)
        }
        BinaryOperator::GtEq => {
            (below(&rhi, &llo, false), below(&lhi, &rlo, true), above_any, below_open)
        }
        _ => (false, false, false, false),
    };
    RowKinds { all_true: row_satisfies, all_false: row_fails, mixed: false }
        .with_pairs(!every_fails, !every_satisfies)
}

/// The value a bound stands at, or `None` for the unbounded bound.
fn bound_value(bound: &Bound<PgNumeric>) -> Option<&PgNumeric> {
    match bound {
        Bound::Included(value) | Bound::Excluded(value) => Some(value),
        Bound::Unbounded => None,
    }
}

/// Whether some value of `left` is below `right`, strictly when `strict`,
/// which decides for an unlisted `left` by its lower bound.
fn has_below(left: &Values, right: &PgNumeric, strict: bool) -> bool {
    match left {
        Values::Listed(listed) => {
            listed
                .iter()
                .filter_map(PgValue::as_numeric)
                .any(|value| if strict { value < *right } else { value <= *right })
        }
        Values::Unlisted(unlisted) => {
            match &unlisted.numbers.low {
                Bound::Unbounded => true,
                Bound::Included(value) => {
                    if strict {
                        value < right
                    } else {
                        value <= right
                    }
                }
                Bound::Excluded(value) => value < right,
            }
        }
    }
}

/// Whether some value of `left` is above `right`, strictly when `strict`,
/// which decides for an unlisted `left` by its upper bound.
fn has_above(left: &Values, right: &PgNumeric, strict: bool) -> bool {
    match left {
        Values::Listed(listed) => {
            listed
                .iter()
                .filter_map(PgValue::as_numeric)
                .any(|value| if strict { value > *right } else { value >= *right })
        }
        Values::Unlisted(unlisted) => {
            match &unlisted.numbers.high {
                Bound::Unbounded => true,
                Bound::Included(value) => {
                    if strict {
                        value > right
                    } else {
                        value >= right
                    }
                }
                Bound::Excluded(value) => value > right,
            }
        }
    }
}

/// The one value a numeric set holds, or `None` when it holds several.
fn single_number(values: &Values) -> Option<PgNumeric> {
    match values {
        Values::Listed(listed) if listed.len() == 1 => listed.first().and_then(PgValue::as_numeric),
        Values::Unlisted(unlisted) => {
            let (Bound::Included(low), Bound::Included(high)) =
                (&unlisted.numbers.low, &unlisted.numbers.high)
            else {
                return None;
            };
            (low == high).then(|| low.clone())
        }
        Values::Listed(_) => None,
    }
}

/// Whether some value of `left` is certainly equal to the one value of
/// `right`, which a row then satisfies with every member.
fn number_eq_match(left: &Values, right: &Values) -> bool {
    let Some(right) = single_number(right) else {
        return false;
    };
    let datum = PgValue::Numeric(right);
    match left {
        Values::Listed(listed) => listed.contains(&datum),
        Values::Unlisted(unlisted) => unlisted.admits(&datum),
    }
}

/// Whether some value of `left` is certainly none of the values of `right`,
/// which a row then violates with every member.
fn number_eq_absent(left: &Values, right: &Values) -> bool {
    match right {
        Values::Listed(right) => {
            match left {
                Values::Listed(left) => left.iter().any(|value| !right.contains(value)),
                // A range with several values outruns a finite set.
                Values::Unlisted(_) => {
                    single_number(left)
                        .is_none_or(|value| !right.contains(&PgValue::Numeric(value.clone())))
                }
            }
        }
        Values::Unlisted(right) => {
            match left {
                Values::Listed(left) => left.iter().any(|value| !right.admits(value)),
                Values::Unlisted(_) => {
                    let (low, high) = (&right.numbers.low, &right.numbers.high);
                    let below = match low {
                        Bound::Unbounded => false,
                        Bound::Included(value) => has_below(left, value, true),
                        Bound::Excluded(value) => has_below(left, value, false),
                    };
                    let above = match high {
                        Bound::Unbounded => false,
                        Bound::Included(value) => has_above(left, value, true),
                        Bound::Excluded(value) => has_above(left, value, false),
                    };
                    below || above
                }
            }
        }
    }
}

/// The kinds of answer a left boolean may give against the right booleans
/// under the equality or the inequality. A boolean evaluation lists its
/// values, both when it holds them.
fn boolean_pairs(left: &Evaluation, right: &Evaluation, equality: bool) -> RowKinds {
    let holds = |a: bool, b: bool| if equality { a == b } else { a != b };
    let rights = [true, false].map(|b| (b, right.may_be(&PgValue::Boolean(b))));
    let held = rights.iter().filter(|(_, may)| *may).count();
    let mut rows = RowKinds::NONE;
    for a in [true, false] {
        if !left.may_be(&PgValue::Boolean(a)) {
            continue;
        }
        let satisfied = rights.iter().filter(|(b, may)| *may && holds(a, *b)).count();
        rows.all_true |= satisfied == held;
        rows.all_false |= satisfied == 0;
        rows.mixed |= satisfied != 0 && satisfied != held;
    }
    rows
}

/// The type a numeric binary operator resolves to.
pub(crate) fn arithmetic_type(left: ScalarType, right: ScalarType) -> ScalarType {
    match (left, right) {
        (a, b) if a.is_number() && b.is_number() => {
            match (a, b) {
                (ScalarType::Integer(a), ScalarType::Integer(b)) => ScalarType::Integer(a.wider(b)),
                _ => ScalarType::Numeric,
            }
        }
        (a, ScalarType::UnknownLiteral) if a.is_number() => a,
        (ScalarType::UnknownLiteral, b) if b.is_number() => b,
        (a, b) if a == ScalarType::Float || b == ScalarType::Float => ScalarType::Float,
        _ => ScalarType::Unknown,
    }
}

/// Whether the interval admits zero.
pub(crate) fn interval_contains_zero(interval: &Interval<PgNumeric>) -> bool {
    let zero = PgNumeric::from(0_i64);
    let low_ok = match &interval.low {
        Bound::Unbounded => true,
        Bound::Included(value) => *value <= zero,
        Bound::Excluded(value) => *value < zero,
    };
    let high_ok = match &interval.high {
        Bound::Unbounded => true,
        Bound::Included(value) => *value >= zero,
        Bound::Excluded(value) => *value > zero,
    };
    low_ok && high_ok
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use core::ops::Bound;

    use sql_values::{IntegerType, PgNumeric};
    use sqlparser::ast::BinaryOperator;

    use super::*;
    use crate::{Interval, ScalarType, Unlisted};

    fn listed(ty: ScalarType, values: Vec<PgValue>) -> Evaluation {
        Evaluation::new(
            ty,
            false,
            false,
            Values::Listed(values.into_iter().collect::<BTreeSet<PgValue>>()),
        )
    }

    fn int4(value: i64) -> PgValue {
        PgValue::Integer(value)
    }

    fn unlisted_numbers(low: Bound<PgNumeric>, high: Bound<PgNumeric>) -> Evaluation {
        Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval { low, high },
                length: Interval::unbounded(),
            }),
        )
    }

    /// A `numeric`-typed range normalisation leaves unlisted, so it keeps
    /// every width.
    fn numeric_range(low: Bound<PgNumeric>, high: Bound<PgNumeric>) -> Evaluation {
        Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval { low, high },
                length: Interval::unbounded(),
            }),
        )
    }

    #[test]
    fn undecidable_sides_keep_every_outcome_open() {
        let ty = ScalarType::Integer(IntegerType::Integer);
        let left = Evaluation::unknown(ty);
        let right = Evaluation::unknown(ty);
        let comparison = standard_comparison(&left, &BinaryOperator::Eq, &right, false);
        assert_eq!(comparison.rows, RowKinds::ANY);
    }

    #[test]
    fn boolean_literals_compare_by_value() {
        let left = listed(ScalarType::Boolean, vec![PgValue::Boolean(true)]);
        let right = listed(ScalarType::Boolean, vec![PgValue::Boolean(false)]);
        let comparison = standard_comparison(&left, &BinaryOperator::Eq, &right, false);
        assert!(!comparison.rows.may_be_true());
        assert!(comparison.rows.may_be_false());
        assert!(!comparison.rows.may_be_true());
        let comparison = standard_comparison(&left, &BinaryOperator::NotEq, &right, false);
        assert!(comparison.rows.may_be_true());
        assert!(!comparison.rows.may_be_false());
        assert!(!comparison.rows.may_be_false());
    }

    #[test]
    fn numeric_literals_compare_by_value() {
        let ty = ScalarType::Integer(IntegerType::Integer);
        let left = listed(ty, vec![int4(1), int4(2)]);
        let right = listed(ty, vec![int4(2)]);
        let comparison = standard_comparison(&left, &BinaryOperator::Eq, &right, false);
        assert!(comparison.rows.may_be_true());
        assert!(comparison.rows.may_be_false());
        let below = listed(ty, vec![int4(1)]);
        let above = listed(ty, vec![int4(3)]);
        let comparison = standard_comparison(&below, &BinaryOperator::Lt, &above, false);
        assert!(comparison.rows.may_be_true());
        assert!(!comparison.rows.may_be_false());
        assert!(!comparison.rows.may_be_false());
        let comparison = standard_comparison(&above, &BinaryOperator::Lt, &below, false);
        assert!(!comparison.rows.may_be_true());
        assert!(comparison.rows.may_be_false());
        assert!(!comparison.rows.may_be_true());
    }

    #[test]
    fn number_pairs_decide_the_row_certainties() {
        let below = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(1)]);
        let above = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(5)]);
        let rows = number_pairs(below.values(), above.values(), &BinaryOperator::Lt);
        assert!(!rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(rows.all_true);
        assert!(!rows.all_false);
        let rows = number_pairs(above.values(), below.values(), &BinaryOperator::Lt);
        assert!(rows.may_be_false());
        assert!(!rows.may_be_true());
        assert!(!rows.all_true);
        assert!(rows.all_false);
    }

    #[test]
    fn boolean_pairs_decide_the_row_certainties() {
        let left = listed(ScalarType::Boolean, vec![PgValue::Boolean(true)]);
        let right = listed(ScalarType::Boolean, vec![PgValue::Boolean(true)]);
        let rows = boolean_pairs(&left, &right, true);
        assert!(!rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(rows.all_true);
        assert!(!rows.all_false);
        let rows = boolean_pairs(&left, &right, false);
        assert!(rows.may_be_false());
        assert!(!rows.may_be_true());
        assert!(!rows.all_true);
        assert!(rows.all_false);
    }

    #[test]
    fn unlisted_bounds_decide_the_pair_analysis() {
        let below = unlisted_numbers(Bound::Unbounded, Bound::Included(PgNumeric::from(3)));
        let above = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(5)]);
        let rows = number_pairs(below.values(), above.values(), &BinaryOperator::Lt);
        assert!(!rows.may_be_false());
        assert!(rows.may_be_true());
        let rows = number_pairs(above.values(), below.values(), &BinaryOperator::Lt);
        assert!(rows.may_be_false());
        // No listed value is below every value the unlisted side admits
        assert!(!rows.may_be_true());
        // A listed value a single unlisted number admits is an equality match
        let single = unlisted_numbers(
            Bound::Included(PgNumeric::from(7)),
            Bound::Included(PgNumeric::from(7)),
        );
        let pair = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(7)]);
        assert!(number_eq_match(pair.values(), single.values()));
        let apart = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(8)]);
        assert!(number_eq_absent(apart.values(), single.values()));
        assert!(number_eq_absent(single.values(), apart.values()));
        // An unlisted side admits values the listed side cannot exclude
        let open = unlisted_numbers(Bound::Unbounded, Bound::Unbounded);
        assert!(number_eq_absent(open.values(), pair.values()));
    }

    #[test]
    fn arithmetic_types_widen_the_operands() {
        assert_eq!(
            arithmetic_type(
                ScalarType::Integer(IntegerType::Smallint),
                ScalarType::Integer(IntegerType::Integer)
            ),
            ScalarType::Integer(IntegerType::Integer)
        );
        assert_eq!(
            arithmetic_type(ScalarType::Integer(IntegerType::Integer), ScalarType::Numeric),
            ScalarType::Numeric
        );
        assert_eq!(
            arithmetic_type(ScalarType::UnknownLiteral, ScalarType::Integer(IntegerType::Bigint)),
            ScalarType::Integer(IntegerType::Bigint)
        );
        assert_eq!(arithmetic_type(ScalarType::Float, ScalarType::Numeric), ScalarType::Float);
        assert_eq!(arithmetic_type(ScalarType::Boolean, ScalarType::Boolean), ScalarType::Unknown);
    }

    #[test]
    fn zero_admits_the_divisor_bounds() {
        let admits = Interval {
            low: Bound::Included(PgNumeric::from(-1)),
            high: Bound::Included(PgNumeric::from(1)),
        };
        assert!(interval_contains_zero(&admits));
        let clear = Interval { low: Bound::Excluded(PgNumeric::from(0)), high: Bound::Unbounded };
        assert!(!interval_contains_zero(&clear));
        let open = Interval {
            low: Bound::Excluded(PgNumeric::from(0)),
            high: Bound::Excluded(PgNumeric::from(0)),
        };
        assert!(!interval_contains_zero(&open));
        let unbounded = Interval { low: Bound::Unbounded, high: Bound::Unbounded };
        assert!(interval_contains_zero(&unbounded));
    }

    #[test]
    fn the_unhandled_operators_leave_the_pair_analysis_open() {
        let side = listed(ScalarType::Boolean, vec![PgValue::Boolean(true)]);
        let rows = pair_analysis(&side, &side, &Category::Boolean, &BinaryOperator::Lt);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(!rows.all_true);
        assert!(!rows.all_false);
        assert_eq!(
            arithmetic_type(ScalarType::Integer(IntegerType::Integer), ScalarType::UnknownLiteral),
            ScalarType::Integer(IntegerType::Integer)
        );
    }

    #[test]
    fn number_pairs_walk_the_bound_shapes() {
        // An empty listed side meets no value.
        let empty = listed(ScalarType::Integer(IntegerType::Integer), Vec::new());
        let one = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(1)]);
        assert_eq!(
            pair_analysis(&empty, &one, &Category::Number, &BinaryOperator::Eq),
            RowKinds::NONE
        );

        // A listed row a range cannot reach is disjoint from it.
        let below_zero = unlisted_numbers(Bound::Unbounded, Bound::Included(PgNumeric::from(0)));
        let pair = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(1), int4(2)]);
        let rows = number_pairs(pair.values(), below_zero.values(), &BinaryOperator::Eq);
        assert!(rows.may_be_false());
        assert!(!rows.may_be_true());
        assert!(!rows.all_true);
        assert!(rows.all_false);

        // Ranges that admit no common value are disjoint through their
        // intervals.
        let left_range =
            numeric_range(Bound::Included(PgNumeric::from(1)), Bound::Included(PgNumeric::from(5)));
        let right_range = numeric_range(
            Bound::Included(PgNumeric::from(6)),
            Bound::Included(PgNumeric::from(10)),
        );
        let rows = number_pairs(left_range.values(), right_range.values(), &BinaryOperator::Eq);
        assert!(rows.may_be_false());
        assert!(!rows.may_be_true());

        // An unbounded side admits a value on either end of every row.
        let any_number = unlisted_numbers(Bound::Unbounded, Bound::Unbounded);
        let rows = number_pairs(one.values(), any_number.values(), &BinaryOperator::Lt);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(!rows.all_true);
        assert!(!rows.all_false);
        let rows = number_pairs(any_number.values(), one.values(), &BinaryOperator::Lt);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(rows.all_true);
        assert!(rows.all_false);

        // An excluded bound decides one step past its value.
        let above_five = unlisted_numbers(Bound::Excluded(PgNumeric::from(5)), Bound::Unbounded);
        let rows = number_pairs(one.values(), above_five.values(), &BinaryOperator::Lt);
        assert!(!rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(rows.all_true);
        assert!(!rows.all_false);
        let rows = number_pairs(pair.values(), below_zero.values(), &BinaryOperator::NotEq);
        assert!(!rows.may_be_false());
        assert!(rows.may_be_true());
        let rows = number_pairs(above_five.values(), one.values(), &BinaryOperator::LtEq);
        assert!(rows.may_be_false());
        assert!(!rows.may_be_true());

        // A numeric range keeps its unbounded ends past the type's clamping.
        let numeric_any = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval { low: Bound::Unbounded, high: Bound::Unbounded },
                length: Interval::unbounded(),
            }),
        );
        let rows = number_pairs(numeric_any.values(), one.values(), &BinaryOperator::Lt);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(rows.all_true);
        assert!(rows.all_false);
        let rows = number_pairs(one.values(), numeric_any.values(), &BinaryOperator::Gt);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(!rows.all_true);
        assert!(!rows.all_false);

        // The greater-than readings run the bounds the other way.
        let rows = number_pairs(above_five.values(), one.values(), &BinaryOperator::Gt);
        assert!(!rows.may_be_false());
        assert!(rows.may_be_true());
        assert!(rows.all_true);
        assert!(!rows.all_false);
        let rows = number_pairs(above_five.values(), one.values(), &BinaryOperator::GtEq);
        assert!(!rows.may_be_false());
        assert!(rows.may_be_true());
        let up_to_five = unlisted_numbers(Bound::Unbounded, Bound::Excluded(PgNumeric::from(5)));
        let rows = number_pairs(up_to_five.values(), one.values(), &BinaryOperator::Gt);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
        let rows = number_pairs(above_five.values(), one.values(), &BinaryOperator::Minus);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
    }

    #[test]
    fn single_number_reads_one_value_from_either_shape() {
        let one = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(7)]);
        assert_eq!(single_number(one.values()), Some(PgNumeric::from(7)));
        let pair = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(7), int4(8)]);
        assert_eq!(single_number(pair.values()), None);
        let point = unlisted_numbers(
            Bound::Included(PgNumeric::from(7)),
            Bound::Included(PgNumeric::from(7)),
        );
        assert_eq!(single_number(point.values()), Some(PgNumeric::from(7)));
        let range = unlisted_numbers(
            Bound::Included(PgNumeric::from(1)),
            Bound::Included(PgNumeric::from(9)),
        );
        assert_eq!(single_number(range.values()), None);

        assert!(!number_eq_match(one.values(), pair.values()));
        assert!(number_eq_match(one.values(), point.values()));
        assert!(number_eq_match(pair.values(), one.values()));
        let left_range = unlisted_numbers(
            Bound::Included(PgNumeric::from(1)),
            Bound::Included(PgNumeric::from(5)),
        );
        assert!(number_eq_absent(pair.values(), left_range.values()));
        assert!(!number_eq_absent(pair.values(), pair.values()));
        let open = unlisted_numbers(Bound::Unbounded, Bound::Unbounded);
        assert!(number_eq_absent(open.values(), pair.values()));
        assert!(!number_eq_absent(point.values(), pair.values()));
        let above_five = unlisted_numbers(Bound::Excluded(PgNumeric::from(5)), Bound::Unbounded);
        assert!(number_eq_absent(above_five.values(), left_range.values()));
    }

    #[test]
    fn listed_rows_disjoint_by_value() {
        let pair = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(1), int4(2)]);
        let one = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(1)]);
        let rows = number_pairs(pair.values(), one.values(), &BinaryOperator::Eq);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
        let up_to_five = unlisted_numbers(Bound::Unbounded, Bound::Excluded(PgNumeric::from(5)));
        let rows = number_pairs(one.values(), up_to_five.values(), &BinaryOperator::Gt);
        assert!(rows.may_be_false());
        assert!(rows.may_be_true());
    }

    #[test]
    fn number_eq_absent_walks_the_unlisted_bounds() {
        let below = numeric_range(Bound::Unbounded, Bound::Excluded(PgNumeric::from(3)));
        let above = numeric_range(Bound::Excluded(PgNumeric::from(7)), Bound::Unbounded);
        assert!(number_eq_absent(below.values(), above.values()));
        let up_to_three = numeric_range(Bound::Unbounded, Bound::Included(PgNumeric::from(3)));
        assert!(number_eq_absent(above.values(), up_to_three.values()));
        let between =
            numeric_range(Bound::Included(PgNumeric::from(1)), Bound::Excluded(PgNumeric::from(5)));
        assert!(number_eq_absent(below.values(), between.values()));
    }

    #[test]
    fn a_single_unlisted_value_is_not_absent() {
        let pair = listed(ScalarType::Integer(IntegerType::Integer), vec![int4(7), int4(8)]);
        let single_point =
            numeric_range(Bound::Included(PgNumeric::from(7)), Bound::Included(PgNumeric::from(7)));
        assert!(!number_eq_absent(single_point.values(), pair.values()));
    }

    #[test]
    fn an_empty_numeric_range_admits_nothing() {
        let empty_numbers =
            numeric_range(Bound::Included(PgNumeric::from(3)), Bound::Excluded(PgNumeric::from(3)));
        assert!(empty_numbers.is_impossible());
    }

    #[test]
    fn boolean_pairs_of_an_empty_side_meet_no_value() {
        let empty = listed(ScalarType::Boolean, Vec::new());
        let side = listed(ScalarType::Boolean, vec![PgValue::Boolean(true)]);
        assert_eq!(
            pair_analysis(&empty, &side, &Category::Boolean, &BinaryOperator::Eq),
            RowKinds::NONE
        );
        assert_eq!(
            pair_analysis(&side, &empty, &Category::Boolean, &BinaryOperator::Eq),
            RowKinds::NONE
        );
    }
}
