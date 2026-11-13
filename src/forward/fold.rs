//! Operators, forward: comparisons, arithmetic and string concatenation.

use alloc::collections::BTreeSet;
use core::{cmp::Ordering, ops::Bound};

use sql_values::postgres::{PgNumeric, PgValue};
use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator, Value};

use super::{
    super::{
        context::Context,
        dialect::EvalDialect,
        domain::{Evaluation, Interval, Outcomes, Unlisted, Values},
        guarantees::Guarantees,
        scalar::{self, ArrayElement, ScalarType, TextRules},
        support,
    },
    State, forward,
};

/// The category two operands compare in, with the text rules when both are
/// text, and the element type when both are arrays.
pub(crate) enum Category {
    Number,
    Boolean,
    Text(TextRules, TextRules),
    Timestamp,
    Array(ScalarType),
}

/// The comparison category of two resolved operand types.
pub(crate) fn comparison_category(left: ScalarType, right: ScalarType) -> Option<Category> {
    Some(match (left, right) {
        // A float compares against a number: PostgreSQL casts the number to
        // the float.
        (a, b)
            if (a.is_number() || a == ScalarType::Float)
                && (b.is_number() || b == ScalarType::Float) =>
        {
            Category::Number
        }
        (ScalarType::Boolean, ScalarType::Boolean) => Category::Boolean,
        (ScalarType::Text(a), ScalarType::Text(b)) => Category::Text(a, b),
        (ScalarType::Date | ScalarType::Timestamp, ScalarType::Date | ScalarType::Timestamp) => {
            Category::Timestamp
        }
        // PostgreSQL compares arrays of one element type only, without the
        // implicit casts between the elements.
        (ScalarType::Array(a), ScalarType::Array(b)) if same_element(a, b) => {
            Category::Array(a.scalar())
        }
        _ => return None,
    })
}

/// Whether two array element types are the one type an array comparison
/// takes, text of any collation being `text`.
fn same_element(left: ArrayElement, right: ArrayElement) -> bool {
    match (left, right) {
        (ArrayElement::Text(_), ArrayElement::Text(_))
        | (ArrayElement::Unknown | ArrayElement::Other, _)
        | (_, ArrayElement::Unknown | ArrayElement::Other) => true,
        _ => left == right,
    }
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

    /// The kinds against any non-empty subset of the right values, as a
    /// subquery's rows may hold any of the values its column may hold, so a
    /// pair that may satisfy is a subset satisfied throughout.
    #[must_use]
    pub(crate) fn over_subsets(self) -> Self {
        Self { all_true: self.may_be_true(), all_false: self.may_be_false(), mixed: self.mixed }
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

/// A dialect's comparison with the operands' `NULL` possibility, which
/// together decide its outcomes.
pub(crate) struct Compared {
    /// The kinds of answer a left value may give against the right values.
    pub(crate) rows: RowKinds,
    /// A side may be `NULL`.
    pub(crate) null: bool,
    /// Evaluating the operands or applying the operator may raise.
    pub(crate) error: bool,
}

impl Compared {
    /// The outcomes the comparison takes.
    pub(crate) fn outcomes(&self) -> Outcomes {
        Outcomes {
            truth: self.rows.may_be_true(),
            falsity: self.rows.may_be_false(),
            null: self.null,
        }
    }
}

/// The comparison of two already-evaluated operands under the dialect.
pub(super) fn comparison_evals<D: EvalDialect, C: Context>(
    state: &State<'_, D, C>,
    left: Evaluation,
    op: &BinaryOperator,
    right: Evaluation,
) -> Compared {
    if left.is_impossible() || right.is_impossible() {
        let error = left.may_raise() || right.may_raise();
        return Compared { rows: RowKinds::NONE, null: false, error };
    }
    let null = left.may_be_null() || right.may_be_null();
    let Comparison { rows, error } = state.dialect.comparison(left, op, right, state.options);
    Compared { rows, null, error }
}

/// The conservative reading of a comparison: number values and, for
/// equality, booleans compare by value, while the standard dialects type
/// untyped literals with their own affinity and compare text under their
/// own collation.
pub(crate) fn standard_comparison(
    left: &Evaluation,
    op: &BinaryOperator,
    right: &Evaluation,
    error: bool,
) -> Comparison {
    // A comparison operator is strict, so an always-`NULL` side meets no
    // value.
    if left.is_always_null() || right.is_always_null() {
        return Comparison::new(RowKinds::NONE, error || !comparable(left, right));
    }
    let category = if left.ty().is_number() && right.ty().is_number() {
        Some(Category::Number)
    } else if left.ty() == ScalarType::Boolean && right.ty() == ScalarType::Boolean {
        (matches!(op, BinaryOperator::Eq | BinaryOperator::NotEq)).then_some(Category::Boolean)
    } else {
        None
    };
    // The engine resolves the values' types under its own rules, so the
    // pair facts are undecidable; every outcome they support stays
    // possible.
    let Some(category) = category else {
        return Comparison::new(RowKinds::ANY, error || !comparable(left, right));
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
                _ => RowKinds::ANY,
            }
        }
        Category::Text(rules_left, rules_right) => {
            text_pairs(left.values(), right.values(), *rules_left, *rules_right, op)
        }
        // Text elements order under their collation, which decides only what
        // its rules allow, while other elements order by value, as dates and
        // timestamps do.
        Category::Array(ScalarType::Text(rules))
            if !(rules.identity_equality
                && (rules.byte_order
                    || matches!(op, BinaryOperator::Eq | BinaryOperator::NotEq))) =>
        {
            RowKinds::ANY
        }
        Category::Timestamp | Category::Array(_) => {
            value_order_pairs(left.values(), right.values(), op)
        }
    }
}

/// An untyped literal re-read as the type that resolved it, the way
/// PostgreSQL types an untyped literal against its use.
pub(crate) fn reinterpret_unknown(evaluation: Evaluation, target: ScalarType) -> Evaluation {
    if evaluation.ty() != ScalarType::UnknownLiteral || target.is_text() {
        return evaluation.with_type(target);
    }
    if let Values::Listed(listed) = evaluation.values() {
        let fold = scalar::cast_datums(listed, |datum| {
            scalar::cast(datum, ScalarType::UnknownLiteral, target)
        });
        let error = evaluation.may_raise() || fold.raises || fold.undecided;
        let values = if fold.undecided {
            Values::Unlisted(Unlisted::any())
        } else {
            Values::Listed(fold.values)
        };
        return Evaluation::new(target, evaluation.may_be_null(), error, values);
    }
    // An unlisted literal may hold text the target's input refuses.
    Evaluation::new(target, evaluation.may_be_null(), true, Values::Unlisted(Unlisted::any()))
}

/// Whether every engine has a comparison between the two sides, an untyped
/// literal only when its text reads as the other side's type.
fn comparable(left: &Evaluation, right: &Evaluation) -> bool {
    let reads_as = |literal: &Evaluation, other: &Evaluation| {
        match other.ty() {
            ScalarType::UnknownLiteral | ScalarType::Unknown | ScalarType::Text(_) => true,
            ScalarType::Float | ScalarType::Other => false,
            ty => !reinterpret_unknown(literal.clone(), ty).may_raise(),
        }
    };
    let number = |ty: ScalarType| ty.is_number() || ty == ScalarType::Float;
    let temporal = |ty: ScalarType| matches!(ty, ScalarType::Date | ScalarType::Timestamp);
    let (l, r) = (left.ty(), right.ty());
    match (l, r) {
        (ScalarType::UnknownLiteral, _) => reads_as(left, right),
        (_, ScalarType::UnknownLiteral) => reads_as(right, left),
        (ScalarType::Unknown, _) | (_, ScalarType::Unknown) => true,
        _ => {
            (number(l) && number(r))
                || (temporal(l) && temporal(r))
                || matches!(
                    (l, r),
                    (ScalarType::Text(_), ScalarType::Text(_))
                        | (ScalarType::Boolean, ScalarType::Boolean)
                )
                || matches!(
                    (l, r),
                    (ScalarType::Array(a), ScalarType::Array(b))
                        if same_element(a, b) && !a.is_text()
                )
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
        Bound::Included(value) => has_beyond(left.values, value, Ordering::Less, true),
        Bound::Excluded(value) => has_beyond(left.values, value, Ordering::Less, false),
    };
    let below_any = bound_value(&rlo)
        .is_some_and(|value| has_beyond(left.values, value, Ordering::Less, false));
    let above_open = match &rhi {
        Bound::Unbounded => false,
        Bound::Included(value) => has_beyond(left.values, value, Ordering::Greater, true),
        Bound::Excluded(value) => has_beyond(left.values, value, Ordering::Greater, false),
    };
    let above_any = bound_value(&rhi)
        .is_some_and(|value| has_beyond(left.values, value, Ordering::Greater, false));
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
        _ => return RowKinds::ANY,
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

/// Whether some value of `left` lies on the `side` of `right`, `Less` for
/// below and `Greater` for above, strictly when `strict`. An unlisted `left`
/// decides by its bound on that side.
fn has_beyond(left: &Values, right: &PgNumeric, side: Ordering, strict: bool) -> bool {
    let beyond = |value: &PgNumeric| {
        let order = value.cmp(right);
        order == side || (!strict && order == Ordering::Equal)
    };
    match left {
        Values::Listed(listed) => listed.iter().filter_map(PgValue::as_numeric).any(|v| beyond(&v)),
        Values::Unlisted(unlisted) => {
            let bound =
                if side == Ordering::Less { &unlisted.numbers.low } else { &unlisted.numbers.high };
            match bound {
                Bound::Unbounded => true,
                Bound::Included(value) => beyond(value),
                // An excluded bound holds values strictly past it.
                Bound::Excluded(value) => value.cmp(right) == side,
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
                        Bound::Included(value) => has_beyond(left, value, Ordering::Less, true),
                        Bound::Excluded(value) => has_beyond(left, value, Ordering::Less, false),
                    };
                    let above = match high {
                        Bound::Unbounded => false,
                        Bound::Included(value) => has_beyond(left, value, Ordering::Greater, true),
                        Bound::Excluded(value) => has_beyond(left, value, Ordering::Greater, false),
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
    let holds = |a: &bool, b: &bool| if equality { a == b } else { a != b };
    pair_certainties(booleans_held(left), &booleans_held(right), |a, b| {
        let satisfies = holds(a, b);
        (satisfies, !satisfies)
    })
}

/// The booleans a boolean evaluation may hold.
fn booleans_held(side: &Evaluation) -> impl Iterator<Item = bool> + Clone + '_ {
    [true, false].into_iter().filter(move |&value| side.may_be(&PgValue::Boolean(value)))
}

/// The kinds of answer a left value may give against the right values.
/// `judge` answers whether a pair certainly satisfies and whether it
/// certainly fails, which an undecidable pair answers with neither.
fn pair_certainties<A, B>(
    lefts: impl Iterator<Item = A>,
    rights: &(impl Iterator<Item = B> + Clone),
    judge: impl Fn(&A, &B) -> (bool, bool),
) -> RowKinds {
    let mut rows = RowKinds::NONE;
    for a in lefts {
        let (mut may_satisfy, mut may_fail) = (false, false);
        let (mut always_may_satisfy, mut always_may_fail) = (true, true);
        for b in rights.clone() {
            let (satisfies, fails) = judge(&a, &b);
            may_satisfy |= !fails;
            may_fail |= !satisfies;
            always_may_satisfy &= !fails;
            always_may_fail &= !satisfies;
        }
        rows = RowKinds {
            all_true: rows.all_true || always_may_satisfy,
            all_false: rows.all_false || always_may_fail,
            mixed: rows.mixed,
        }
        .with_pairs(may_satisfy, may_fail);
    }
    rows
}

/// The kinds of answer a left text may give against the right texts under
/// `op`, under the collations both sides carry.
fn text_pairs(
    left: &Values,
    right: &Values,
    rules_left: TextRules,
    rules_right: TextRules,
    op: &BinaryOperator,
) -> RowKinds {
    let identity = rules_left.identity_equality && rules_right.identity_equality;
    let byte_order = rules_left.byte_order && rules_right.byte_order;
    let (Values::Listed(l), Values::Listed(r)) = (left, right) else {
        // An unlisted left side admits values the listed right side cannot
        // exclude, so its rows take the kinds the listed values alone
        // admit, and every pair may satisfy and may fail.
        if let (Values::Unlisted(l), Values::Listed(r)) = (left, right)
            && !r.is_empty()
            && !l.length.is_empty()
        {
            let admits_member =
                r.len() == 1 && r.iter().next().is_some_and(|value| l.admits(value));
            // Identity equality alone stops one value matching two members.
            let (all_true, all_false) = match op {
                BinaryOperator::Eq => (!identity || admits_member, true),
                BinaryOperator::NotEq => (true, !identity || admits_member),
                _ => (true, true),
            };
            return RowKinds { all_true, all_false, mixed: false }.with_pairs(true, true);
        }
        return RowKinds::ANY;
    };
    // A byte-identical pair is certain under any collation; a differing one
    // only under an identity collation, and an ordering only under a byte
    // collation.
    let rights = r.iter().filter_map(PgValue::as_text);
    pair_certainties(l.iter().filter_map(PgValue::as_text), &rights, |a, b| {
        let ordered = |satisfies: bool| (satisfies, !satisfies);
        match op {
            BinaryOperator::Eq if a == b => (true, false),
            BinaryOperator::Eq => (false, identity),
            BinaryOperator::NotEq if a == b => (false, true),
            BinaryOperator::NotEq => (identity, false),
            BinaryOperator::Lt if byte_order => ordered(a.as_bytes() < b.as_bytes()),
            BinaryOperator::LtEq if byte_order => ordered(a.as_bytes() <= b.as_bytes()),
            BinaryOperator::Gt if byte_order => ordered(a.as_bytes() > b.as_bytes()),
            BinaryOperator::GtEq if byte_order => ordered(a.as_bytes() >= b.as_bytes()),
            _ => (false, false),
        }
    })
}

/// The kinds of answer a left value may give against the right ones under
/// `op`, for dates, timestamps and arrays, whose operators follow the value
/// order.
fn value_order_pairs(left: &Values, right: &Values, op: &BinaryOperator) -> RowKinds {
    let (Values::Listed(l), Values::Listed(r)) = (left, right) else {
        return RowKinds::ANY;
    };
    let satisfies: Option<fn(&PgValue, &PgValue) -> bool> = match op {
        BinaryOperator::Eq => Some(|a, b| a == b),
        BinaryOperator::NotEq => Some(|a, b| a != b),
        BinaryOperator::Lt => Some(|a, b| a < b),
        BinaryOperator::LtEq => Some(|a, b| a <= b),
        BinaryOperator::Gt => Some(|a, b| a > b),
        BinaryOperator::GtEq => Some(|a, b| a >= b),
        _ => None,
    };
    let Some(satisfies) = satisfies else {
        return RowKinds::ANY;
    };
    // A date compares with a timestamp as `date_cmp_timestamp` does, which
    // the value order follows.
    pair_certainties(l.iter(), &r.iter(), |a, b| {
        let satisfies = satisfies(a, b);
        (satisfies, !satisfies)
    })
}

/// The comparison of two expressions, forward.
pub(super) fn comparison<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    op: &BinaryOperator,
    right: &Expr,
) -> Result<Evaluation, C::Error> {
    let left = forward(state, left)?;
    let right = forward(state, right)?;
    let comparison = comparison_evals(state, left, op, right);
    Ok(Evaluation::condition(comparison.outcomes(), comparison.error))
}

/// `[ NOT ] BETWEEN`, forward.
pub(super) fn between<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
    low: &Expr,
    high: &Expr,
    negated: bool,
) -> Result<Evaluation, C::Error> {
    let x = forward(state, expr)?;
    let low = forward(state, low)?;
    let high = forward(state, high)?;
    let low_cmp = comparison_evals(state, x.clone(), &BinaryOperator::GtEq, low);
    let high_cmp = comparison_evals(state, x, &BinaryOperator::LtEq, high);
    let outcomes = support::and_outcomes(low_cmp.outcomes(), high_cmp.outcomes());
    let outcomes = if negated { support::not_outcomes(outcomes) } else { outcomes };
    Ok(Evaluation::condition(outcomes, low_cmp.error || high_cmp.error))
}

/// The operator of a binary expression, forward.
pub(super) fn binary<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    op: &BinaryOperator,
    right: &Expr,
) -> Result<Evaluation, C::Error> {
    match op {
        BinaryOperator::Eq => {
            if let Some(operand) = state.dialect.null_equals_operand(left, right, state.options) {
                return super::functions::is_null(state, operand, false);
            }
            comparison(state, left, op, right)
        }
        BinaryOperator::NotEq
        | BinaryOperator::Lt
        | BinaryOperator::LtEq
        | BinaryOperator::Gt
        | BinaryOperator::GtEq => comparison(state, left, op, right),
        BinaryOperator::StringConcat => left_deep(state, left, op, right, Class::Concat),
        BinaryOperator::Plus | BinaryOperator::Minus => {
            left_deep(state, left, op, right, Class::Additive)
        }
        BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Modulo => {
            left_deep(state, left, op, right, Class::Multiplicative)
        }
        BinaryOperator::And | BinaryOperator::Or => {
            let and = *op == BinaryOperator::And;
            and_or(state, &support::and_or_terms(left, right, and), and)
        }
        _ => {
            // An operator the reading does not model may raise or answer
            // `NULL`.
            forward(state, left)?;
            forward(state, right)?;
            Ok(Evaluation::new(ScalarType::Unknown, true, true, Values::Unlisted(Unlisted::any())))
        }
    }
}

/// The forward evaluation of the flat `AND`/`OR` chain of `terms`, walked
/// iteratively. While the path budget holds, a term is read under what each
/// outcome of the prefix seen so far implies, so the terms stay correlated;
/// past the budget, the terms read independently under the base guarantees.
fn and_or<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    terms: &[&Expr],
    and: bool,
) -> Result<Evaluation, C::Error> {
    let combine = if and { support::and_outcomes_set } else { support::or_outcomes_set };
    // The guarantees each outcome of the prefix seen so far implies, and
    // the guarantee of the prefix not being false, or not being true for an
    // `OR`, which the outcome updates draw on.
    let base = state.guarantees.clone();
    let mut true_guarantee = if and { base.clone() } else { Guarantees::impossible() };
    let mut false_guarantee = if and { Guarantees::impossible() } else { base.clone() };
    let mut null_guarantee = Guarantees::impossible();
    let mut helper_guarantee = base.clone();
    let mut outcomes = if and { Outcomes::TRUE } else { Outcomes::FALSE };
    let mut error = false;
    // A term that decides the chain may be read first, and the engine then
    // skips a term that would raise.
    let mut deciding = false;
    let mut first = true;
    for term in terms {
        // The first term has no prefix to read under, nor has a term after a
        // prefix that always raises, and each later term reads correlated
        // only while the budget holds.
        if !first && !outcomes.is_empty() && state.budget.get() > 0 {
            state.budget.set(state.budget.get() - 1);
            let mut next_outcomes = Outcomes::NONE;
            // The term is read once per outcome the prefix may take, under
            // what that outcome implies, so the sides correlate.
            for (single, guarantee) in [
                (Outcomes::TRUE, &true_guarantee),
                (Outcomes::FALSE, &false_guarantee),
                (Outcomes::NULL, &null_guarantee),
            ] {
                if single.intersection(outcomes).is_empty() {
                    continue;
                }
                let evaluation = support::as_condition(super::evaluate_within(
                    term,
                    state.dialect,
                    state.options,
                    state.context,
                    guarantee,
                    state.budget,
                )?);
                let evaluation_outcomes = evaluation.outcomes();
                deciding |=
                    if and { evaluation_outcomes.falsity } else { evaluation_outcomes.truth };
                for reading in support::OUTCOMES {
                    if reading.intersection(evaluation_outcomes).is_empty() {
                        continue;
                    }
                    next_outcomes = next_outcomes.union(combine(single, reading));
                }
                error |= evaluation.may_raise();
            }
            outcomes = next_outcomes;
        } else {
            let evaluation = support::as_condition(super::evaluate_within(
                term,
                state.dialect,
                state.options,
                state.context,
                state.guarantees,
                state.budget,
            )?);
            let evaluation_outcomes = evaluation.outcomes();
            deciding |= if and { evaluation_outcomes.falsity } else { evaluation_outcomes.truth };
            outcomes = combine(outcomes, evaluation_outcomes);
            error |= evaluation.may_raise();
        }
        first = false;
        // The running guarantees feed the next term's correlated reads.
        if state.budget.get() > 0 {
            if and {
                let next_true = narrow_within(state, term, &true_guarantee, Outcomes::TRUE)?;
                let next_null = Guarantees::join(
                    &narrow_within(state, term, &null_guarantee, Outcomes::NOT_FALSE)?,
                    &narrow_within(state, term, &true_guarantee, Outcomes::NULL)?,
                );
                let next_false = Guarantees::join(
                    &false_guarantee,
                    &narrow_within(state, term, &helper_guarantee, Outcomes::FALSE)?,
                );
                let next_helper =
                    narrow_within(state, term, &helper_guarantee, Outcomes::NOT_FALSE)?;
                (true_guarantee, false_guarantee, null_guarantee, helper_guarantee) =
                    (next_true, next_false, next_null, next_helper);
            } else {
                let next_false = narrow_within(state, term, &false_guarantee, Outcomes::FALSE)?;
                // The term alone may be the true one, so it is read under
                // the base, not the prefix.
                let next_true = Guarantees::join(
                    &true_guarantee,
                    &narrow_within(state, term, &base, Outcomes::TRUE)?,
                );
                let next_null = Guarantees::join(
                    &narrow_within(state, term, &null_guarantee, Outcomes::NOT_TRUE)?,
                    &narrow_within(state, term, &helper_guarantee, Outcomes::NULL)?,
                );
                let next_helper =
                    narrow_within(state, term, &helper_guarantee, Outcomes::NOT_TRUE)?;
                (true_guarantee, false_guarantee, null_guarantee, helper_guarantee) =
                    (next_true, next_false, next_null, next_helper);
            }
        }
    }
    if deciding {
        outcomes = outcomes.union(if and { Outcomes::FALSE } else { Outcomes::TRUE });
    }
    Ok(Evaluation::condition(outcomes, error))
}

/// [`backward::implied_within`](super::super::backward::implied_within) on one
/// term, drawing on the pass's budget.
fn narrow_within<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    term: &Expr,
    base: &Guarantees<C::Reference>,
    single: Outcomes,
) -> Result<Guarantees<C::Reference>, C::Error> {
    super::super::backward::implied_within(
        term,
        single,
        state.dialect,
        state.options,
        state.context,
        base,
        state.budget,
    )
}

/// The unary operator of an expression, forward.
pub(super) fn unary<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    op: UnaryOperator,
    expr: &Expr,
) -> Result<Evaluation, C::Error> {
    match op {
        UnaryOperator::Not => {
            let inner = support::as_condition(forward(state, expr)?);
            if inner.is_impossible() {
                return Ok(Evaluation::impossible(ScalarType::Boolean));
            }
            let outcomes = inner.outcomes();
            Ok(Evaluation::condition(
                Outcomes { truth: outcomes.falsity, falsity: outcomes.truth, null: outcomes.null },
                inner.may_raise(),
            ))
        }
        UnaryOperator::Minus => negate(state, expr),
        UnaryOperator::Plus => {
            let inner = forward(state, expr)?;
            if state.dialect.unary_plus_preserves(&inner, state.options) {
                return Ok(inner);
            }
            Ok(Evaluation::new(
                inner.ty(),
                inner.may_be_null(),
                inner.may_raise() || support::sign_may_raise(&inner),
                Values::Unlisted(Unlisted::any()),
            ))
        }
        _ => {
            // An operator the reading does not model may raise or answer
            // `NULL`.
            forward(state, expr)?;
            Ok(Evaluation::new(ScalarType::Unknown, true, true, Values::Unlisted(Unlisted::any())))
        }
    }
}

/// The precedence class a flat left-nested chain of binary operators walks
/// over; operators of a class fold left-associatively along their own spine.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Additive,
    Multiplicative,
    Concat,
}

/// The chain class of `op`, when it walks a left-deep spine.
fn chain_class(op: &BinaryOperator) -> Option<Class> {
    match op {
        BinaryOperator::Plus | BinaryOperator::Minus => Some(Class::Additive),
        BinaryOperator::Multiply | BinaryOperator::Divide | BinaryOperator::Modulo => {
            Some(Class::Multiplicative)
        }
        BinaryOperator::StringConcat => Some(Class::Concat),
        _ => None,
    }
}

/// The forward evaluation of a flat left-nested chain of `op` and the same
/// precedence class, walked iteratively so a long spine keeps the stack flat:
/// the leftmost operand is read once, then the (operator, right operand)
/// pairs fold upward.
fn left_deep<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    op: &BinaryOperator,
    right: &Expr,
    class: Class,
) -> Result<Evaluation, C::Error> {
    // The spine's (operator, right operand) pairs read top-down, the root
    // first; the walk stops at a different class or a non-operator.
    let mut pairs: alloc::vec::Vec<(&BinaryOperator, &Expr)> = alloc::vec::Vec::new();
    let mut spine = support::unnest(left);
    pairs.push((op, right));
    loop {
        let node = support::unnest(spine);
        let Expr::BinaryOp { left: l, op: o, right: r } = node else {
            break;
        };
        if chain_class(o) != Some(class) {
            break;
        }
        pairs.push((o, support::unnest(r)));
        spine = support::unnest(l);
    }
    let mut acc = forward(state, support::unnest(spine))?;
    for (o, r) in pairs.iter().rev() {
        let right = forward(state, r)?;
        acc = match class {
            Class::Additive | Class::Multiplicative => arithmetic_combine(state, acc, o, right),
            Class::Concat => concat_combine(state, &acc, &right),
        };
    }
    Ok(acc)
}

/// The combination of two already-evaluated operands of `left || right`.
fn concat_combine<D: EvalDialect, C: Context>(
    state: &State<'_, D, C>,
    left: &Evaluation,
    right: &Evaluation,
) -> Evaluation {
    if left.is_uninformative() || right.is_uninformative() {
        return Evaluation::unknown(ScalarType::Unknown);
    }
    let error = left.may_raise() || right.may_raise();
    let null = left.may_be_null() || right.may_be_null();
    if left.is_impossible() || right.is_impossible() {
        return Evaluation::impossible(ScalarType::Text(TextRules::DEFAULT));
    }
    if let Some(array) = super::arrays::concat(left, right) {
        return array;
    }
    // An untyped `NULL` takes the text side `||` needs.
    let is_text = |evaluation: &Evaluation| {
        evaluation.ty().is_text()
            || evaluation.ty() == ScalarType::UnknownLiteral
            || evaluation.is_untyped_null()
    };
    if !is_text(left) && !is_text(right) {
        // Neither side is text, so `||` has no type to resolve to.
        return Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            null,
            true,
            Values::Unlisted(Unlisted::any()),
        );
    }
    // `||` is strict: a `NULL` side nulls the result.
    if left.is_always_null() || right.is_always_null() {
        return Evaluation::null(ScalarType::Text(TextRules::DEFAULT)).with_error(error);
    }
    if let Some(folded) = state.dialect.concat(left, right, state.options) {
        return folded;
    }
    Evaluation::new(
        ScalarType::Text(TextRules::DEFAULT),
        null,
        error,
        Values::Unlisted(Unlisted::any()),
    )
}

/// The combination of two already-evaluated operands of a numeric operator,
/// deferring the fold and the bounds to the dialect.
fn arithmetic_combine<D: EvalDialect, C: Context>(
    state: &State<'_, D, C>,
    left: Evaluation,
    op: &BinaryOperator,
    right: Evaluation,
) -> Evaluation {
    if left.is_uninformative() || right.is_uninformative() {
        return Evaluation::unknown(ScalarType::Unknown);
    }
    let (left, right) = arithmetic_sides(left, right);
    if left.is_impossible() || right.is_impossible() {
        return Evaluation::impossible(ScalarType::Unknown);
    }
    state.dialect.arithmetic(&left, op, &right, state.options)
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

/// The arithmetic operands with untyped literals resolved to the type of the
/// other operand, or to the default `int4` when both are untyped, the way
/// PostgreSQL types them.
fn arithmetic_sides(left: Evaluation, right: Evaluation) -> (Evaluation, Evaluation) {
    let left_ty = left.ty();
    let right_ty = right.ty();
    match (left_ty, right_ty) {
        (ScalarType::UnknownLiteral, right_ty) if right_ty.is_number() => {
            (reinterpret_unknown(left, right_ty), right)
        }
        (left_ty, ScalarType::UnknownLiteral) if left_ty.is_number() => {
            (left, reinterpret_unknown(right, left_ty))
        }
        // An engine may find no unique operator for two untyped literals.
        (ScalarType::UnknownLiteral, ScalarType::UnknownLiteral) => {
            let integer = ScalarType::Integer(sql_values::postgres::IntegerType::Integer);
            (
                reinterpret_unknown(left, integer).with_error(true),
                reinterpret_unknown(right, integer),
            )
        }
        _ => (left, right),
    }
}

/// The bound negated.
pub(crate) fn neg_bound(bound: &Bound<PgNumeric>) -> Bound<PgNumeric> {
    match bound {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(value) => Bound::Included(-value.clone()),
        Bound::Excluded(value) => Bound::Excluded(-value.clone()),
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

/// The forward evaluation of `-expr`.
fn negate<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
) -> Result<Evaluation, C::Error> {
    // A minus chain over a number literal folds into the literal, typed the
    // way PostgreSQL's `make_const` types the signed constant.
    let mut node = support::unnest(expr);
    let mut negated = true;
    while let Expr::UnaryOp { op: UnaryOperator::Minus, expr: inner } = node {
        negated = !negated;
        node = support::unnest(inner);
    }
    if let Expr::Value(value) = node
        && let Value::Number(number, _) = &value.value
        && let Some((value, ty)) = scalar::number_literal(number, negated)
    {
        return Ok(Evaluation::known(value, ty));
    }
    let inner = forward(state, expr)?;
    let ty = inner.ty();
    if !ty.is_number() {
        // A `float` negates to a `float` the reading does not fold.
        let negated = if ty == ScalarType::Float { ty } else { ScalarType::Unknown };
        return Ok(Evaluation::new(
            negated,
            inner.may_be_null(),
            inner.may_raise() || support::sign_may_raise(&inner),
            Values::Unlisted(Unlisted::any()),
        ));
    }
    if inner.is_impossible() {
        return Ok(Evaluation::impossible(ty));
    }
    let null = inner.may_be_null();
    let error0 = inner.may_raise();
    match inner.values() {
        Values::Listed(listed) => {
            let mut out = BTreeSet::new();
            let mut error = error0;
            for datum in listed {
                // A numeric type lists only numeric datums.
                if let PgValue::Integer(value) = datum {
                    if let ScalarType::Integer(target) = ty
                        && *value == *target.range().start()
                    {
                        error = true;
                        continue;
                    }
                    // The negation of the `i64` minimum lies in the wider type.
                    let negated = value.checked_neg().map_or_else(
                        || PgValue::Numeric(-PgNumeric::from(*value)),
                        PgValue::Integer,
                    );
                    out.insert(negated);
                }
                if let PgValue::Numeric(numeric) = datum {
                    out.insert(PgValue::Numeric(-numeric.clone()));
                }
            }
            Ok(Evaluation::new(ty, null, error, Values::Listed(out)))
        }
        Values::Unlisted(unlisted) => {
            let (low, high) = (neg_bound(&unlisted.numbers.high), neg_bound(&unlisted.numbers.low));
            let mut error = error0;
            if let ScalarType::Integer(target) = ty {
                let min = PgNumeric::from(*target.range().start());
                if unlisted.numbers.contains(&min) {
                    error = true;
                }
            }
            let mut narrowed = unlisted.clone();
            narrowed.numbers = Interval { low, high };
            Ok(Evaluation::new(ty, null, error, Values::Unlisted(narrowed)))
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use core::{cell::Cell, ops::Bound};

    use sql_values::postgres::{IntegerType, PgNumeric, PgValue};
    use sqlparser::{
        ast::{BinaryOperator, Expr, UnaryOperator, Value, ValueWithSpan},
        dialect::PostgreSqlDialect,
        tokenizer::Span,
    };

    use super::{
        Category, RowKinds, arithmetic_type, interval_contains_zero, pair_analysis,
        standard_comparison,
    };
    use crate::{
        domain::{Evaluation, Interval, Unlisted, Values},
        forward::{evaluate, evaluate_within},
        guarantees::{Guarantees, Subject},
        postgres::PostgreSqlOptions,
        scalar::{ScalarType, TextRules},
        test_context::{ColumnContext, parse, read, read_default},
    };

    fn read_guarantees(
        mut context: ColumnContext,
        guarantees: &Guarantees<String>,
        sql: &str,
    ) -> Evaluation {
        evaluate(
            &parse(sql),
            &PostgreSqlDialect {},
            &PostgreSqlOptions::default(),
            &mut context,
            guarantees,
        )
        .expect("resolves")
    }

    fn int4(values: Values) -> Evaluation {
        Evaluation::new(ScalarType::Integer(IntegerType::Integer), true, false, values)
    }

    fn int4_unlisted() -> Evaluation {
        int4(Values::Unlisted(Unlisted::any()))
    }

    fn numeric_unlisted() -> Evaluation {
        Evaluation::new(ScalarType::Numeric, true, false, Values::Unlisted(Unlisted::any()))
    }

    fn point(value: i64) -> Evaluation {
        Evaluation::known(PgValue::Integer(value), ScalarType::Integer(IntegerType::Integer))
    }

    fn numbers(interval: Interval<PgNumeric>) -> Evaluation {
        Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted { numbers: interval, ..Unlisted::any() }),
        )
    }

    fn number_interval(low: i64, high: i64) -> Interval<PgNumeric> {
        Interval {
            low: Bound::Included(PgNumeric::from(low)),
            high: Bound::Included(PgNumeric::from(high)),
        }
    }

    fn bound(low: &str, low_value: i64, high: &str, high_value: i64) -> Interval<PgNumeric> {
        let bound = |mode: &str, value: i64| {
            match mode {
                "i" => Bound::Included(PgNumeric::from(value)),
                "e" => Bound::Excluded(PgNumeric::from(value)),
                _ => Bound::Unbounded,
            }
        };
        Interval { low: bound(low, low_value), high: bound(high, high_value) }
    }

    fn guarantees_for(
        x: &str,
        ty: ScalarType,
        interval: Interval<PgNumeric>,
    ) -> Guarantees<String> {
        let mut guarantees = Guarantees::new();
        guarantees.restrict(
            Subject::Reference(x.to_string()),
            Evaluation::new(
                ty,
                false,
                false,
                Values::Unlisted(Unlisted { numbers: interval, ..Unlisted::any() }),
            ),
        );
        guarantees
    }

    #[test]
    fn standard_comparison_of_columns_stays_open() {
        let result = read_default(
            ColumnContext::new()
                .with_column("a", int4_unlisted())
                .with_column("b", int4_unlisted()),
            "a = b",
        );
        assert!(result.may_be_true());
        assert!(result.may_be_false());
        assert!(result.may_be_null());
        assert!(!result.may_raise());
    }

    /// A resolved category computes each pair certainty from the values,
    /// the facts `ANY` and `ALL` read for their outcomes.
    #[test]
    fn a_resolved_comparison_computes_the_pair_certainties() {
        let one = point(1);
        let two = point(2);
        // every non-`NULL` pair satisfies, closing `ALL`'s falsity
        assert!(!standard_comparison(&one, &BinaryOperator::Eq, &one, false).rows.may_be_false());
        // no non-`NULL` pair satisfies, closing `ANY`'s truth
        assert!(!standard_comparison(&one, &BinaryOperator::Eq, &two, false).rows.may_be_true());
        // some left value satisfies every right value, keeping `ALL`'s
        // `NULL` reading when a `NULL` member is possible
        assert!(standard_comparison(&one, &BinaryOperator::Eq, &one, false).rows.all_true);
        // some left value satisfies no right value, keeping `ANY`'s
        // `NULL` reading when a `NULL` member is possible
        assert!(standard_comparison(&one, &BinaryOperator::Eq, &two, false).rows.all_false);
    }

    /// An unresolved category leaves every outcome the pair facts support
    /// open.
    #[test]
    fn an_unresolved_comparison_leaves_every_pair_fact_open() {
        let text = |value: &str| {
            Evaluation::new(
                ScalarType::Text(TextRules::DEFAULT),
                false,
                false,
                Values::Listed(BTreeSet::from([PgValue::Text(value.to_string())])),
            )
        };
        let comparison = standard_comparison(&text("a"), &BinaryOperator::Eq, &text("b"), false);
        assert_eq!(comparison.rows, RowKinds::ANY);
    }

    #[test]
    fn number_pairs_certainties_follow_the_bounds() {
        // a closed interval below the other admits every pair
        assert_eq!(
            pair_analysis(&point(1), &point(2), &Category::Number, &BinaryOperator::Lt),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        // a closed interval above the other admits no pair
        assert_eq!(
            pair_analysis(&point(2), &point(1), &Category::Number, &BinaryOperator::Lt),
            RowKinds { all_true: false, all_false: true, mixed: false },
        );
        // an excluded lower bound decides a strict row certainty, and the
        // unbounded upper bound keeps the opposite one possible
        let below = numbers(bound("e", 1, "_", 0));
        assert_eq!(
            pair_analysis(&below, &point(3), &Category::Number, &BinaryOperator::Lt),
            RowKinds { all_true: true, all_false: true, mixed: false },
        );
        // a value below any right value admits every strict pair
        assert_eq!(
            pair_analysis(
                &numbers(bound("_", 0, "i", 0)),
                &point(3),
                &Category::Number,
                &BinaryOperator::Lt,
            ),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        // an unbounded right range admits no strict certainty
        assert_eq!(
            pair_analysis(
                &point(0),
                &numbers(bound("_", 0, "_", 0)),
                &Category::Number,
                &BinaryOperator::Lt,
            ),
            RowKinds { all_true: false, all_false: false, mixed: true },
        );
        // every value is above an excluded upper bound
        assert_eq!(
            pair_analysis(
                &point(5),
                &numbers(number_interval(1, 3)),
                &Category::Number,
                &BinaryOperator::Gt
            ),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        // an excluded upper bound decides a strict row certainty
        let above = numbers(bound("e", 1, "e", 10));
        assert_eq!(
            pair_analysis(&above, &point(3), &Category::Number, &BinaryOperator::Gt),
            RowKinds { all_true: true, all_false: true, mixed: false },
        );
    }

    #[test]
    fn unlisted_intervals_admit_pair_certainties() {
        // disjoint unlisted intervals admit no equal pair, and a value
        // below the right range is certainly unequal
        assert_eq!(
            pair_analysis(
                &numbers(number_interval(0, 2)),
                &numbers(number_interval(3, 5)),
                &Category::Number,
                &BinaryOperator::Eq,
            ),
            RowKinds { all_true: false, all_false: true, mixed: false },
        );
        // overlapping unlisted intervals: a value below the right range is
        // certainly unequal
        assert_eq!(
            pair_analysis(
                &numbers(number_interval(0, 2)),
                &numbers(number_interval(1, 3)),
                &Category::Number,
                &BinaryOperator::NotEq,
            ),
            RowKinds { all_true: true, all_false: false, mixed: true },
        );
        // an excluded right bound still admits a certain difference
        assert_eq!(
            pair_analysis(
                &numbers(number_interval(0, 2)),
                &numbers(bound("e", 1, "e", 3)),
                &Category::Number,
                &BinaryOperator::NotEq,
            ),
            RowKinds { all_true: true, all_false: false, mixed: true },
        );
        // an unbounded right range admits no certain difference
        assert_eq!(
            pair_analysis(
                &numbers(number_interval(0, 2)),
                &numbers(bound("_", 0, "_", 0)),
                &Category::Number,
                &BinaryOperator::NotEq,
            ),
            RowKinds { all_true: false, all_false: false, mixed: true },
        );
        // a listed value outside an unlisted range is different from every
        // member of it, so no pair is ever equal
        assert_eq!(
            pair_analysis(
                &point(5),
                &numbers(number_interval(0, 3)),
                &Category::Number,
                &BinaryOperator::NotEq,
            ),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        // a single point of an unlisted range matches its value
        assert_eq!(
            pair_analysis(
                &point(2),
                &numbers(number_interval(2, 2)),
                &Category::Number,
                &BinaryOperator::Eq
            ),
            RowKinds { all_true: true, all_false: false, mixed: true },
        );
        // a numeric value equal to an unlisted point
        let numeric_point =
            Evaluation::known(PgValue::Numeric(PgNumeric::from(2)), ScalarType::Numeric);
        assert_eq!(
            pair_analysis(
                &numeric_point,
                &numbers(number_interval(2, 2)),
                &Category::Number,
                &BinaryOperator::Eq
            ),
            RowKinds { all_true: true, all_false: false, mixed: true },
        );
    }

    #[test]
    fn an_empty_value_set_meets_nothing_and_a_boolean_ordering_stays_open() {
        // an empty value set meets no value
        let empty =
            Evaluation::new(ScalarType::Integer(IntegerType::Integer), false, true, Values::none());
        assert_eq!(
            pair_analysis(&empty, &point(1), &Category::Number, &BinaryOperator::Eq),
            RowKinds::NONE,
        );
        // a boolean pair under an ordering leaves every row open
        let flag = Evaluation::new(
            ScalarType::Boolean,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Boolean(true)])),
        );
        assert_eq!(
            pair_analysis(&flag, &flag, &Category::Boolean, &BinaryOperator::Lt),
            RowKinds::ANY,
        );
        // an empty boolean side meets no value either
        let none = Evaluation::impossible(ScalarType::Boolean);
        assert_eq!(
            pair_analysis(&flag, &none, &Category::Boolean, &BinaryOperator::Eq),
            RowKinds::NONE,
        );
    }

    #[test]
    fn boolean_equality_pairs_decide_per_value() {
        let known = |value: bool| Evaluation::known(PgValue::Boolean(value), ScalarType::Boolean);
        assert_eq!(
            pair_analysis(&known(true), &known(true), &Category::Boolean, &BinaryOperator::Eq),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        assert_eq!(
            pair_analysis(&known(true), &known(false), &Category::Boolean, &BinaryOperator::Eq),
            RowKinds { all_true: false, all_false: true, mixed: false },
        );
        let both = Evaluation::new(
            ScalarType::Boolean,
            true,
            false,
            Values::Listed(BTreeSet::from([PgValue::Boolean(false), PgValue::Boolean(true)])),
        );
        assert_eq!(
            pair_analysis(&both, &known(true), &Category::Boolean, &BinaryOperator::Eq),
            RowKinds { all_true: true, all_false: true, mixed: false },
        );
        assert_eq!(
            pair_analysis(&both, &both, &Category::Boolean, &BinaryOperator::NotEq),
            RowKinds { all_true: false, all_false: false, mixed: true },
        );
    }

    #[test]
    fn and_or_correlate_the_sides() {
        let context = || ColumnContext::new().with_column("x", int4_unlisted());
        let result = read(context(), "x IS NULL OR x IS NOT NULL");
        assert!(result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_be_null());
        let result = read(context(), "x IS NULL AND x IS NOT NULL");
        assert!(!result.may_be_true());
        assert!(result.may_be_false());
        assert!(!result.may_be_null());
    }

    #[test]
    fn a_depleted_path_budget_stays_sound() {
        let budget = Cell::new(1);
        let expression = parse("x = 1 AND x = 2 AND x = 3");
        let result = evaluate_within(
            &expression,
            &PostgreSqlDialect {},
            &PostgreSqlOptions::default(),
            &mut ColumnContext::new().with_column("x", int4_unlisted()),
            &Guarantees::new(),
            &budget,
        )
        .expect("resolves");
        assert!(!result.may_be_true());
        assert!(result.may_be_false());
        assert!(result.may_be_null());
    }

    fn number_value(token: &str) -> Expr {
        Expr::Value(ValueWithSpan {
            value: Value::Number(token.to_string(), false),
            span: Span::empty(),
        })
    }

    #[test]
    fn an_unmodelled_binary_operator_stays_unknown() {
        let expression = Expr::BinaryOp {
            left: Box::new(number_value("1")),
            op: BinaryOperator::Xor,
            right: Box::new(number_value("2")),
        };
        let result = evaluate(
            &expression,
            &PostgreSqlDialect {},
            &PostgreSqlOptions::default(),
            &mut ColumnContext::new(),
            &Guarantees::new(),
        )
        .expect("resolves");
        assert_eq!(result.ty(), ScalarType::Unknown);
        assert!(matches!(result.values(), Values::Unlisted(_)));
        assert!(result.may_be_null() && result.may_raise());
    }

    #[test]
    fn not_flips_the_outcomes() {
        let result = read(ColumnContext::new(), "NOT true");
        assert!(result.may_be_false());
        assert!(!result.may_be_true());
        assert!(!result.may_be_null());
        let result = read(ColumnContext::new(), "NOT NULL");
        assert!(result.may_be_null());
        assert!(!result.may_be_true());
        assert!(!result.may_be_false());
        let flag =
            Evaluation::new(ScalarType::Boolean, true, false, Values::Unlisted(Unlisted::any()));
        let result = read(ColumnContext::new().with_column("flag", flag), "NOT flag");
        assert!(result.may_be_true());
        assert!(result.may_be_false());
        assert!(result.may_be_null());
        let impossible = Evaluation::impossible(ScalarType::Boolean);
        let result = read(ColumnContext::new().with_column("flag", impossible), "NOT flag");
        assert!(result.is_impossible());
    }

    #[test]
    fn unary_plus_keeps_a_number_and_opens_other_types() {
        let result = read(ColumnContext::new(), "+5");
        assert_eq!(result.constant(), Some(&PgValue::Integer(5)));
        let result = read(ColumnContext::new().with_column("x", int4_unlisted()), "+x");
        assert!(matches!(result.values(), Values::Unlisted(_)));
        assert!(!result.may_raise());
        // the standard dialect does not fold the sign
        let result = read_default(ColumnContext::new().with_column("x", int4_unlisted()), "+x");
        assert!(matches!(result.values(), Values::Unlisted(_)));
        // postgres does not know + over a non-number, so it keeps the type
        // and opens the values
        let s = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        let result = read(ColumnContext::new().with_column("s", s), "+s");
        assert_eq!(result.ty(), ScalarType::Text(TextRules::DEFAULT));
        assert!(matches!(result.values(), Values::Unlisted(_)));
    }

    #[test]
    fn an_unmodelled_unary_operator_stays_unknown() {
        let expression = Expr::UnaryOp {
            op: UnaryOperator::Hash,
            expr: Box::new(Expr::Value(ValueWithSpan {
                value: Value::Boolean(true),
                span: Span::empty(),
            })),
        };
        let result = evaluate(
            &expression,
            &PostgreSqlDialect {},
            &PostgreSqlOptions::default(),
            &mut ColumnContext::new(),
            &Guarantees::new(),
        )
        .expect("resolves");
        assert_eq!(result.ty(), ScalarType::Unknown);
        assert!(result.may_be_null() && result.may_raise());
    }

    fn text_unlisted() -> Evaluation {
        Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        )
    }

    #[test]
    fn negation_folds_and_bounds() {
        let result = read(ColumnContext::new(), "-5");
        assert_eq!(result.constant(), Some(&PgValue::Integer(-5)));
        // the minimum int4 has no negative
        let min = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Integer(i64::from(i32::MIN)),
                PgValue::Integer(1),
            ])),
        );
        let result = read(ColumnContext::new().with_column("x", min), "-x");
        assert!(result.may_raise());
        assert_eq!(result.constant_value(), Some(&PgValue::Integer(-1)));
        // an unlisted int4 may hold the minimum, so negation may raise and
        // the interval is mirrored
        let result = read(ColumnContext::new().with_column("x", int4_unlisted()), "-x");
        assert!(result.may_raise());
        let Values::Unlisted(unlisted) = result.values() else { panic!("stays unlisted") };
        // the mirrored interval is clamped to the int4 range, which the
        // result type admits
        assert_eq!(unlisted.numbers.low, Bound::Included(PgNumeric::from(-i64::from(i32::MAX))));
        assert_eq!(unlisted.numbers.high, Bound::Included(PgNumeric::from(i64::from(i32::MAX))));
        // a listed numeric negates value by value
        let listed = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Numeric(
                PgNumeric::parse("1.5").expect("valid"),
            )])),
        );
        let result = read(ColumnContext::new().with_column("n", listed), "-n");
        assert!(!result.may_raise());
        assert_eq!(
            result.constant_value(),
            Some(&PgValue::Numeric(PgNumeric::parse("-1.5").expect("valid"))),
        );
        // a numeric range away from zero negates cleanly with mirrored
        // excluded bounds
        let guarantees = guarantees_for("n", ScalarType::Numeric, bound("e", 0, "e", 5));
        let result = read_guarantees(
            ColumnContext::new().with_column("n", numeric_unlisted()),
            &guarantees,
            "-n",
        );
        assert!(!result.may_raise());
        let Values::Unlisted(unlisted) = result.values() else { panic!("stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Excluded(PgNumeric::from(-5)));
        assert_eq!(unlisted.numbers.high, Bound::Excluded(PgNumeric::from(0_i64)));
        // a wide integer range away from the minimum negates cleanly
        let guarantees =
            guarantees_for("x", ScalarType::Integer(IntegerType::Integer), number_interval(1, 100));
        let result = read_guarantees(
            ColumnContext::new().with_column("x", int4_unlisted()),
            &guarantees,
            "-x",
        );
        assert!(!result.may_raise());
        let Values::Unlisted(unlisted) = result.values() else { panic!("stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Included(PgNumeric::from(-100)));
        assert_eq!(unlisted.numbers.high, Bound::Included(PgNumeric::from(-1)));
        // a non-number negates to an unknown type
        let result = read(ColumnContext::new().with_column("s", text_unlisted()), "-s");
        assert_eq!(result.ty(), ScalarType::Unknown);
        assert!(matches!(result.values(), Values::Unlisted(_)));
    }

    #[test]
    fn a_divisor_may_be_zero_only_when_its_interval_admits_it() {
        let result = read(ColumnContext::new().with_column("x", int4_unlisted()), "1 / x");
        assert!(result.may_raise());
        assert!(result.may_be_null());
        let result = read(ColumnContext::new().with_column("n", numeric_unlisted()), "1 / n");
        assert!(result.may_raise());
        // a divisor proven non-zero cannot raise
        let guarantees = guarantees_for("n", ScalarType::Numeric, bound("e", 0, "_", 0));
        let result = read_guarantees(
            ColumnContext::new().with_column("n", numeric_unlisted()),
            &guarantees,
            "1 / n",
        );
        assert!(!result.may_raise());
        assert!(!result.may_be_null());
    }

    #[test]
    fn interval_contains_zero_checks_its_bounds() {
        assert!(interval_contains_zero(&bound("_", 0, "_", 0)));
        assert!(interval_contains_zero(&number_interval(-1, 1)));
        assert!(!interval_contains_zero(&bound("e", 0, "_", 0)));
        assert!(!interval_contains_zero(&bound("_", 0, "e", 0)));
        assert!(!interval_contains_zero(&number_interval(1, 2)));
    }

    #[test]
    fn between_keeps_its_bounds_inclusive() {
        let context = || ColumnContext::new().with_column("x", int4_unlisted());
        let result = read(context(), "x BETWEEN 1 AND 3");
        assert!(result.may_be_true());
        assert!(result.may_be_false());
        assert!(result.may_be_null());
        let result = read(ColumnContext::new(), "2 BETWEEN 1 AND 3");
        assert!(result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_be_null());
        let result = read(ColumnContext::new(), "1 BETWEEN 1 AND 3");
        assert!(result.may_be_true());
        assert!(!result.may_be_false());
        let result = read(ColumnContext::new(), "4 BETWEEN 1 AND 3");
        assert!(result.may_be_false());
        assert!(!result.may_be_true());
        let result = read(ColumnContext::new(), "0 NOT BETWEEN 0 AND 0");
        assert!(result.may_be_false());
        assert!(!result.may_be_true());
        let result = read(ColumnContext::new(), "NULL BETWEEN 1 AND 3");
        assert!(result.may_be_null());
        assert!(!result.may_be_true());
        assert!(!result.may_be_false());
        let result = read(ColumnContext::new(), "1 / 0 BETWEEN 1 AND 3");
        assert!(result.always_raises());
    }

    #[test]
    fn an_always_null_side_of_a_refused_pair_may_raise() {
        let generic = |sql: &str| {
            evaluate(
                &parse(sql),
                &sqlparser::dialect::GenericDialect,
                &(),
                &mut ColumnContext::new().with_column(
                    "f",
                    Evaluation::new(
                        ScalarType::Float,
                        true,
                        false,
                        Values::Unlisted(Unlisted::any()),
                    ),
                ),
                &Guarantees::new(),
            )
            .expect("the context resolves")
        };
        assert!(generic("NULL::text = -1").may_raise());
        assert!(!generic("NULL = -1").may_raise());
        assert!(generic("1 = ''").may_raise());
        assert!(generic("'1' = f").may_raise());
    }

    #[test]
    fn concatenation_folds_its_known_sides() {
        let result = read(ColumnContext::new(), "'a' || 'b'");
        assert_eq!(result.constant(), Some(&PgValue::Text("ab".to_string())));
        let result = read(ColumnContext::new().with_column("s", text_unlisted()), "s || 'x'");
        assert!(matches!(result.values(), Values::Unlisted(_)));
        assert!(!result.may_raise());
        assert!(result.may_be_null());
        // two non-text operands have no type to resolve, so the value
        // stays open and the error stays possible
        let result = read(ColumnContext::new(), "1 || 2");
        assert!(result.may_raise());
        assert!(matches!(result.values(), Values::Unlisted(_)));
        let known = Evaluation::known(
            PgValue::Text("he".to_string()),
            ScalarType::Text(TextRules::DEFAULT),
        );
        let result = read(ColumnContext::new().with_column("s", known), "s || 'llo'");
        assert_eq!(result.constant(), Some(&PgValue::Text("hello".to_string())));
        let impossible = Evaluation::impossible(ScalarType::Text(TextRules::DEFAULT));
        let result = read(ColumnContext::new().with_column("s", impossible), "'a' || s");
        assert!(result.is_impossible());
    }

    #[test]
    fn arithmetic_resolves_untyped_literals_against_the_number() {
        let result = read(ColumnContext::new(), "'1' + 1");
        assert_eq!(result.constant(), Some(&PgValue::Integer(2)));
        let result = read(ColumnContext::new(), "1 + '1'");
        assert_eq!(result.constant(), Some(&PgValue::Integer(2)));
        // Two untyped literals leave the operator ambiguous.
        assert!(read(ColumnContext::new(), "'1' + '2'").may_raise());
    }

    /// `integer = text` resolves the text literal as an integer, which raises.
    #[test]
    fn a_number_compared_with_a_text_literal_raises() {
        let result = read(ColumnContext::new(), "5 = 'a'");
        assert!(result.may_raise());
        assert!(!result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_be_null());
    }

    #[test]
    fn arithmetic_type_widens_and_resolves() {
        let int2 = ScalarType::Integer(IntegerType::Smallint);
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let int8 = ScalarType::Integer(IntegerType::Bigint);
        assert_eq!(arithmetic_type(int2, int4), int4);
        assert_eq!(arithmetic_type(int4, int8), int8);
        assert_eq!(arithmetic_type(int4, ScalarType::Numeric), ScalarType::Numeric);
        assert_eq!(arithmetic_type(ScalarType::Float, int4), ScalarType::Float);
        assert_eq!(arithmetic_type(int4, ScalarType::UnknownLiteral), int4);
        assert_eq!(
            arithmetic_type(ScalarType::UnknownLiteral, ScalarType::Numeric),
            ScalarType::Numeric,
        );
        assert_eq!(
            arithmetic_type(ScalarType::Text(TextRules::DEFAULT), int4),
            ScalarType::Unknown,
        );
        let result = read(ColumnContext::new().with_column("s", text_unlisted()), "s + 1");
        // PostgreSQL has no `+` for a text and an integer, so the pair
        // raises.
        assert_eq!(result.ty(), ScalarType::Unknown);
        assert!(result.may_raise());
        let result = read(
            ColumnContext::new().with_column(
                "f",
                Evaluation::new(ScalarType::Float, true, false, Values::Unlisted(Unlisted::any())),
            ),
            "f + 1",
        );
        assert_eq!(result.ty(), ScalarType::Float);
        assert!(matches!(result.values(), Values::Unlisted(_)));
    }

    fn text(rules: TextRules, values: &[&str]) -> Evaluation {
        Evaluation::new(
            ScalarType::Text(rules),
            false,
            false,
            Values::Listed(values.iter().map(|value| PgValue::Text(value.to_string())).collect()),
        )
    }

    #[test]
    fn text_pairs_follow_the_collation_and_the_order() {
        // A listed left against an unlisted right admits no pair certainty.
        assert_eq!(
            pair_analysis(
                &text(TextRules::DEFAULT, &["a"]),
                &Evaluation::new(
                    ScalarType::Text(TextRules::DEFAULT),
                    false,
                    false,
                    Values::Unlisted(Unlisted::any()),
                ),
                &Category::Text(TextRules::DEFAULT, TextRules::DEFAULT),
                &BinaryOperator::Eq,
            ),
            RowKinds::ANY,
        );
        // An equal pair under the inequality is a certain falsity per row.
        assert_eq!(
            pair_analysis(
                &text(TextRules::DEFAULT, &["a"]),
                &text(TextRules::DEFAULT, &["a"]),
                &Category::Text(TextRules::DEFAULT, TextRules::DEFAULT),
                &BinaryOperator::NotEq,
            ),
            RowKinds { all_true: false, all_false: true, mixed: false },
        );
        // An empty side admits no pair at all.
        let empty =
            Evaluation::new(ScalarType::Text(TextRules::DEFAULT), false, true, Values::none());
        assert_eq!(
            pair_analysis(
                &empty,
                &text(TextRules::DEFAULT, &["a"]),
                &Category::Text(TextRules::DEFAULT, TextRules::DEFAULT),
                &BinaryOperator::Eq,
            ),
            RowKinds::NONE,
        );
    }

    fn instant(value: &str) -> Evaluation {
        Evaluation::known(
            PgValue::Timestamp(
                sql_values::postgres::parse_timestamp(value).expect("a valid timestamp"),
            ),
            ScalarType::Timestamp,
        )
    }

    #[test]
    fn timestamp_pairs_follow_their_operator() {
        let earlier = instant("2026-01-01 00:00:00");
        let later = instant("2026-01-01 01:00:00");
        assert_eq!(
            pair_analysis(
                &instant("2026-01-01 00:00:00"),
                &instant("2026-01-01 00:00:00"),
                &Category::Timestamp,
                &BinaryOperator::NotEq,
            ),
            RowKinds { all_true: false, all_false: true, mixed: false },
        );
        assert_eq!(
            pair_analysis(&earlier, &later, &Category::Timestamp, &BinaryOperator::LtEq),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        assert_eq!(
            pair_analysis(
                &instant("2026-01-01 00:00:00"),
                &instant("2026-01-01 00:00:00"),
                &Category::Timestamp,
                &BinaryOperator::GtEq,
            ),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        // An operator the timestamps do not support admits no certainty.
        assert_eq!(
            pair_analysis(&earlier, &later, &Category::Timestamp, &BinaryOperator::Spaceship),
            RowKinds::ANY,
        );
        // An empty side admits no pair at all.
        let empty = Evaluation::new(ScalarType::Timestamp, false, true, Values::none());
        assert_eq!(
            pair_analysis(&earlier, &empty, &Category::Timestamp, &BinaryOperator::Eq),
            RowKinds::NONE,
        );
    }

    #[test]
    fn an_unmodelled_number_operator_admits_no_certainty() {
        assert_eq!(
            pair_analysis(&point(1), &point(1), &Category::Number, &BinaryOperator::Spaceship),
            RowKinds::ANY,
        );
    }

    #[test]
    fn an_unlisted_range_against_a_listed_value_is_certainly_different() {
        // some member of the range is outside the listed value
        assert_eq!(
            pair_analysis(
                &numbers(number_interval(0, 3)),
                &point(2),
                &Category::Number,
                &BinaryOperator::NotEq
            ),
            RowKinds { all_true: true, all_false: true, mixed: false },
        );
        // a single-valued range disjoint from the value differs on every
        // pair, and admits the row certainty through its point
        assert_eq!(
            pair_analysis(
                &numbers(number_interval(2, 2)),
                &point(3),
                &Category::Number,
                &BinaryOperator::NotEq
            ),
            RowKinds { all_true: true, all_false: false, mixed: false },
        );
        // a single-valued range holding the value is certainly equal
        assert_eq!(
            pair_analysis(
                &numbers(number_interval(2, 2)),
                &point(2),
                &Category::Number,
                &BinaryOperator::NotEq
            ),
            RowKinds { all_true: false, all_false: true, mixed: true },
        );
    }

    #[test]
    fn an_impossible_side_nulls_the_arithmetic_and_the_negation() {
        let impossible = Evaluation::impossible(ScalarType::Integer(IntegerType::Integer));
        let context = || ColumnContext::new().with_column("x", impossible.clone());
        let result = read(context(), "1 + x");
        assert!(result.is_impossible());
        let result = read(context(), "-x");
        assert!(result.is_impossible());
    }

    #[test]
    fn negation_of_an_unbounded_numeric_stays_unbounded() {
        let result = read(ColumnContext::new().with_column("n", numeric_unlisted()), "-n");
        assert!(!result.may_raise());
        let Values::Unlisted(unlisted) = result.values() else { panic!("stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Unbounded);
        assert_eq!(unlisted.numbers.high, Bound::Unbounded);
        // an excluded integer bound stays excluded when mirrored
        let guarantees =
            guarantees_for("x", ScalarType::Integer(IntegerType::Integer), bound("e", 0, "i", 100));
        let result = read_guarantees(
            ColumnContext::new().with_column("x", int4_unlisted()),
            &guarantees,
            "-x",
        );
        assert!(!result.may_raise());
        let Values::Unlisted(unlisted) = result.values() else { panic!("stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Included(PgNumeric::from(-100)));
        assert_eq!(unlisted.numbers.high, Bound::Excluded(PgNumeric::from(0_i64)));
    }

    #[test]
    fn a_date_column_compared_with_a_timestamp_resolves() {
        let date = Evaluation::new(
            ScalarType::Date,
            true,
            false,
            Values::Listed(BTreeSet::from([PgValue::Date(
                sql_values::postgres::parse_date("2026-01-01").expect("valid"),
            )])),
        );
        let result = read(
            ColumnContext::new().with_column("d", date),
            "d = TIMESTAMP '2026-01-01 00:00:00'",
        );
        assert!(result.may_be_true());
        assert!(!result.may_be_false());
        let ts = Evaluation::new(
            ScalarType::Timestamp,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Timestamp(
                sql_values::postgres::parse_timestamp("2026-01-01 01:00:00").expect("valid"),
            )])),
        );
        let result = read(ColumnContext::new().with_column("t", ts), "t = DATE '2026-01-01'");
        assert!(!result.may_be_true());
        assert!(result.may_be_false());
    }

    /// An unknown literal whose reading a setting decides stays open.
    #[test]
    fn an_unknown_literal_a_setting_decides_stays_open() {
        let literal =
            Evaluation::known(PgValue::Text("13/01/2026".to_string()), ScalarType::UnknownLiteral);
        let resolved = super::reinterpret_unknown(literal, ScalarType::Date);
        assert!(matches!(resolved.values(), Values::Unlisted(_)));
        assert!(resolved.may_raise());
    }

    /// A minus over a number literal folds into the constant before it is
    /// typed, as PostgreSQL's `make_const` types `-2147483648` as `integer`.
    #[test]
    fn a_negated_number_literal_is_typed_with_its_sign() {
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let int8 = ScalarType::Integer(IntegerType::Bigint);
        let minimum = read(ColumnContext::new(), "-2147483648");
        assert_eq!(minimum.ty(), int4);
        assert_eq!(minimum.constant(), Some(&PgValue::Integer(-2_147_483_648)));
        assert_eq!(read(ColumnContext::new(), "-2147483648 + 1").ty(), int4);
        // A double minus lifts the constant back past `integer`.
        let lifted = read(ColumnContext::new(), "- -2147483648");
        assert_eq!(lifted.ty(), int8);
        assert!(!lifted.may_raise());
        assert_eq!(read(ColumnContext::new(), "-(1.5)").ty(), ScalarType::Numeric);
    }

    /// One untyped literal the target cannot read raises on its own rows,
    /// while the others keep the values they fold to.
    #[test]
    fn a_raising_literal_keeps_the_values_the_others_fold_to() {
        let literals = Evaluation::new(
            ScalarType::UnknownLiteral,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Text("5".to_string()),
                PgValue::Text("oops".to_string()),
            ])),
        );
        let resolved =
            super::reinterpret_unknown(literals, ScalarType::Integer(IntegerType::Integer));
        assert!(resolved.may_raise());
        assert_eq!(resolved.values(), &Values::Listed(BTreeSet::from([PgValue::Integer(5)])));
    }

    /// A negation past `i64` keeps the value the wider type holds.
    #[test]
    fn a_negation_past_i64_keeps_the_wider_value() {
        let evaluation =
            read(ColumnContext::new(), "-(CASE WHEN true THEN -9223372036854775808 ELSE 1.5 END)");
        assert!(
            evaluation.may_be(&PgValue::Numeric(
                PgNumeric::parse("9223372036854775808").expect("a number")
            ))
        );
    }

    /// An untyped literal may raise against a type its text does not read as,
    /// and an always-`NULL` side keeps the refusal of its type.
    #[test]
    fn an_untyped_or_null_side_keeps_the_refusal() {
        let float =
            Evaluation::new(ScalarType::Float, true, false, Values::Unlisted(Unlisted::any()));
        let read = |sql: &str| read(ColumnContext::new().with_column("f", float.clone()), sql);
        assert!(read("'' BETWEEN 1 AND 10").may_raise());
        assert!(read("'' = ANY (ARRAY[-1])").may_raise());
        assert!(!read("'1' = 1").may_raise());
        assert!(read("1 = ''").may_raise());
        assert!(read("'1' = f").may_raise());
        assert!(read("NULL::text = -1").may_raise());
        assert!(!read("NULL = -1").may_raise());
    }

    /// Operands whose operator an engine may not resolve read the raise.
    #[test]
    fn an_unresolved_operator_may_raise() {
        let read = |sql: &str| read(ColumnContext::new(), sql);
        assert!(read("-(NULL)").may_raise());
        let float =
            Evaluation::new(ScalarType::Float, true, false, Values::Unlisted(Unlisted::any()));
        let negated = super::super::evaluate(
            &parse("-f"),
            &sqlparser::dialect::GenericDialect,
            &(),
            &mut ColumnContext::new().with_column("f", float),
            &Guarantees::new(),
        )
        .expect("the context resolves");
        assert_eq!(negated.ty(), ScalarType::Float);
        assert!(read("1 || NULL::int").may_raise());
        assert!(!read("NULL || 'a'").may_raise());
    }

    /// An untyped literal reads as the `boolean` its input gives, and another
    /// type as any `boolean` an engine may coerce it to.
    #[test]
    fn a_condition_reads_its_operand_as_boolean() {
        let read = |sql: &str| read(ColumnContext::new(), sql);
        assert!(read("'1' AND '1'").may_be_true());
        assert!(read("'1' AND NULL").may_be_null());
        assert!(read("NOT 't'").may_be_false());
        assert!(!read("'t' OR false").may_raise());
        assert!(read("'a' AND true").may_raise());
        let integer = read("1 AND true");
        assert!(integer.may_raise() && integer.may_be_true());
    }

    /// A term that decides an `AND` or an `OR` may be read before a raising
    /// term, which the engine then never reads.
    #[test]
    fn a_deciding_term_may_skip_a_raising_term() {
        let read = |sql: &str| read(ColumnContext::new(), sql);
        let and = read("false AND (1 / 0 = 1)");
        assert!(and.may_be_false() && !and.may_be_true());
        let or = read("(1 / 0 = 1) OR true");
        assert!(or.may_be_true() && !or.may_be_false());
        assert!(read("true AND (1 / 0 = 1)").always_raises());
    }

    /// An operand an engine refuses under a boolean, a sign or a comparison
    /// reads the raise.
    #[test]
    fn a_refused_operand_type_reads_the_raise() {
        let column =
            |ty: ScalarType| Evaluation::new(ty, true, false, Values::Unlisted(Unlisted::any()));
        let context = || {
            ColumnContext::new()
                .with_column("t", column(ScalarType::Text(crate::scalar::TextRules::DEFAULT)))
                .with_column("d", column(ScalarType::Date))
        };
        for sql in
            ["1 AND true", "'a' AND true", "NOT 1", "+t", "-t", "-'abc'", "d IN (1)", "1 = true"]
        {
            assert!(read(context(), sql).may_raise(), "{sql}");
        }
    }
}
