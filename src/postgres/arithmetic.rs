//! PostgreSQL's reading of arithmetic on two evaluated operands.

use alloc::collections::BTreeSet;
use core::ops::Bound;

use sql_values::postgres::{IntegerType, PgNumeric, PgValue};
use sqlparser::ast::BinaryOperator;

use crate::{
    domain::{Evaluation, Interval, Unlisted, Values},
    forward::fold::{arithmetic_type, interval_contains_zero, neg_bound},
    scalar::ScalarType,
    support::{intervals_disjoint, number_bounds},
};

/// The result of `left op right` under PostgreSQL's folding and interval
/// arithmetic.
pub(super) fn arithmetic(left: &Evaluation, op: &BinaryOperator, right: &Evaluation) -> Evaluation {
    let error = left.may_raise() || right.may_raise();
    let null = left.may_be_null() || right.may_be_null();
    if let Some(result) = date_days(left, right, op, null, error) {
        return result;
    }
    // An untyped `NULL` reads as the type of a number on the other side.
    let number = |side: &Evaluation| side.ty().is_number() || side.ty() == ScalarType::Float;
    if left.is_untyped_null() && number(right) {
        return arithmetic(&Evaluation::null(right.ty()).with_error(left.may_raise()), op, right);
    }
    if right.is_untyped_null() && number(left) {
        return arithmetic(left, op, &Evaluation::null(left.ty()).with_error(right.may_raise()));
    }
    let ty = arithmetic_type(left.ty(), right.ty());
    if !arithmetic_has_operator(left.ty(), op, right.ty()) {
        // PostgreSQL picks the operator from the operand types before it
        // reads a value, so a pair of modelled types without one raises
        // whatever its values.
        if modelled(left.ty()) && modelled(right.ty()) {
            return Evaluation::raising(ScalarType::Unknown);
        }
        // For a type the reading does not model an operator may exist or not.
        return Evaluation::new(ScalarType::Unknown, null, true, Values::Unlisted(Unlisted::any()));
    }
    // A `date` and a `timestamp` subtract into an `interval`, which the
    // reading leaves open and which raises past the `interval` range, or for
    // a date past the `timestamp` range.
    let temporal = |ty: ScalarType| matches!(ty, ScalarType::Date | ScalarType::Timestamp);
    if temporal(left.ty()) && temporal(right.ty()) {
        return Evaluation::new(ScalarType::Other, null, true, Values::Unlisted(Unlisted::any()));
    }
    // The operator is strict, so a `NULL` operand answers `NULL` without a
    // raise, save the widening of a `numeric` side to `double precision`
    // before it, which may overflow.
    if left.is_always_null() || right.is_always_null() {
        let widened = ty == ScalarType::Float
            && [left, right]
                .iter()
                .any(|side| side.ty() == ScalarType::Numeric && !side.is_always_null());
        return Evaluation::null(ty).with_error(error || widened);
    }
    // Never folded, and a float may overflow or divide by zero.
    if matches!(ty, ScalarType::Float | ScalarType::Unknown) {
        let error = error || ty == ScalarType::Float;
        return Evaluation::new(ty, null, error, Values::Unlisted(Unlisted::any()));
    }
    if let (Some(left), Some(right)) = (left.constant(), right.constant()) {
        return match fold_arithmetic(left, op, right, ty.integer_bounds()) {
            Ok(datum) => Evaluation::known(datum, ty),
            Err(true) => Evaluation::raising(ty),
            Err(false) => Evaluation::new(ty, null, error, Values::Unlisted(Unlisted::any())),
        };
    }
    let left_interval = number_bounds(left.values());
    let right_interval = number_bounds(right.values());
    let (values, arith_error) = interval_values(op, ty, &left_interval, &right_interval, error);
    Evaluation::new(ty, null, arith_error, values)
}

/// Whether the reading knows every arithmetic operator PostgreSQL has for
/// the type, an array having none.
fn modelled(ty: ScalarType) -> bool {
    matches!(
        ty,
        ScalarType::Boolean
            | ScalarType::Integer(_)
            | ScalarType::Numeric
            | ScalarType::Float
            | ScalarType::Text(_)
            | ScalarType::Date
            | ScalarType::Timestamp
            | ScalarType::Array(_)
    )
}

/// Whether PostgreSQL has the arithmetic operator `op` for the pair of
/// types. Every number pair has the four operations and `%` save for
/// `double precision`, and a `date` and a `timestamp` subtract into an
/// `interval`. The `date` and `integer` pairs a `date_days` reading covers
/// are handled before this question is asked.
fn arithmetic_has_operator(left: ScalarType, op: &BinaryOperator, right: ScalarType) -> bool {
    let numeric = |ty: ScalarType| ty.is_number() || ty == ScalarType::Float;
    let temporal = |ty: ScalarType| matches!(ty, ScalarType::Date | ScalarType::Timestamp);
    if numeric(left) && numeric(right) {
        return *op != BinaryOperator::Modulo
            || (left != ScalarType::Float && right != ScalarType::Float);
    }
    temporal(left) && temporal(right) && *op == BinaryOperator::Minus
}

/// The day counts of every `left - right` pair of listed dates, with whether
/// a pair raises, as one with an infinite date does.
fn date_day_differences(
    lefts: &BTreeSet<PgValue>,
    rights: &BTreeSet<PgValue>,
) -> (BTreeSet<PgValue>, bool) {
    let mut raises = false;
    let mut differences = BTreeSet::new();
    for left in lefts.iter().filter_map(PgValue::as_date) {
        for right in rights.iter().filter_map(PgValue::as_date) {
            match left.days_since(right) {
                Ok(days) => {
                    differences.insert(PgValue::Integer(i64::from(days)));
                }
                Err(_) => raises = true,
            }
        }
    }
    (differences, raises)
}

/// `date ± integer`, `integer + date` and `date - date`, which PostgreSQL
/// reads with the integer as a count of days. PostgreSQL has these operators
/// for `integer` alone (a `smallint` widens to it), so `bigint` and
/// `timestamp` with an integer raise.
fn date_days(
    left: &Evaluation,
    right: &Evaluation,
    op: &BinaryOperator,
    null: bool,
    error: bool,
) -> Option<Evaluation> {
    let integer_type = ScalarType::Integer(IntegerType::Integer);
    match (left.ty(), right.ty()) {
        (ScalarType::Date, ScalarType::Date) if *op == BinaryOperator::Minus => {
            return Some(match (left.values(), right.values()) {
                (Values::Listed(lefts), Values::Listed(rights)) => {
                    let (differences, raises) = date_day_differences(lefts, rights);
                    Evaluation::new(
                        integer_type,
                        null,
                        error || raises,
                        Values::Listed(differences),
                    )
                }
                // Either date may be infinite, which raises.
                _ => Evaluation::new(integer_type, null, true, Values::Unlisted(Unlisted::any())),
            });
        }
        (ScalarType::Timestamp, ScalarType::Integer(_))
        | (ScalarType::Integer(_), ScalarType::Timestamp) => {
            return Some(Evaluation::raising(ScalarType::Timestamp));
        }
        (ScalarType::Date, ScalarType::Integer(IntegerType::Bigint))
        | (ScalarType::Integer(IntegerType::Bigint), ScalarType::Date) => {
            return Some(Evaluation::raising(ScalarType::Date));
        }
        _ => {}
    }
    let (temporal, integer, sign) = match (left.ty(), right.ty(), op) {
        (ScalarType::Date, ScalarType::Integer(_), BinaryOperator::Plus) => (left, right, 1),
        (ScalarType::Date, ScalarType::Integer(_), BinaryOperator::Minus) => (left, right, -1),
        (ScalarType::Integer(_), ScalarType::Date, BinaryOperator::Plus) => (right, left, 1),
        (ScalarType::Date, ScalarType::Integer(_), _)
        | (ScalarType::Integer(_), ScalarType::Date, _) => {
            return Some(Evaluation::raising(ScalarType::Date));
        }
        _ => return None,
    };
    let (Values::Listed(dates), Values::Listed(days)) = (temporal.values(), integer.values())
    else {
        // A shifted date may leave the date range, which raises.
        return Some(Evaluation::new(
            ScalarType::Date,
            null,
            true,
            Values::Unlisted(Unlisted::any()),
        ));
    };
    let mut values = BTreeSet::new();
    let mut raises = error;
    // A listed integer may hold the same day count as a numeric value, which
    // still shifts the date but cannot be folded here. An `integer` or
    // `smallint` operand lists `int4` values.
    let mut misfitted = false;
    let day_count = |datum: &PgValue| {
        match datum {
            PgValue::Integer(days) => i32::try_from(*days).ok(),
            _ => None,
        }
    };
    for date in dates {
        for shift in days {
            let (PgValue::Date(date), Some(shift)) = (date, day_count(shift)) else {
                misfitted = true;
                continue;
            };
            let shifted =
                if sign < 0 { date.checked_sub_days(shift) } else { date.checked_add_days(shift) };
            match shifted {
                Ok(shifted) => {
                    values.insert(PgValue::Date(shifted));
                }
                Err(_) => raises = true,
            }
        }
    }
    let values = if misfitted { Values::Unlisted(Unlisted::any()) } else { Values::Listed(values) };
    Some(Evaluation::new(ScalarType::Date, null, raises, values))
}

/// The fold of a constant arithmetic operation, `Err(true)` when it raises.
fn fold_arithmetic(
    left: &PgValue,
    op: &BinaryOperator,
    right: &PgValue,
    bounds: Option<(i64, i64)>,
) -> Result<PgValue, bool> {
    if let (PgValue::Integer(a), PgValue::Integer(b)) = (left, right) {
        let (a, b) = (i128::from(*a), i128::from(*b));
        let result = match op {
            BinaryOperator::Plus => a.checked_add(b),
            BinaryOperator::Minus => a.checked_sub(b),
            BinaryOperator::Multiply => a.checked_mul(b),
            BinaryOperator::Divide => {
                if b == 0 {
                    return Err(true);
                }
                Some(a / b)
            }
            BinaryOperator::Modulo => {
                if b == 0 {
                    return Err(true);
                }
                Some(a % b)
            }
            _ => None,
        };
        let Some(result) = result else {
            return Err(true);
        };
        let (lo, hi) = bounds.unwrap_or((i64::MIN, i64::MAX));
        if result < i128::from(lo) || result > i128::from(hi) {
            return Err(true);
        }
        Ok(PgValue::Integer(i64::try_from(result).map_err(|_| true)?))
    } else {
        let left = left.as_numeric().ok_or(true)?;
        let right = right.as_numeric().ok_or(true)?;
        let result = match op {
            BinaryOperator::Plus => left.checked_add(&right),
            BinaryOperator::Minus => left.checked_sub(&right),
            BinaryOperator::Multiply => left.checked_mul(&right),
            BinaryOperator::Divide => left.checked_div(&right),
            BinaryOperator::Modulo => left.checked_rem(&right),
            _ => return Err(false),
        }
        .map_err(|_| true)?;
        Ok(PgValue::Numeric(result))
    }
}

/// The values the interval arithmetic of `op` keeps, with the error flag
/// it adds.
fn interval_values(
    op: &BinaryOperator,
    ty: ScalarType,
    left: &Interval<PgNumeric>,
    right: &Interval<PgNumeric>,
    error: bool,
) -> (Values, bool) {
    let (left, right) = (&finite_bounds(left), &finite_bounds(right));
    let mut values = Values::Unlisted(Unlisted::any());
    let mut error = error;
    let mut error0 = false;
    let interval = match op {
        BinaryOperator::Plus => Some(interval_add(left, right, &mut error0)),
        BinaryOperator::Minus => Some(interval_sub(left, right, &mut error0)),
        BinaryOperator::Multiply => Some(interval_mul(left, right, &mut error0)),
        BinaryOperator::Divide => {
            if interval_contains_zero(right) {
                error = true;
                None
            } else {
                Some(interval_div(left, right, &mut error0))
            }
        }
        BinaryOperator::Modulo => {
            if interval_contains_zero(right) {
                error = true;
            }
            None
        }
        _ => None,
    };
    if let Some(mut interval) = interval {
        if matches!(ty, ScalarType::Integer(_)) {
            interval = integer_result_interval(&interval);
        } else if matches!(left.high, Bound::Unbounded) || matches!(right.high, Bound::Unbounded) {
            // A `numeric` operand unbounded above may be `NaN` or
            // `Infinity`, which make the result `NaN`, above every number.
            interval.high = Bound::Unbounded;
        }
        if let ScalarType::Integer(target) = ty {
            let (lo, hi) = target.range().into_inner();
            let (min, max) = (PgNumeric::from(lo), PgNumeric::from(hi));
            // The result overflows when it admits a value the type cannot
            // hold, past either end of the range.
            let below = Interval { low: Bound::Unbounded, high: Bound::Excluded(min) };
            let above = Interval { low: Bound::Excluded(max), high: Bound::Unbounded };
            if !intervals_disjoint(&interval, &below) || !intervals_disjoint(&interval, &above) {
                error = true;
            }
        }
        let (low, high) = (interval.low, interval.high);
        values = Values::Unlisted(Unlisted {
            excluded: BTreeSet::new(),
            numbers: Interval { low, high },
            length: Interval::unbounded(),
        });
    } else if matches!(op, BinaryOperator::Divide | BinaryOperator::Modulo) {
        // The divisor may be zero; the values are undecided.
    } else {
        error = true;
    }
    (values, error)
}

/// The interval with a `NaN` or infinite bound widened to unbounded, which
/// keeps the bound arithmetic to finite values.
fn finite_bounds(interval: &Interval<PgNumeric>) -> Interval<PgNumeric> {
    let finite = |bound: &Bound<PgNumeric>| {
        match bound {
            Bound::Included(value) | Bound::Excluded(value)
                if value.is_nan() || value.is_infinite() =>
            {
                Bound::Unbounded
            }
            bound => bound.clone(),
        }
    };
    Interval { low: finite(&interval.low), high: finite(&interval.high) }
}

/// Whether either end of the interval is unbounded.
fn reaches_unbounded(interval: &Interval<PgNumeric>) -> bool {
    matches!(interval.low, Bound::Unbounded) || matches!(interval.high, Bound::Unbounded)
}

/// The interval of `left + right`.
fn interval_add(
    left: &Interval<PgNumeric>,
    right: &Interval<PgNumeric>,
    error: &mut bool,
) -> Interval<PgNumeric> {
    Interval {
        low: add_bound(&left.low, &right.low, error),
        high: add_bound(&left.high, &right.high, error),
    }
}

/// The interval of `left - right`.
fn interval_sub(
    left: &Interval<PgNumeric>,
    right: &Interval<PgNumeric>,
    error: &mut bool,
) -> Interval<PgNumeric> {
    Interval {
        low: add_bound(&left.low, &neg_bound(&right.high), error),
        high: add_bound(&left.high, &neg_bound(&right.low), error),
    }
}

/// The bound `a + b`, unbounded when either is, open when either is open.
fn add_bound(a: &Bound<PgNumeric>, b: &Bound<PgNumeric>, error: &mut bool) -> Bound<PgNumeric> {
    match (a, b) {
        (Bound::Unbounded, _) | (_, Bound::Unbounded) => Bound::Unbounded,
        (
            va @ (Bound::Included(_) | Bound::Excluded(_)),
            vb @ (Bound::Included(_) | Bound::Excluded(_)),
        ) => {
            let open = matches!(a, Bound::Excluded(_)) || matches!(b, Bound::Excluded(_));
            let (va, vb) = (bound_number(va), bound_number(vb));
            if let Ok(sum) = va.checked_add(&vb) {
                if open { Bound::Excluded(sum) } else { Bound::Included(sum) }
            } else {
                *error = true;
                Bound::Unbounded
            }
        }
    }
}

/// The value a finite bound stands at, `zero` for the unbounded bound.
fn bound_number(bound: &Bound<PgNumeric>) -> PgNumeric {
    match bound {
        Bound::Included(value) | Bound::Excluded(value) => value.clone(),
        Bound::Unbounded => PgNumeric::from(0_i64),
    }
}

/// The interval of `left * right`, widened to the unbounded when a factor
/// is unbounded at either end, since it then reaches every size, an
/// infinity times zero included.
fn interval_mul(
    left: &Interval<PgNumeric>,
    right: &Interval<PgNumeric>,
    error: &mut bool,
) -> Interval<PgNumeric> {
    corner_interval(left, right, error, |a, b| a.checked_mul(b).ok())
}

/// The interval of a binary operation whose extremes over two bounded
/// intervals lie at their corners, widened to the unbounded when a side is
/// unbounded at either end. A corner whose result overflows widens the end
/// its sign falls on, which is the sign of a product or a quotient.
fn corner_interval(
    left: &Interval<PgNumeric>,
    right: &Interval<PgNumeric>,
    error: &mut bool,
    operation: impl Fn(&PgNumeric, &PgNumeric) -> Option<PgNumeric>,
) -> Interval<PgNumeric> {
    if reaches_unbounded(left) || reaches_unbounded(right) {
        return Interval::unbounded();
    }
    let corners = [
        (left.low.clone(), right.low.clone()),
        (left.low.clone(), right.high.clone()),
        (left.high.clone(), right.low.clone()),
        (left.high.clone(), right.high.clone()),
    ];
    let mut low: Option<(PgNumeric, bool)> = None;
    let mut high: Option<(PgNumeric, bool)> = None;
    let (mut low_overflow, mut high_overflow) = (false, false);
    for (a, b) in corners {
        let (first, first_open) = unbind(a);
        let (second, second_open) = unbind(b);
        if let Some(result) = operation(&first, &second) {
            let open = first_open || second_open;
            if low.as_ref().is_none_or(|(min, _)| result < *min) {
                low = Some((result.clone(), open));
            }
            if high.as_ref().is_none_or(|(max, _)| result > *max) {
                high = Some((result, open));
            }
        } else {
            // The corner's result overflows, and its sign is still known, so
            // widen the end that holds it.
            *error = true;
            if (first < PgNumeric::from(0_i64)) == (second < PgNumeric::from(0_i64)) {
                high_overflow = true;
            } else {
                low_overflow = true;
            }
        }
    }
    match (low, high) {
        (Some((low, open)), Some((high, high_open))) => {
            Interval {
                low: if low_overflow {
                    Bound::Unbounded
                } else if open {
                    Bound::Excluded(low)
                } else {
                    Bound::Included(low)
                },
                high: if high_overflow {
                    Bound::Unbounded
                } else if high_open {
                    Bound::Excluded(high)
                } else {
                    Bound::Included(high)
                },
            }
        }
        _ => Interval::unbounded(),
    }
}

/// The bound apart from its openness.
fn unbind(bound: Bound<PgNumeric>) -> (PgNumeric, bool) {
    match bound {
        Bound::Included(value) => (value, false),
        Bound::Excluded(value) => (value, true),
        Bound::Unbounded => (PgNumeric::from(0_i64), false),
    }
}

/// The interval of `left / right`, `right` known not to contain zero,
/// widened to the unbounded when a side is unbounded at either end.
fn interval_div(
    left: &Interval<PgNumeric>,
    right: &Interval<PgNumeric>,
    error: &mut bool,
) -> Interval<PgNumeric> {
    corner_interval(left, right, error, |a, b| a.checked_div(b).ok())
}

/// The interval widened to hold the integer results of the division over
/// it, for an integer result type, widening an unrepresentable bound to the
/// unbounded to keep the interval sound.
fn integer_result_interval(interval: &Interval<PgNumeric>) -> Interval<PgNumeric> {
    let (low, high) = (interval.low.clone(), interval.high.clone());
    let lo = match low {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(value) => floor_number(&value).map_or(Bound::Unbounded, Bound::Included),
        // the first integer strictly above `value`
        Bound::Excluded(value) => {
            floor_number(&value)
                .and_then(|floored| floored.checked_add(&PgNumeric::from(1)).ok())
                .map_or(Bound::Unbounded, Bound::Included)
        }
    };
    let hi = match high {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(value) => ceil_number(&value).map_or(Bound::Unbounded, Bound::Included),
        // the last integer strictly below `value`
        Bound::Excluded(value) => {
            ceil_number(&value)
                .and_then(|ceiled| ceiled.checked_sub(&PgNumeric::from(1)).ok())
                .map_or(Bound::Unbounded, Bound::Included)
        }
    };
    Interval { low: lo, high: hi }
}

/// The largest integer at or below `value`, or `None` when that integer is
/// past the representable range.
fn floor_number(value: &PgNumeric) -> Option<PgNumeric> {
    value.clone().floor().ok()
}

/// The smallest integer at or above `value`, or `None` when that integer is
/// past the representable range.
fn ceil_number(value: &PgNumeric) -> Option<PgNumeric> {
    value.clone().ceil().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar::TextRules;

    fn number(text: &str) -> PgNumeric {
        PgNumeric::parse(text).expect("a number")
    }

    fn known_int(value: i64, width: IntegerType) -> Evaluation {
        Evaluation::known(PgValue::Integer(value), ScalarType::Integer(width))
    }

    fn known_number(text: &str) -> Evaluation {
        Evaluation::known(PgValue::Numeric(number(text)), ScalarType::Numeric)
    }

    fn known_date(text: &str) -> Evaluation {
        let date = sql_values::postgres::parse_date(text).expect("a date");
        Evaluation::known(PgValue::Date(date), ScalarType::Date)
    }

    /// The evaluation whose numbers lie between two bounds.
    fn unlisted(ty: ScalarType, low: &str, high: &str) -> Evaluation {
        Evaluation::new(
            ty,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Included(number(low)),
                    high: Bound::Included(number(high)),
                },
                length: Interval::unbounded(),
            }),
        )
    }

    /// The numbers an unlisted value set lies in.
    fn values_numbers(values: &Values) -> Interval<PgNumeric> {
        match values {
            Values::Unlisted(unlisted) => unlisted.numbers.clone(),
            Values::Listed(listed) => {
                let mut values =
                    listed.iter().map(|value| value.as_numeric().expect("a numeric value").clone());
                let low = values.next().expect("a value");
                let high = values.fold(low.clone(), Ord::max);
                Interval { low: Bound::Included(low), high: Bound::Included(high) }
            }
        }
    }

    /// The numbers a result's unlisted values lie in.
    fn result_numbers(result: &Evaluation) -> Interval<PgNumeric> {
        values_numbers(result.values())
    }

    /// The one value a constant pair folds to.
    fn folded(left: &Evaluation, op: &BinaryOperator, right: &Evaluation) -> PgValue {
        arithmetic(left, op, right).constant().expect("a fold to a single value").clone()
    }

    /// A raising result of the right type.
    fn raised(result: &Evaluation, ty: ScalarType) {
        assert_eq!(result.ty(), ty);
        assert!(result.may_raise());
    }

    /// Float pairs are never folded, and may overflow or divide by zero.
    #[test]
    fn float_pairs_are_never_folded() {
        let float =
            Evaluation::new(ScalarType::Float, false, false, Values::Unlisted(Unlisted::any()));
        let result = arithmetic(&float, &BinaryOperator::Plus, &float);
        assert_eq!(result.ty(), ScalarType::Float);
        assert!(result.may_raise());
        let one = known_int(1, IntegerType::Integer);
        assert!(arithmetic(&one, &BinaryOperator::Divide, &float).may_raise());
    }

    /// A `numeric` widens to `double precision` before the operator reads a
    /// `NULL` side, and may overflow it.
    #[test]
    fn a_numeric_widening_may_raise_beside_a_null_float() {
        let numeric =
            Evaluation::new(ScalarType::Numeric, false, false, Values::Unlisted(Unlisted::any()));
        let null = Evaluation::null(ScalarType::Float);
        let result = arithmetic(&numeric, &BinaryOperator::Multiply, &null);
        assert!(result.may_be_null());
        assert!(result.may_raise());
        let null_numeric = Evaluation::null(ScalarType::Numeric);
        assert!(!arithmetic(&null_numeric, &BinaryOperator::Multiply, &null).may_raise());
    }

    /// A pair PostgreSQL has no operator for raises, on every operator.
    #[test]
    fn operator_less_pairs_raise() {
        let date = known_date("2020-01-01");
        for op in [
            BinaryOperator::Plus,
            BinaryOperator::Multiply,
            BinaryOperator::Divide,
            BinaryOperator::Modulo,
        ] {
            let result = arithmetic(&date, &op, &date);
            raised(&result, ScalarType::Unknown);
        }
        // `date + float` has no operator either
        let float =
            Evaluation::new(ScalarType::Float, false, false, Values::Unlisted(Unlisted::any()));
        raised(&arithmetic(&date, &BinaryOperator::Plus, &float), ScalarType::Unknown);
        let text = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        // Neither `int4 + text` nor `text + text` has an operator
        raised(
            &arithmetic(&known_int(1, IntegerType::Integer), &BinaryOperator::Plus, &text),
            ScalarType::Unknown,
        );
        raised(&arithmetic(&text, &BinaryOperator::Plus, &text), ScalarType::Unknown);
        // Two `timestamp`s have no operator between them
        let timestamp =
            Evaluation::new(ScalarType::Timestamp, false, false, Values::Unlisted(Unlisted::any()));
        raised(&arithmetic(&timestamp, &BinaryOperator::Plus, &timestamp), ScalarType::Unknown);
        let boolean =
            Evaluation::new(ScalarType::Boolean, false, false, Values::Unlisted(Unlisted::any()));
        raised(
            &arithmetic(&boolean, &BinaryOperator::Plus, &known_int(1, IntegerType::Integer)),
            ScalarType::Unknown,
        );
        // `double precision` has no `%`
        raised(
            &arithmetic(&float, &BinaryOperator::Modulo, &known_int(2, IntegerType::Integer)),
            ScalarType::Unknown,
        );
    }

    /// PostgreSQL picks the operator from the operand types before it reads
    /// a value, so a typed `NULL` without one raises, as
    /// `NULL::date + '2020-01-01'::date` does.
    #[test]
    fn operator_less_pairs_raise_whatever_their_values() {
        let date = Evaluation::null(ScalarType::Date);
        raised(
            &arithmetic(&date, &BinaryOperator::Plus, &known_date("2020-01-01")),
            ScalarType::Unknown,
        );
    }

    /// An untyped `NULL` resolves against a number, as `NULL + 1` is an
    /// `integer`, and against any other type may find no operator, as
    /// `NULL + true` raises while `current_date - NULL` is an `integer`.
    #[test]
    fn an_untyped_null_resolves_against_the_other_side() {
        let null = Evaluation::null(ScalarType::Unknown);
        let plus_one =
            arithmetic(&null, &BinaryOperator::Plus, &known_int(1, IntegerType::Integer));
        assert!(!plus_one.may_raise());
        assert!(plus_one.may_be_null());
        assert_eq!(plus_one.ty(), ScalarType::Integer(IntegerType::Integer));
        let float =
            Evaluation::new(ScalarType::Float, true, false, Values::Unlisted(Unlisted::any()));
        assert!(arithmetic(&float, &BinaryOperator::Modulo, &null).always_raises());
        let boolean = Evaluation::known(PgValue::Boolean(true), ScalarType::Boolean);
        let plus_true = arithmetic(&null, &BinaryOperator::Plus, &boolean);
        assert!(plus_true.may_raise());
        let minus_null = arithmetic(&known_date("2020-01-01"), &BinaryOperator::Minus, &null);
        assert!(minus_null.may_raise());
        assert!(minus_null.may_be_null());
    }

    /// Two `timestamp`s, or a `date` and a `timestamp`, subtract into an
    /// `interval`, which the reading leaves open, and which raises past the
    /// `interval` range or for a date past the `timestamp` range.
    #[test]
    fn timestamps_subtract_into_an_interval() {
        let timestamp =
            Evaluation::new(ScalarType::Timestamp, false, false, Values::Unlisted(Unlisted::any()));
        for (left, right) in [
            (&timestamp, &timestamp),
            (&known_date("2020-01-01"), &timestamp),
            (&timestamp, &known_date("2020-01-01")),
        ] {
            let difference = arithmetic(left, &BinaryOperator::Minus, right);
            assert_eq!(difference.ty(), ScalarType::Other);
            assert!(difference.may_raise());
            assert!(matches!(difference.values(), Values::Unlisted(_)));
        }
    }

    /// A constant integer difference folds at the pair's width.
    #[test]
    fn a_constant_integer_difference_folds() {
        assert_eq!(
            folded(
                &known_int(1, IntegerType::Integer),
                &BinaryOperator::Minus,
                &known_int(2, IntegerType::Integer),
            ),
            PgValue::Integer(-1)
        );
    }

    /// An unbounded integer operand may overflow an integer result.
    #[test]
    fn an_unbounded_integer_operand_may_overflow() {
        let x = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        let result = arithmetic(&x, &BinaryOperator::Plus, &known_int(1, IntegerType::Integer));
        assert_eq!(result.ty(), ScalarType::Integer(IntegerType::Integer));
        assert!(result.may_raise());
    }

    /// Dividing an open interval keeps the open ends.
    #[test]
    fn dividing_an_open_interval_keeps_the_open_ends() {
        let x = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Excluded(PgNumeric::from(0_i64)),
                    high: Bound::Excluded(number("10")),
                },
                length: Interval::unbounded(),
            }),
        );
        let result = arithmetic(&x, &BinaryOperator::Divide, &known_number("2"));
        let numbers = result_numbers(&result);
        assert_eq!(numbers.low, Bound::Excluded(PgNumeric::from(0_i64)));
        assert_eq!(numbers.high, Bound::Excluded(number("5")));
    }

    /// Constant integer pairs fold at the width the type of the pair gives
    /// them, and raise when the result leaves that width.
    #[test]
    fn constant_integer_folds_follow_the_width() {
        assert_eq!(
            folded(
                &known_int(32767, IntegerType::Smallint),
                &BinaryOperator::Plus,
                &known_int(1, IntegerType::Integer)
            ),
            PgValue::Integer(32768)
        );
        let result = arithmetic(
            &known_int(32767, IntegerType::Smallint),
            &BinaryOperator::Plus,
            &known_int(1, IntegerType::Smallint),
        );
        raised(&result, ScalarType::Integer(IntegerType::Smallint));
        let result = arithmetic(
            &known_int(i64::from(i16::MIN), IntegerType::Smallint),
            &BinaryOperator::Plus,
            &known_int(-1, IntegerType::Smallint),
        );
        raised(&result, ScalarType::Integer(IntegerType::Smallint));
        let result = arithmetic(
            &known_int(i64::from(i16::MIN), IntegerType::Smallint),
            &BinaryOperator::Modulo,
            &known_int(-1, IntegerType::Smallint),
        );
        assert_eq!(result.constant(), Some(&PgValue::Integer(0)));

        let result = arithmetic(
            &known_int(2_147_483_647, IntegerType::Integer),
            &BinaryOperator::Plus,
            &known_int(1, IntegerType::Integer),
        );
        raised(&result, ScalarType::Integer(IntegerType::Integer));
        let result = arithmetic(
            &known_int(2_147_483_647, IntegerType::Integer),
            &BinaryOperator::Multiply,
            &known_int(2, IntegerType::Integer),
        );
        raised(&result, ScalarType::Integer(IntegerType::Integer));
        // the `int4` operand widens the pair to `int8`, whose range the
        // sum fits
        assert_eq!(
            folded(
                &known_int(i64::MIN, IntegerType::Bigint),
                &BinaryOperator::Plus,
                &known_int(2_147_483_647, IntegerType::Integer)
            ),
            PgValue::Integer(-9_223_372_034_707_292_161)
        );

        // the minimum of a width negates past the width, as `MIN / -1` does
        let result = arithmetic(
            &known_int(i64::MIN, IntegerType::Bigint),
            &BinaryOperator::Divide,
            &known_int(-1, IntegerType::Bigint),
        );
        raised(&result, ScalarType::Integer(IntegerType::Bigint));
        let result = arithmetic(
            &known_int(i64::MIN, IntegerType::Bigint),
            &BinaryOperator::Modulo,
            &known_int(-1, IntegerType::Bigint),
        );
        assert_eq!(result.constant(), Some(&PgValue::Integer(0)));

        // division truncates toward zero, the remainder takes the sign of
        // the dividend, and a zero divisor raises
        assert_eq!(
            folded(
                &known_int(-7, IntegerType::Integer),
                &BinaryOperator::Divide,
                &known_int(2, IntegerType::Integer)
            ),
            PgValue::Integer(-3)
        );
        assert_eq!(
            folded(
                &known_int(-7, IntegerType::Integer),
                &BinaryOperator::Modulo,
                &known_int(2, IntegerType::Integer)
            ),
            PgValue::Integer(-1)
        );
        assert_eq!(
            folded(
                &known_int(7, IntegerType::Integer),
                &BinaryOperator::Modulo,
                &known_int(-2, IntegerType::Integer)
            ),
            PgValue::Integer(1)
        );
        let result = arithmetic(
            &known_int(7, IntegerType::Integer),
            &BinaryOperator::Divide,
            &known_int(0, IntegerType::Integer),
        );
        raised(&result, ScalarType::Integer(IntegerType::Integer));
        let result = arithmetic(
            &known_int(7, IntegerType::Integer),
            &BinaryOperator::Modulo,
            &known_int(0, IntegerType::Integer),
        );
        raised(&result, ScalarType::Integer(IntegerType::Integer));

        // a non-arithmetic operator on integers raises, as PostgreSQL's
        // `operator does not exist` does
        let result = arithmetic(
            &known_int(1, IntegerType::Integer),
            &BinaryOperator::And,
            &known_int(2, IntegerType::Integer),
        );
        raised(&result, ScalarType::Integer(IntegerType::Integer));
    }

    /// Constant numeric pairs fold at the numeric scale rules, and a pair
    /// that does not fit the `numeric` format raises.
    #[test]
    fn constant_numeric_folds_keep_the_numeric_rules() {
        assert_eq!(
            folded(&known_number("1.5"), &BinaryOperator::Plus, &known_number("2.25")),
            PgValue::Numeric(number("3.75"))
        );
        assert_eq!(
            folded(&known_number("7.5"), &BinaryOperator::Minus, &known_number("3.25")),
            PgValue::Numeric(number("4.25"))
        );
        assert_eq!(
            folded(&known_number("1.5"), &BinaryOperator::Multiply, &known_number("2.0")),
            PgValue::Numeric(number("3.00"))
        );
        assert_eq!(
            folded(&known_number("7.5"), &BinaryOperator::Divide, &known_number("2")),
            PgValue::Numeric(number("3.75000000000000000000"))
        );
        assert_eq!(
            folded(&known_number("7.5"), &BinaryOperator::Modulo, &known_number("2")),
            PgValue::Numeric(number("1.5"))
        );
        let result = arithmetic(&known_number("1.5"), &BinaryOperator::Divide, &known_number("0"));
        raised(&result, ScalarType::Numeric);

        let big = known_number(&"9".repeat(131_072));
        let result = arithmetic(&big, &BinaryOperator::Plus, &known_number("1"));
        raised(&result, ScalarType::Numeric);

        // a non-arithmetic operator on numerics leaves the values undecided
        let result = arithmetic(&known_number("1.5"), &BinaryOperator::Or, &known_number("2.0"));
        assert_eq!(result.ty(), ScalarType::Numeric);
        assert!(!result.may_raise());
        assert!(matches!(result.values(), Values::Unlisted(_)));
    }

    /// `date + integer` and `integer + date` count days, `date - date`
    /// gives a day count, and a `timestamp` with an `integer` raises.
    #[test]
    fn date_arithmetic_counts_days() {
        let result = arithmetic(
            &known_date("2020-01-31"),
            &BinaryOperator::Plus,
            &known_int(1, IntegerType::Integer),
        );
        assert_eq!(result.ty(), ScalarType::Date);
        assert!(!result.may_raise());
        assert_eq!(
            result.constant(),
            Some(&PgValue::Date(sql_values::postgres::parse_date("2020-02-01").expect("a date")))
        );
        let result = arithmetic(
            &known_int(1, IntegerType::Integer),
            &BinaryOperator::Plus,
            &known_date("2020-01-31"),
        );
        assert_eq!(
            result.constant(),
            Some(&PgValue::Date(sql_values::postgres::parse_date("2020-02-01").expect("a date")))
        );
        let result = arithmetic(
            &known_date("2020-01-31"),
            &BinaryOperator::Minus,
            &known_int(1, IntegerType::Integer),
        );
        assert_eq!(
            result.constant(),
            Some(&PgValue::Date(sql_values::postgres::parse_date("2020-01-30").expect("a date")))
        );

        // a date minus a date is the day span, of type `int4`
        let result = arithmetic(
            &known_date("2020-01-05"),
            &BinaryOperator::Minus,
            &known_date("2020-01-01"),
        );
        assert_eq!(result.ty(), ScalarType::Integer(IntegerType::Integer));
        assert_eq!(result.constant(), Some(&PgValue::Integer(4)));
        let result = arithmetic(
            &known_date("2020-01-01"),
            &BinaryOperator::Minus,
            &known_date("2020-01-05"),
        );
        assert_eq!(result.constant(), Some(&PgValue::Integer(-4)));
        // A date column may hold `infinity`, which `date - date` refuses.
        let dates =
            Evaluation::new(ScalarType::Date, false, false, Values::Unlisted(Unlisted::any()));
        let result = arithmetic(&dates, &BinaryOperator::Minus, &dates);
        assert_eq!(result.ty(), ScalarType::Integer(IntegerType::Integer));
        assert!(result.may_raise());
        let infinite = Evaluation::known(
            PgValue::Date(sql_values::postgres::PgDate::INFINITY),
            ScalarType::Date,
        );
        let result = arithmetic(&infinite, &BinaryOperator::Minus, &known_date("2020-01-01"));
        assert!(result.always_raises());
    }

    /// A date arithmetic that cannot hold its result raises, and an
    /// operator the date-integer pair does not take raises as well.
    #[test]
    fn date_arithmetic_that_cannot_hold_the_result_raises() {
        // a timestamp has no operator with an integer
        let timestamp =
            Evaluation::new(ScalarType::Timestamp, false, false, Values::Unlisted(Unlisted::any()));
        let result =
            arithmetic(&timestamp, &BinaryOperator::Plus, &known_int(1, IntegerType::Integer));
        raised(&result, ScalarType::Timestamp);
        let result =
            arithmetic(&known_int(1, IntegerType::Integer), &BinaryOperator::Plus, &timestamp);
        raised(&result, ScalarType::Timestamp);

        // PostgreSQL has no `date + bigint` or `bigint + date`
        let result = arithmetic(
            &known_date("2020-01-01"),
            &BinaryOperator::Plus,
            &known_int(1, IntegerType::Bigint),
        );
        raised(&result, ScalarType::Date);
        let result = arithmetic(
            &known_int(1, IntegerType::Bigint),
            &BinaryOperator::Plus,
            &known_date("2020-01-01"),
        );
        raised(&result, ScalarType::Date);

        // no other operator applies to a date with an integer
        let result = arithmetic(
            &known_date("2020-01-01"),
            &BinaryOperator::Multiply,
            &known_int(2, IntegerType::Integer),
        );
        raised(&result, ScalarType::Date);
        let result = arithmetic(
            &known_int(2, IntegerType::Integer),
            &BinaryOperator::Minus,
            &known_date("2020-01-01"),
        );
        raised(&result, ScalarType::Date);
        let result = arithmetic(
            &known_int(2, IntegerType::Integer),
            &BinaryOperator::Divide,
            &known_date("2020-01-01"),
        );
        raised(&result, ScalarType::Date);

        // an unlisted day count may shift the date out of range, so it
        // raises
        let dates =
            Evaluation::new(ScalarType::Date, false, false, Values::Unlisted(Unlisted::any()));
        let result = arithmetic(
            &dates,
            &BinaryOperator::Plus,
            &unlisted(ScalarType::Integer(IntegerType::Integer), "1", "2"),
        );
        raised(&result, ScalarType::Date);

        // a shift past the date range raises, leaving no value behind
        let result = arithmetic(
            &Evaluation::known(
                PgValue::Date(sql_values::postgres::PgDate::from_ymd(5_874_897, 12, 31).unwrap()),
                ScalarType::Date,
            ),
            &BinaryOperator::Plus,
            &known_int(1, IntegerType::Integer),
        );
        raised(&result, ScalarType::Date);
        let Values::Listed(listed) = result.values() else { panic!("expected listed values") };
        assert!(listed.is_empty());
        let result = arithmetic(
            &Evaluation::known(
                PgValue::Date(sql_values::postgres::PgDate::from_ymd(-4713, 11, 24).unwrap()),
                ScalarType::Date,
            ),
            &BinaryOperator::Minus,
            &known_int(1, IntegerType::Integer),
        );
        raised(&result, ScalarType::Date);

        // a value the type cannot hold is simply skipped
        let days = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Numeric(number("1.5"))])),
        );
        let result = arithmetic(&known_date("2020-01-01"), &BinaryOperator::Plus, &days);
        assert_eq!(
            result.constant(),
            Some(&PgValue::Date(sql_values::postgres::parse_date("2020-01-02").expect("a date")))
        );
    }

    /// The interval arithmetic of `+` and `-` moves both bounds, open when
    /// either side is, unbounded when either side is, and adds an error when
    /// a bound overflows the `numeric` format.
    #[test]
    fn the_interval_arithmetic_of_addition_and_subtraction() {
        let result = arithmetic(
            &unlisted(ScalarType::Numeric, "1", "2"),
            &BinaryOperator::Plus,
            &unlisted(ScalarType::Numeric, "3", "4"),
        );
        assert_eq!(
            result_numbers(&result),
            Interval { low: Bound::Included(number("4")), high: Bound::Included(number("6")) }
        );
        assert!(!result.may_raise());

        let result = arithmetic(
            &unlisted(ScalarType::Numeric, "1", "10"),
            &BinaryOperator::Minus,
            &unlisted(ScalarType::Numeric, "2", "3"),
        );
        assert_eq!(
            result_numbers(&result),
            Interval { low: Bound::Included(number("-2")), high: Bound::Included(number("8")) }
        );

        let interval =
            Interval { low: Bound::Excluded(number("1")), high: Bound::Included(number("2")) };
        let result = interval_add(
            &interval,
            &Interval { low: Bound::Included(number("3")), high: Bound::Included(number("4")) },
            &mut false,
        );
        assert_eq!(result.low, Bound::Excluded(number("4")));
        assert_eq!(result.high, Bound::Included(number("6")));

        let result = interval_add(
            &Interval { low: Bound::Included(number("1")), high: Bound::Unbounded },
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &mut false,
        );
        assert_eq!(result.low, Bound::Included(number("2")));
        assert_eq!(result.high, Bound::Unbounded);

        let mut error = false;
        let result = interval_add(
            &Interval { low: Bound::Unbounded, high: Bound::Unbounded },
            &Interval { low: Bound::Unbounded, high: Bound::Unbounded },
            &mut error,
        );
        assert!(result.is_unbounded());
        assert!(!error);

        // a bound past the numeric format overflows to the unbounded
        let big = number(&"9".repeat(131_072));
        let mut error = false;
        let result = interval_add(
            &Interval { low: Bound::Included(big.clone()), high: Bound::Included(big) },
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("1")) },
            &mut error,
        );
        assert!(result.is_unbounded());
        assert!(error);

        assert_eq!(bound_number(&Bound::Unbounded), PgNumeric::from(0_i64));
        assert_eq!(unbind(Bound::Unbounded), (PgNumeric::from(0_i64), false));
    }

    /// The interval arithmetic of `*` takes the corners of the sign
    /// combinations, widens an unbounded factor to the unbounded, and adds an
    /// error when a corner overflows the `numeric` format.
    #[test]
    fn the_interval_arithmetic_of_multiplication() {
        let result = arithmetic(
            &unlisted(ScalarType::Numeric, "-2", "3"),
            &BinaryOperator::Multiply,
            &unlisted(ScalarType::Numeric, "-1", "2"),
        );
        assert_eq!(
            result_numbers(&result),
            Interval { low: Bound::Included(number("-4")), high: Bound::Included(number("6")) }
        );

        let interval =
            Interval { low: Bound::Excluded(number("-2")), high: Bound::Excluded(number("3")) };
        let result = interval_mul(
            &interval,
            &Interval { low: Bound::Excluded(number("-1")), high: Bound::Excluded(number("2")) },
            &mut false,
        );
        assert_eq!(result.low, Bound::Excluded(number("-4")));
        assert_eq!(result.high, Bound::Excluded(number("6")));

        let result = interval_mul(
            &Interval { low: Bound::Unbounded, high: Bound::Unbounded },
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &mut false,
        );
        assert!(result.is_unbounded());

        // a positive product past the format overflows the high end
        let big = number(&format!("1e{}", 70000));
        let bigger = number(&format!("1e{}", 70001));
        let big_interval = Interval { low: Bound::Included(big), high: Bound::Included(bigger) };
        let mut error = false;
        let result = interval_mul(&big_interval, &big_interval, &mut error);
        assert!(result.is_unbounded());
        assert!(error);

        // a mixed-sign overflow only widens the low end
        let interval = Interval {
            low: Bound::Included(number(&format!("-1e{}", 70001))),
            high: Bound::Included(number("-1")),
        };
        let result = interval_mul(&interval, &big_interval, &mut error);
        assert_eq!(result.low, Bound::Unbounded);
        assert_eq!(result.high, Bound::Included(number(&format!("-1e{}", 70000))));

        // a same-sign negative overflow only widens the high end
        let interval = Interval {
            low: Bound::Included(number(&format!("-1e{}", 70001))),
            high: Bound::Included(number(&format!("-1e{}", 70000))),
        };
        let result = interval_mul(
            &interval,
            &Interval {
                low: Bound::Included(number(&format!("-1e{}", 70001))),
                high: Bound::Included(number("1")),
            },
            &mut error,
        );
        assert_eq!(result.low, Bound::Included(number(&format!("-1e{}", 70001))));
        assert_eq!(result.high, Bound::Unbounded);
    }

    /// The interval arithmetic of `/` takes the corners, refuses a divisor
    /// that contains zero, and widens an overflow to the end it holds.
    #[test]
    fn the_interval_arithmetic_of_division() {
        let result = arithmetic(
            &unlisted(ScalarType::Numeric, "1", "4"),
            &BinaryOperator::Divide,
            &unlisted(ScalarType::Numeric, "2", "4"),
        );
        assert_eq!(
            result_numbers(&result),
            Interval { low: Bound::Included(number("0.25")), high: Bound::Included(number("2")) }
        );
        assert!(!result.may_raise());

        let result = interval_div(
            &Interval { low: Bound::Unbounded, high: Bound::Unbounded },
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &mut false,
        );
        assert!(result.is_unbounded());

        // a quotient by a bound at zero errors and widens the end it holds
        let mut error = false;
        let result = interval_div(
            &Interval { low: Bound::Included(number("-2")), high: Bound::Included(number("-1")) },
            &Interval { low: Bound::Excluded(number("0")), high: Bound::Included(number("1")) },
            &mut error,
        );
        assert_eq!(result.low, Bound::Unbounded);
        assert_eq!(result.high, Bound::Included(number("-1")));
        assert!(error);

        let result = interval_div(
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &Interval { low: Bound::Excluded(number("0")), high: Bound::Included(number("1")) },
            &mut error,
        );
        assert_eq!(result.low, Bound::Included(number("1")));
        assert_eq!(result.high, Bound::Unbounded);

        // a quotient whose every corner errors is only the unbounded
        let result = interval_div(
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("1")) },
            &Interval { low: Bound::Excluded(number("0")), high: Bound::Excluded(number("0")) },
            &mut error,
        );
        assert!(result.is_unbounded());
        assert!(error);
    }

    /// A divisor that may be zero adds an error and leaves the values
    /// undecided, for `/` and `%` alike.
    #[test]
    fn a_divisor_that_may_be_zero_adds_an_error() {
        let (values, error) = interval_values(
            &BinaryOperator::Divide,
            ScalarType::Numeric,
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &Interval { low: Bound::Included(number("-1")), high: Bound::Included(number("1")) },
            false,
        );
        assert!(matches!(values, Values::Unlisted(_)));
        assert!(error);

        let (values, error) = interval_values(
            &BinaryOperator::Modulo,
            ScalarType::Numeric,
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &Interval {
                low: Bound::Included(number("-0.5")),
                high: Bound::Included(number("0.5")),
            },
            false,
        );
        assert!(matches!(values, Values::Unlisted(_)));
        assert!(error);

        let (values, error) = interval_values(
            &BinaryOperator::Modulo,
            ScalarType::Numeric,
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("1")) },
            false,
        );
        assert!(matches!(values, Values::Unlisted(_)));
        assert!(!error);

        // a non-arithmetic operator leaves the values undecided and adds an
        // error
        let (values, error) = interval_values(
            &BinaryOperator::And,
            ScalarType::Numeric,
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            &Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) },
            false,
        );
        assert!(matches!(values, Values::Unlisted(_)));
        assert!(error);
    }

    /// An integer result keeps only the integer values the interval admits,
    /// and raises when the interval leaves the type's width.
    #[test]
    fn an_integer_result_keeps_the_integer_values() {
        let result = arithmetic(
            &unlisted(ScalarType::Integer(IntegerType::Integer), "1", "2"),
            &BinaryOperator::Plus,
            &unlisted(ScalarType::Integer(IntegerType::Integer), "3", "4"),
        );
        assert_eq!(
            result_numbers(&result),
            Interval { low: Bound::Included(number("4")), high: Bound::Included(number("6")) }
        );
        assert!(!result.may_raise());

        // an endpoint past the width adds an error
        let result = arithmetic(
            &unlisted(ScalarType::Integer(IntegerType::Integer), "2147483646", "2147483647"),
            &BinaryOperator::Plus,
            &unlisted(ScalarType::Integer(IntegerType::Integer), "1", "1"),
        );
        assert!(result.may_raise());
        let result = arithmetic(
            &unlisted(ScalarType::Integer(IntegerType::Integer), "-2147483648", "-2147483647"),
            &BinaryOperator::Plus,
            &unlisted(ScalarType::Integer(IntegerType::Integer), "-1", "0"),
        );
        assert!(result.may_raise());

        // a bound at the width, excluded or not, admits no overflow
        let (values, error) = interval_values(
            &BinaryOperator::Minus,
            ScalarType::Integer(IntegerType::Integer),
            &Interval {
                low: Bound::Excluded(number("-2147483648")),
                high: Bound::Included(number("0")),
            },
            &Interval { low: Bound::Included(number("0")), high: Bound::Included(number("0")) },
            false,
        );
        assert!(!error);
        assert_eq!(
            values_numbers(&values),
            Interval {
                low: Bound::Included(number("-2147483647")),
                high: Bound::Included(number("0")),
            }
        );
        let (values, error) = interval_values(
            &BinaryOperator::Minus,
            ScalarType::Integer(IntegerType::Integer),
            &Interval {
                low: Bound::Included(number("-2147483648")),
                high: Bound::Included(number("0")),
            },
            &Interval { low: Bound::Included(number("0")), high: Bound::Included(number("0")) },
            false,
        );
        assert!(!error);
        assert_eq!(
            values_numbers(&values),
            Interval {
                low: Bound::Included(number("-2147483648")),
                high: Bound::Included(number("0")),
            }
        );
        let (_, error) = interval_values(
            &BinaryOperator::Minus,
            ScalarType::Integer(IntegerType::Integer),
            &Interval {
                low: Bound::Included(number("-2147483648.5")),
                high: Bound::Included(number("0")),
            },
            &Interval { low: Bound::Included(number("0")), high: Bound::Included(number("0")) },
            false,
        );
        assert!(error);

        // an unlisted endpoint the width cannot hold raises
        let result = arithmetic(
            &Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                false,
                false,
                Values::Unlisted(Unlisted::any()),
            ),
            &BinaryOperator::Plus,
            &unlisted(ScalarType::Integer(IntegerType::Integer), "1", "2"),
        );
        assert!(result.may_raise());

        // a numeric result keeps the non-integer endpoints as they are
        let result = arithmetic(
            &unlisted(ScalarType::Numeric, "0.5", "1.5"),
            &BinaryOperator::Plus,
            &unlisted(ScalarType::Numeric, "0.5", "0.5"),
        );
        assert_eq!(
            result_numbers(&result),
            Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) }
        );
    }

    /// The integer results of an interval keep the integer endpoints and
    /// widen a past-format bound to the unbounded.
    #[test]
    fn integer_result_intervals_keep_the_integer_endpoints() {
        // the integer results of an interval with non-integer endpoints
        let interval =
            Interval { low: Bound::Included(number("0.5")), high: Bound::Included(number("5.5")) };
        assert_eq!(
            integer_result_interval(&interval),
            Interval { low: Bound::Included(number("0")), high: Bound::Included(number("6")) }
        );
        let interval =
            Interval { low: Bound::Excluded(number("0.5")), high: Bound::Excluded(number("5.5")) };
        assert_eq!(
            integer_result_interval(&interval),
            Interval { low: Bound::Included(number("1")), high: Bound::Included(number("5")) }
        );
        let interval =
            Interval { low: Bound::Excluded(number("3")), high: Bound::Excluded(number("10")) };
        assert_eq!(
            integer_result_interval(&interval),
            Interval { low: Bound::Included(number("4")), high: Bound::Included(number("9")) }
        );
        assert_eq!(
            integer_result_interval(&Interval { low: Bound::Unbounded, high: Bound::Unbounded }),
            Interval { low: Bound::Unbounded, high: Bound::Unbounded }
        );

        // a bound whose integer neighbour is past the format widens to the
        // unbounded, which keeps the interval sound
        let nines = "9".repeat(131_072);
        let neg_half = number(&format!("-{nines}.5"));
        assert_eq!(floor_number(&neg_half), None);
        assert_eq!(ceil_number(&number(&format!("{nines}.5"))), None);
        let result = integer_result_interval(&Interval {
            low: Bound::Excluded(neg_half),
            high: Bound::Included(number(&nines)),
        });
        assert_eq!(result.low, Bound::Unbounded);
        assert_eq!(result.high, Bound::Included(number(&nines)));
        let result = integer_result_interval(&Interval {
            low: Bound::Unbounded,
            high: Bound::Excluded(number(&format!("{nines}.5"))),
        });
        assert_eq!(result.high, Bound::Unbounded);
    }

    /// The floor and ceil of a number round to the surrounding integers, or
    /// to none when the surrounding integer is past the format.
    #[test]
    fn the_floor_and_ceil_round_to_the_integer() {
        assert_eq!(floor_number(&number("2.5")), Some(number("2")));
        assert_eq!(floor_number(&number("-2.5")), Some(number("-3")));
        assert_eq!(floor_number(&number("2.0")), Some(number("2")));
        assert_eq!(ceil_number(&number("2.5")), Some(number("3")));
        assert_eq!(ceil_number(&number("-2.5")), Some(number("-2")));
        assert_eq!(ceil_number(&number("2.0")), Some(number("2")));
    }

    /// The numbers a `numeric` evaluation lies between.
    fn numeric_between(low: Bound<PgNumeric>, high: Bound<PgNumeric>) -> Evaluation {
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

    /// A factor or divisor bounded on one side may be any size on the other,
    /// so the product and the quotient admit every value it reaches.
    #[test]
    fn a_half_bounded_operand_keeps_the_values_it_reaches() {
        let from_zero = Interval { low: Bound::Included(number("0")), high: Bound::Unbounded };
        let one_two =
            Interval { low: Bound::Included(number("1")), high: Bound::Included(number("2")) };
        // `5 * 2`, `2 * 5` and `10 / 1`.
        assert!(interval_mul(&from_zero, &one_two, &mut false).contains(&number("10")));
        assert!(interval_mul(&one_two, &from_zero, &mut false).contains(&number("10")));
        assert!(interval_div(&from_zero, &one_two, &mut false).contains(&number("10")));
        // `1 / 1000000`.
        let from_one = Interval { low: Bound::Included(number("1")), high: Bound::Unbounded };
        assert!(interval_div(&one_two, &from_one, &mut false).contains(&number("0.000001")));
        // `-Infinity * 0` is `NaN`, which sorts above every number.
        let to_zero = Interval { low: Bound::Unbounded, high: Bound::Included(number("0")) };
        let zero =
            Interval { low: Bound::Included(number("0")), high: Bound::Included(number("0")) };
        assert!(interval_mul(&to_zero, &zero, &mut false).contains(&PgNumeric::NAN));
    }

    /// A `numeric` operand that may be `NaN` or an infinity makes the
    /// difference `NaN`, as `2 - 'NaN'` and `'Infinity' - 'Infinity'` are.
    #[test]
    fn a_numeric_difference_admits_nan() {
        let one_two = numeric_between(Bound::Included(number("1")), Bound::Included(number("2")));
        let from_zero = numeric_between(Bound::Included(number("0")), Bound::Unbounded);
        let result = arithmetic(&one_two, &BinaryOperator::Minus, &from_zero);
        assert!(result_numbers(&result).contains(&PgNumeric::NAN));
        let to_infinity =
            numeric_between(Bound::Included(number("0")), Bound::Included(PgNumeric::INFINITY));
        let result = arithmetic(&to_infinity, &BinaryOperator::Minus, &to_infinity);
        assert!(result_numbers(&result).contains(&PgNumeric::NAN));
    }
}
