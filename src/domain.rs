//! The sets of outcomes an expression can evaluate to.

use alloc::collections::BTreeSet;
use core::{cmp::Ordering, ops::Bound};

use sql_values::{PgNumeric, PgValue};

use super::{ScalarType, scalar::rounded_i64};

/// The most values a [`Values::Listed`] set holds before it widens into the
/// [`Unlisted`] hull of its members.
const LISTED_LIMIT: usize = 64;

/// The values between two bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Interval<T> {
    /// The lower bound.
    pub low: Bound<T>,
    /// The upper bound.
    pub high: Bound<T>,
}

/// Which of two lower or two upper bounds admits fewer values, `Less` for
/// the first.
fn tighter<T: Ord>(left: &Bound<T>, right: &Bound<T>, lower: bool) -> Ordering {
    match (left, right) {
        (Bound::Unbounded, Bound::Unbounded) => Ordering::Equal,
        (Bound::Unbounded, _) => Ordering::Greater,
        (_, Bound::Unbounded) => Ordering::Less,
        (
            Bound::Included(left_value) | Bound::Excluded(left_value),
            Bound::Included(right_value) | Bound::Excluded(right_value),
        ) => {
            let by_value =
                if lower { right_value.cmp(left_value) } else { left_value.cmp(right_value) };
            by_value.then_with(|| {
                match (left, right) {
                    (Bound::Excluded(_), Bound::Included(_)) => Ordering::Less,
                    (Bound::Included(_), Bound::Excluded(_)) => Ordering::Greater,
                    _ => Ordering::Equal,
                }
            })
        }
    }
}

impl<T: Ord + Clone> Interval<T> {
    /// Every value.
    #[must_use]
    pub fn unbounded() -> Self {
        Self { low: Bound::Unbounded, high: Bound::Unbounded }
    }

    /// The one value `value`.
    #[must_use]
    pub fn point(value: T) -> Self {
        Self { low: Bound::Included(value.clone()), high: Bound::Included(value) }
    }

    /// Whether the interval bounds neither side.
    #[must_use]
    pub fn is_unbounded(&self) -> bool {
        matches!((&self.low, &self.high), (Bound::Unbounded, Bound::Unbounded))
    }

    /// Whether `value` lies within the interval.
    #[must_use]
    pub fn contains(&self, value: &T) -> bool {
        let above = match &self.low {
            Bound::Unbounded => true,
            Bound::Included(low) => value >= low,
            Bound::Excluded(low) => value > low,
        };
        let below = match &self.high {
            Bound::Unbounded => true,
            Bound::Included(high) => value <= high,
            Bound::Excluded(high) => value < high,
        };
        above && below
    }

    /// Whether no value lies within the interval.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match (&self.low, &self.high) {
            (Bound::Included(low), Bound::Included(high)) => low > high,
            (
                Bound::Included(low) | Bound::Excluded(low),
                Bound::Included(high) | Bound::Excluded(high),
            ) => low >= high,
            _ => false,
        }
    }

    /// The values within both intervals.
    #[must_use]
    pub fn meet(&self, other: &Self) -> Self {
        let pick = |left: &Bound<T>, right: &Bound<T>, lower: bool| {
            if tighter(left, right, lower).is_le() { left.clone() } else { right.clone() }
        };
        Self { low: pick(&self.low, &other.low, true), high: pick(&self.high, &other.high, false) }
    }

    /// The smallest interval holding both intervals.
    #[must_use]
    pub fn hull(&self, other: &Self) -> Self {
        let pick = |left: &Bound<T>, right: &Bound<T>, lower: bool| {
            if tighter(left, right, lower).is_ge() { left.clone() } else { right.clone() }
        };
        Self { low: pick(&self.low, &other.low, true), high: pick(&self.high, &other.high, false) }
    }
}

/// The non-`NULL` values that are not listed one by one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Unlisted {
    /// Values known not to occur.
    pub excluded: BTreeSet<PgValue>,
    /// The interval numbers lie in, unbounded for other types.
    pub numbers: Interval<PgNumeric>,
    /// The interval the character length of text lies in, unbounded for
    /// other types.
    pub length: Interval<u64>,
}

impl Unlisted {
    /// Every value.
    #[must_use]
    pub fn any() -> Self {
        Self {
            excluded: BTreeSet::new(),
            numbers: Interval::unbounded(),
            length: Interval::unbounded(),
        }
    }

    /// Whether `value` is among these values.
    #[must_use]
    pub fn admits(&self, value: &PgValue) -> bool {
        if self.excluded.contains(value) {
            return false;
        }
        if let Some(number) = value.as_numeric()
            && !self.numbers.contains(&number)
        {
            return false;
        }
        crate::scalar::text_length(value).is_none_or(|length| self.length.contains(&length))
    }
}

/// The non-`NULL` values an expression can take.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Values {
    /// One of these values, none at all when the set is empty.
    Listed(BTreeSet<PgValue>),
    /// Any value of the type that the description admits.
    Unlisted(Unlisted),
}

impl Values {
    /// No value.
    #[must_use]
    pub fn none() -> Self {
        Self::Listed(BTreeSet::new())
    }

    /// Whether `value` is among these values.
    #[must_use]
    pub fn admits(&self, value: &PgValue) -> bool {
        match self {
            Self::Listed(listed) => listed.contains(value),
            Self::Unlisted(unlisted) => unlisted.admits(value),
        }
    }

    /// Whether there is no value.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Listed(listed) if listed.is_empty())
    }

    fn meet(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Listed(left), Self::Listed(right)) => {
                Self::Listed(left.intersection(right).cloned().collect())
            }
            (Self::Listed(listed), Self::Unlisted(unlisted))
            | (Self::Unlisted(unlisted), Self::Listed(listed)) => {
                Self::Listed(
                    listed.iter().filter(|value| unlisted.admits(value)).cloned().collect(),
                )
            }
            (Self::Unlisted(left), Self::Unlisted(right)) => {
                Self::Unlisted(Unlisted {
                    excluded: left.excluded.union(&right.excluded).cloned().collect(),
                    numbers: left.numbers.meet(&right.numbers),
                    length: left.length.meet(&right.length),
                })
            }
        }
    }

    fn join(&self, other: &Self) -> Self {
        match (self, other) {
            (Self::Listed(left), Self::Listed(right)) => {
                Self::Listed(left.union(right).cloned().collect())
            }
            (Self::Listed(listed), Self::Unlisted(unlisted))
            | (Self::Unlisted(unlisted), Self::Listed(listed)) => {
                let mut widened = unlisted.clone();
                widened.excluded.retain(|value| !listed.contains(value));
                widen_to(&mut widened, listed);
                Self::Unlisted(widened)
            }
            (Self::Unlisted(left), Self::Unlisted(right)) => {
                Self::Unlisted(Unlisted {
                    excluded: left.excluded.intersection(&right.excluded).cloned().collect(),
                    numbers: left.numbers.hull(&right.numbers),
                    length: left.length.hull(&right.length),
                })
            }
        }
    }
}

/// Widens the intervals of `unlisted` to hold every member of `listed`.
fn widen_to(unlisted: &mut Unlisted, listed: &BTreeSet<PgValue>) {
    for value in listed {
        if let Some(number) = value.as_numeric() {
            unlisted.numbers = unlisted.numbers.hull(&Interval::point(number));
        }
        if let Some(length) = crate::scalar::text_length(value) {
            unlisted.length = unlisted.length.hull(&Interval::point(length));
        }
    }
}

/// The hull of `listed` once it holds too many values to list.
fn hull_of(listed: &BTreeSet<PgValue>) -> Unlisted {
    let mut numbers = None::<Interval<PgNumeric>>;
    let mut length = None::<Interval<u64>>;
    let mut every_number = true;
    let mut every_text = true;
    for value in listed {
        match value.as_numeric() {
            Some(number) => {
                let point = Interval::point(number);
                numbers = Some(numbers.map_or_else(|| point.clone(), |hull| hull.hull(&point)));
            }
            None => every_number = false,
        }
        match crate::scalar::text_length(value) {
            Some(characters) => {
                let point = Interval::point(characters);
                length = Some(length.map_or_else(|| point.clone(), |hull| hull.hull(&point)));
            }
            None => every_text = false,
        }
    }
    Unlisted {
        excluded: BTreeSet::new(),
        numbers: numbers.filter(|_| every_number).unwrap_or_else(Interval::unbounded),
        length: length.filter(|_| every_text).unwrap_or_else(Interval::unbounded),
    }
}

/// The smallest integer at or above `bound` taken as a lower bound, `None`
/// when it is unbounded or out of `i64`.
fn integer_low(bound: &Bound<PgNumeric>) -> Option<i64> {
    match bound {
        Bound::Unbounded => None,
        Bound::Included(value) => {
            let rounded = rounded_i64(value)?;
            Some(if PgNumeric::from(rounded) < *value { rounded.checked_add(1)? } else { rounded })
        }
        Bound::Excluded(value) => {
            let rounded = rounded_i64(value)?;
            Some(if PgNumeric::from(rounded) <= *value { rounded.checked_add(1)? } else { rounded })
        }
    }
}

/// The largest integer at or below `bound` taken as an upper bound, `None`
/// when it is unbounded or out of `i64`.
fn integer_high(bound: &Bound<PgNumeric>) -> Option<i64> {
    match bound {
        Bound::Unbounded => None,
        Bound::Included(value) => {
            let rounded = rounded_i64(value)?;
            Some(if PgNumeric::from(rounded) > *value { rounded.checked_sub(1)? } else { rounded })
        }
        Bound::Excluded(value) => {
            let rounded = rounded_i64(value)?;
            Some(if PgNumeric::from(rounded) >= *value { rounded.checked_sub(1)? } else { rounded })
        }
    }
}

/// Which of true, false and `NULL` a condition may evaluate to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Outcomes {
    /// Whether it may be true.
    pub truth: bool,
    /// Whether it may be false.
    pub falsity: bool,
    /// Whether it may be `NULL`.
    pub null: bool,
}

impl Outcomes {
    /// True alone.
    pub const TRUE: Self = Self { truth: true, falsity: false, null: false };
    /// False alone.
    pub const FALSE: Self = Self { truth: false, falsity: true, null: false };
    /// `NULL` alone.
    pub const NULL: Self = Self { truth: false, falsity: false, null: true };
    /// True or `NULL`, what a CHECK constraint accepts.
    pub const NOT_FALSE: Self = Self { truth: true, falsity: false, null: true };
    /// False or `NULL`, what a `WHERE` clause or a policy rejects.
    pub const NOT_TRUE: Self = Self { truth: false, falsity: true, null: true };
    /// Any of the three.
    pub const ANY: Self = Self { truth: true, falsity: true, null: true };
    /// None of the three.
    pub const NONE: Self = Self { truth: false, falsity: false, null: false };

    /// The outcomes in either set.
    #[must_use]
    pub fn union(self, other: Self) -> Self {
        Self {
            truth: self.truth || other.truth,
            falsity: self.falsity || other.falsity,
            null: self.null || other.null,
        }
    }

    /// The outcomes in both sets.
    #[must_use]
    pub fn intersection(self, other: Self) -> Self {
        Self {
            truth: self.truth && other.truth,
            falsity: self.falsity && other.falsity,
            null: self.null && other.null,
        }
    }

    /// Whether the set is empty.
    #[must_use]
    pub fn is_empty(self) -> bool {
        !(self.truth || self.falsity || self.null)
    }

    /// The outcomes with true and false swapped, what `NOT` makes of them.
    #[must_use]
    pub fn negated(self) -> Self {
        Self { truth: self.falsity, falsity: self.truth, null: self.null }
    }
}

/// What an expression can evaluate to over every row it may see.
///
/// Every field over-approximates: `null` and `error` say what may happen,
/// and `values` holds every non-`NULL` value that may occur.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Evaluation {
    ty: ScalarType,
    null: bool,
    error: bool,
    values: Values,
}

impl Evaluation {
    /// Builds an evaluation and brings `values` to its normal form.
    #[must_use]
    pub fn new(ty: ScalarType, null: bool, error: bool, values: Values) -> Self {
        let mut evaluation = Self { ty, null, error, values };
        evaluation.normalise();
        evaluation
    }

    /// Nothing known: any value of `ty`, `NULL`, or an error.
    #[must_use]
    pub fn unknown(ty: ScalarType) -> Self {
        Self::new(ty, true, true, Values::Unlisted(Unlisted::any()))
    }

    /// Always `NULL`.
    #[must_use]
    pub fn null(ty: ScalarType) -> Self {
        Self::new(ty, true, false, Values::none())
    }

    /// Always `value`.
    #[must_use]
    pub fn known(value: PgValue, ty: ScalarType) -> Self {
        Self::new(ty, false, false, Values::Listed(BTreeSet::from([value])))
    }

    /// Always raises an error.
    #[must_use]
    pub fn raising(ty: ScalarType) -> Self {
        Self::new(ty, false, true, Values::none())
    }

    /// No outcome at all, what an expression evaluates to over no row.
    #[must_use]
    pub fn impossible(ty: ScalarType) -> Self {
        Self::new(ty, false, false, Values::none())
    }

    /// A condition taking `outcomes`, raising when `error` is set.
    #[must_use]
    pub fn condition(outcomes: Outcomes, error: bool) -> Self {
        let mut values = BTreeSet::new();
        if outcomes.truth {
            values.insert(PgValue::Boolean(true));
        }
        if outcomes.falsity {
            values.insert(PgValue::Boolean(false));
        }
        Self::new(ScalarType::Boolean, outcomes.null, error, Values::Listed(values))
    }

    /// The type of the values.
    #[must_use]
    pub fn ty(&self) -> ScalarType {
        self.ty
    }

    /// The non-`NULL` values it may take.
    #[must_use]
    pub fn values(&self) -> &Values {
        &self.values
    }

    /// Whether it may be `NULL`.
    #[must_use]
    pub fn may_be_null(&self) -> bool {
        self.null
    }

    /// Whether it may raise an error.
    #[must_use]
    pub fn may_raise(&self) -> bool {
        self.error
    }

    /// Whether every evaluation that does not raise is `NULL`.
    #[must_use]
    pub fn is_always_null(&self) -> bool {
        self.values.is_empty() && !self.error && self.null
    }

    /// Whether every evaluation raises an error.
    #[must_use]
    pub fn always_raises(&self) -> bool {
        self.values.is_empty() && !self.null && self.error
    }

    /// Whether this is the nothing-known reading over the untyped
    /// `Unknown`: nullable, able to raise, and admitting any value. The
    /// forward pass answers it when it gives up at the depth limit, and a
    /// callee may resolve to it. Either way it carries no information, so
    /// the backward pass reads it as "narrow nothing": restricting a
    /// subject on a reading that admits every value would rule out the
    /// solutions where the subject takes one of the other values. A
    /// genuine raising value is value-less, which keeps it apart.
    #[must_use]
    pub fn is_uninformative(&self) -> bool {
        self.ty == ScalarType::Unknown
            && self.null
            && self.error
            && matches!(self.values, Values::Unlisted(_))
    }

    /// Whether it has no outcome at all, not even an error.
    #[must_use]
    pub fn is_impossible(&self) -> bool {
        self.values.is_empty() && !self.null && !self.error
    }

    /// Whether it may be the value `value`.
    #[must_use]
    pub fn may_be(&self, value: &PgValue) -> bool {
        self.values.admits(value)
    }

    /// Whether it may be true.
    #[must_use]
    pub fn may_be_true(&self) -> bool {
        self.may_be(&PgValue::Boolean(true))
    }

    /// Whether it may be false.
    #[must_use]
    pub fn may_be_false(&self) -> bool {
        self.may_be(&PgValue::Boolean(false))
    }

    /// Which of true, false and `NULL` it may be.
    #[must_use]
    pub fn outcomes(&self) -> Outcomes {
        Outcomes { truth: self.may_be_true(), falsity: self.may_be_false(), null: self.null }
    }

    /// The one value it takes whenever it neither is `NULL` nor raises.
    #[must_use]
    pub fn constant_value(&self) -> Option<&PgValue> {
        match &self.values {
            Values::Listed(listed) if listed.len() == 1 => listed.first(),
            _ => None,
        }
    }

    /// The one value it always takes: never `NULL`, never an error.
    #[must_use]
    pub fn constant(&self) -> Option<&PgValue> {
        self.constant_value().filter(|_| !self.null && !self.error)
    }

    /// The interval the numbers it may take lie in.
    #[must_use]
    pub fn number_range(&self) -> Interval<PgNumeric> {
        crate::support::number_bounds(&self.values)
    }

    /// The interval the character lengths of the text it may take lie in.
    #[must_use]
    pub fn length_range(&self) -> Interval<u64> {
        crate::support::length_bounds(&self.values)
    }

    /// The same values with the `NULL` possibility set to `null`.
    #[must_use]
    pub fn with_null(mut self, null: bool) -> Self {
        self.null = null;
        self
    }

    /// The same values with the error possibility set to `error`.
    #[must_use]
    pub fn with_error(mut self, error: bool) -> Self {
        self.error = error;
        self
    }

    /// The same outcomes read as type `ty`.
    #[must_use]
    pub fn with_type(mut self, ty: ScalarType) -> Self {
        self.ty = ty;
        self.normalise();
        self
    }

    /// The same `NULL` and error possibilities with the values `values`.
    #[must_use]
    pub fn with_values(mut self, values: Values) -> Self {
        self.values = values;
        self.normalise();
        self
    }

    /// The outcomes both evaluations allow.
    #[must_use]
    pub fn meet(&self, other: &Self) -> Self {
        let ty = if matches!(self.ty, ScalarType::Unknown | ScalarType::UnknownLiteral) {
            other.ty
        } else {
            self.ty
        };
        Self::new(
            ty,
            self.null && other.null,
            self.error && other.error,
            self.values.meet(&other.values),
        )
    }

    /// The outcomes either evaluation allows.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        let unknown = |ty| matches!(ty, ScalarType::Unknown | ScalarType::UnknownLiteral);
        let ty = if self.ty == other.ty || unknown(other.ty) {
            self.ty
        } else if unknown(self.ty) {
            other.ty
        } else {
            ScalarType::Other
        };
        Self::new(
            ty,
            self.null || other.null,
            self.error || other.error,
            self.values.join(&other.values),
        )
    }

    fn normalise(&mut self) {
        self.drop_values_outside_type();
        if let Values::Listed(listed) = &self.values
            && listed.len() > LISTED_LIMIT
        {
            self.values = Values::Unlisted(hull_of(listed));
            return;
        }
        let Values::Unlisted(unlisted) = &mut self.values else {
            return;
        };
        if unlisted.length.low == Bound::Unbounded {
            unlisted.length.low = Bound::Included(0);
        }
        match self.ty {
            ScalarType::Boolean => {
                let listed = [false, true]
                    .into_iter()
                    .map(PgValue::Boolean)
                    .filter(|value| !unlisted.excluded.contains(value))
                    .collect();
                self.values = Values::Listed(listed);
            }
            ScalarType::Integer(integer) => {
                let (min, max) = integer.range().into_inner();
                unlisted.numbers = unlisted.numbers.meet(&Interval {
                    low: Bound::Included(PgNumeric::from(min)),
                    high: Bound::Included(PgNumeric::from(max)),
                });
                match (integer_low(&unlisted.numbers.low), integer_high(&unlisted.numbers.high)) {
                    (Some(low), Some(high)) if low <= high => {
                        let count = i128::from(high) - i128::from(low) + 1;
                        if usize::try_from(count).is_ok_and(|count| count <= LISTED_LIMIT) {
                            let listed = (low..=high)
                                .map(PgValue::Integer)
                                .filter(|value| !unlisted.excluded.contains(value))
                                .collect();
                            self.values = Values::Listed(listed);
                        }
                    }
                    // An endpoint the type's range cannot hold admits no
                    // value of the type, as does an empty range.
                    _ => {
                        self.values = Values::none();
                    }
                }
            }
            ScalarType::Text(_)
                if unlisted.length.contains(&0)
                    && !unlisted.length.contains(&1)
                    && unlisted.length.high != Bound::Unbounded =>
            {
                let empty = PgValue::Text(alloc::string::String::new());
                let listed = BTreeSet::from([empty])
                    .into_iter()
                    .filter(|value| !unlisted.excluded.contains(value))
                    .collect();
                self.values = Values::Listed(listed);
            }
            _ => {}
        }
        let numbers_admit_nothing = |unlisted: &Unlisted| {
            match (&unlisted.numbers.low, &unlisted.numbers.high) {
                (Bound::Included(low), Bound::Included(high)) if low == high => {
                    unlisted.excluded.contains(&PgValue::Numeric(low.clone()))
                }
                _ => unlisted.numbers.is_empty(),
            }
        };
        if let Values::Unlisted(unlisted) = &self.values
            && ((self.ty.is_number() && numbers_admit_nothing(unlisted))
                || (self.ty.is_text() && unlisted.length.is_empty()))
        {
            self.values = Values::none();
        }
    }

    /// Drops the listed values the type cannot hold, as a type resolution
    /// such as a `meet` against a subject's type may leave them behind.
    fn drop_values_outside_type(&mut self) {
        let ty = self.ty;
        let Values::Listed(listed) = &self.values else {
            return;
        };
        let keep = |value: &PgValue| {
            match ty {
                ScalarType::Unknown
                | ScalarType::UnknownLiteral
                | ScalarType::Float
                | ScalarType::Other => true,
                ScalarType::Boolean => matches!(value, PgValue::Boolean(_)),
                ScalarType::Integer(target) => {
                    match value {
                        PgValue::Integer(value) => {
                            let (min, max) = target.range().into_inner();
                            *value >= min && *value <= max
                        }
                        // A numeric value may stand for the same integer.
                        PgValue::Numeric(value) => {
                            let (min, max) = target.range().into_inner();
                            value.is_integral()
                                && *value >= PgNumeric::from(min)
                                && *value <= PgNumeric::from(max)
                        }
                        _ => false,
                    }
                }
                ScalarType::Numeric => value.as_numeric().is_some(),
                ScalarType::Text(_) => matches!(value, PgValue::Text(_)),
                ScalarType::Date => matches!(value, PgValue::Date(_)),
                // A date reads as its midnight in a timestamp context.
                ScalarType::Timestamp => {
                    matches!(value, PgValue::Timestamp(_) | PgValue::Date(_))
                }
            }
        };
        if listed.iter().any(|value| !keep(value)) {
            let kept: BTreeSet<PgValue> =
                listed.iter().filter(|value| keep(value)).cloned().collect();
            self.values = Values::Listed(kept);
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use core::{cmp::Ordering, ops::Bound};

    use sql_values::IntegerType;

    use super::*;
    use crate::{
        ScalarType, TextRules,
        guarantees::{Guarantees, Subject},
    };

    fn num(text: &str) -> PgNumeric {
        PgNumeric::parse(text).unwrap()
    }

    #[test]
    fn tighter_orders_bounds() {
        // An excluded side admits fewer values than an included one
        assert_eq!(
            tighter(&Bound::Excluded(num("5")), &Bound::Included(num("5")), true),
            Ordering::Less
        );
        assert_eq!(
            tighter(&Bound::Included(num("5")), &Bound::Excluded(num("5")), true),
            Ordering::Greater
        );
        assert_eq!(
            tighter(&Bound::Included(num("5")), &Bound::Included(num("5")), false),
            Ordering::Equal
        );
        // A lower bound nearer the value is tighter
        assert_eq!(
            tighter(&Bound::Included(num("3")), &Bound::Included(num("5")), true),
            Ordering::Greater
        );
        // An upper bound nearer the value is tighter
        assert_eq!(
            tighter(&Bound::Included(num("5")), &Bound::Included(num("3")), false),
            Ordering::Greater
        );
        assert_eq!(
            tighter(&Bound::<PgNumeric>::Unbounded, &Bound::<PgNumeric>::Unbounded, true),
            Ordering::Equal
        );
        assert_eq!(
            tighter(&Bound::<PgNumeric>::Unbounded, &Bound::Included(num("5")), true),
            Ordering::Greater
        );
        assert_eq!(
            tighter(&Bound::Included(num("5")), &Bound::<PgNumeric>::Unbounded, false),
            Ordering::Less
        );
    }

    #[test]
    fn interval_meet_and_hull() {
        assert!(Interval::<u64>::unbounded().is_unbounded());
        assert!(!Interval::point(1u64).is_unbounded());
        // The tighter low wins a meet, the looser low wins a hull
        let open = Interval { low: Bound::Excluded(5u64), high: Bound::Unbounded };
        let closed = Interval { low: Bound::Included(5u64), high: Bound::Unbounded };
        assert_eq!(open.meet(&closed), open);
        assert_eq!(open.hull(&closed), closed);
        // An excluded endpoint keeps its value out
        assert!(!open.contains(&5));
        assert!(closed.contains(&5));
        // An empty interval admits nothing
        assert!(Interval { low: Bound::Included(5u64), high: Bound::Included(4u64) }.is_empty());
    }

    #[test]
    fn unlisted_admits_by_number_and_length() {
        let mut unlisted = Unlisted::any();
        unlisted.numbers =
            Interval { low: Bound::Included(num("10")), high: Bound::Included(num("20")) };
        unlisted.length = Interval::point(2);
        unlisted.excluded.insert(PgValue::Integer(15));
        assert!(!unlisted.admits(&PgValue::Integer(15)));
        assert!(!unlisted.admits(&PgValue::Integer(5)));
        assert!(unlisted.admits(&PgValue::Text(alloc::string::String::from("ab"))));
        assert!(!unlisted.admits(&PgValue::Text(alloc::string::String::from("abc"))));
    }

    #[test]
    fn join_widens_the_unlisted_side() {
        let listed = Evaluation::new(
            ScalarType::Unknown,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Integer(1),
                PgValue::Text(alloc::string::String::from("ab")),
            ])),
        );
        // A narrow numeric reading joins against the listed side: the
        // intervals widen to hold every member
        let narrow = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::point(num("2")),
                length: Interval::point(3),
            }),
        );
        let joined = listed.join(&narrow);
        assert_eq!(joined.ty(), ScalarType::Numeric);
        assert_eq!(
            joined.values(),
            &Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Included(num("1")),
                    high: Bound::Included(num("2"))
                },
                length: Interval { low: Bound::Included(2), high: Bound::Included(3) },
            })
        );
        // A join against a typed evaluation resolves the type
        let typed = Evaluation::unknown(ScalarType::Unknown).join(&Evaluation::known(
            PgValue::Integer(1),
            ScalarType::Integer(IntegerType::Integer),
        ));
        assert_eq!(typed.ty(), ScalarType::Integer(IntegerType::Integer));
        // Two unrelated types join to `Other`
        let mixed = Evaluation::known(
            PgValue::Text(alloc::string::String::from("a")),
            ScalarType::Text(TextRules::DEFAULT),
        )
        .join(&Evaluation::known(PgValue::Integer(1), ScalarType::Integer(IntegerType::Integer)));
        assert_eq!(mixed.ty(), ScalarType::Other);
    }

    #[test]
    fn hull_of_reads_the_envelope() {
        let integers: BTreeSet<PgValue> = (1..=65).map(PgValue::Integer).collect();
        let hull = hull_of(&integers);
        assert_eq!(
            hull.numbers,
            Interval { low: Bound::Included(num("1")), high: Bound::Included(num("65")) }
        );
        assert!(hull.length.is_unbounded());
        let texts: BTreeSet<PgValue> = BTreeSet::from([
            PgValue::Text(alloc::string::String::from("ab")),
            PgValue::Text(alloc::string::String::from("c")),
        ]);
        let hull = hull_of(&texts);
        assert_eq!(hull.length, Interval { low: Bound::Included(1), high: Bound::Included(2) });
        assert!(hull.numbers.is_unbounded());
        // A mixed hull is unbounded on both sides
        let mixed = hull_of(&BTreeSet::from([
            PgValue::Integer(1),
            PgValue::Text(alloc::string::String::from("ab")),
        ]));
        assert!(mixed.numbers.is_unbounded());
        assert!(mixed.length.is_unbounded());
    }

    #[test]
    fn integer_ends_of_the_number_interval() {
        // `integer_low`
        assert_eq!(integer_low(&Bound::Unbounded), None);
        assert_eq!(integer_low(&Bound::Included(num("5.1"))), Some(6));
        assert_eq!(integer_low(&Bound::Included(num("5"))), Some(5));
        assert_eq!(integer_low(&Bound::Excluded(num("5"))), Some(6));
        assert_eq!(integer_low(&Bound::Excluded(num("4.9"))), Some(5));
        // `integer_high`
        assert_eq!(integer_high(&Bound::Unbounded), None);
        assert_eq!(integer_high(&Bound::Included(num("5.9"))), Some(5));
        assert_eq!(integer_high(&Bound::Included(num("5"))), Some(5));
        assert_eq!(integer_high(&Bound::Excluded(num("5.1"))), Some(5));
        assert_eq!(integer_high(&Bound::Excluded(num("5"))), Some(4));
    }

    #[test]
    fn impossible_evaluation() {
        assert!(Evaluation::impossible(ScalarType::Text(TextRules::DEFAULT)).is_impossible());
        assert!(!Evaluation::null(ScalarType::Text(TextRules::DEFAULT)).is_impossible());
        assert!(!Evaluation::raising(ScalarType::Text(TextRules::DEFAULT)).is_impossible());
    }

    #[test]
    fn normalise_enumerates_small_integer_intervals() {
        let int4 = ScalarType::Integer(IntegerType::Integer);
        // A small interval enumerates its members, honouring exclusions
        let evaluation = Evaluation::new(
            int4,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::from([PgValue::Integer(4)]),
                numbers: Interval {
                    low: Bound::Included(num("1")),
                    high: Bound::Included(num("5")),
                },
                length: Interval::unbounded(),
            }),
        );
        assert_eq!(
            evaluation.values(),
            &Values::Listed(BTreeSet::from([
                PgValue::Integer(1),
                PgValue::Integer(2),
                PgValue::Integer(3),
                PgValue::Integer(5)
            ]))
        );
        // An interval past the listing limit stays a hull
        let evaluation = Evaluation::new(
            ScalarType::Integer(IntegerType::Bigint),
            false,
            false,
            Values::Listed((1..=65).map(PgValue::Integer).collect()),
        );
        assert_eq!(
            evaluation.values(),
            &Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Included(num("1")),
                    high: Bound::Included(num("65"))
                },
                // the hull path widens before the length lower bound is set
                length: Interval::unbounded(),
            })
        );
        // An inverted integer range admits no value of the type
        let evaluation = Evaluation::new(
            int4,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Included(num("5")),
                    high: Bound::Included(num("4")),
                },
                length: Interval::unbounded(),
            }),
        );
        assert!(evaluation.values().is_empty());
        // A low bound past the type's range admits no value
        let evaluation = Evaluation::new(
            ScalarType::Integer(IntegerType::Bigint),
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Excluded(num(&i64::MAX.to_string())),
                    high: Bound::Included(num(&i64::MAX.to_string())),
                },
                length: Interval::unbounded(),
            }),
        );
        assert!(evaluation.values().is_empty());
    }

    #[test]
    fn normalise_collapses_the_empty_string() {
        let text = ScalarType::Text(TextRules::DEFAULT);
        let length = |low: Bound<u64>, high: Bound<u64>| {
            Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::unbounded(),
                length: Interval { low, high },
            }
        };
        // Only the empty string fits
        let evaluation = Evaluation::new(
            text,
            false,
            false,
            Values::Unlisted(length(Bound::Included(0), Bound::Included(0))),
        );
        assert_eq!(
            evaluation.values(),
            &Values::Listed(BTreeSet::from([PgValue::Text(alloc::string::String::new())]))
        );
        // An excluded empty string leaves no value
        let mut excluded = length(Bound::Included(0), Bound::Included(0));
        excluded.excluded.insert(PgValue::Text(alloc::string::String::new()));
        let evaluation = Evaluation::new(text, false, false, Values::Unlisted(excluded));
        assert!(evaluation.values().is_empty());
        // An interval still admitting one character stays unlisted
        let evaluation = Evaluation::new(
            text,
            false,
            false,
            Values::Unlisted(length(Bound::Included(0), Bound::Included(1))),
        );
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    }

    #[test]
    fn normalise_drops_values_the_type_cannot_hold() {
        // An `int4` holds integers and the integrals in its range, not
        // fractions, out-of-range numerics or text
        let evaluation = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Integer(1),
                PgValue::Numeric(num("2")),
                PgValue::Numeric(num("2.5")),
                PgValue::Numeric(num("4294967296")),
                PgValue::Text(alloc::string::String::from("a")),
            ])),
        );
        assert_eq!(
            evaluation.values(),
            &Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Numeric(num("2")),]))
        );
        // A `smallint` range is narrower still
        let evaluation = Evaluation::new(
            ScalarType::Integer(IntegerType::Smallint),
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Numeric(num("32767")),
                PgValue::Numeric(num("32768")),
            ])),
        );
        assert_eq!(
            evaluation.values(),
            &Values::Listed(BTreeSet::from([PgValue::Numeric(num("32767"))]))
        );
        // A `timestamp` holds timestamps and their midnights, not text
        let date = sql_values::parse_date("2024-02-29").unwrap();
        let timestamp = sql_values::parse_timestamp("2024-02-29 12:34:56").unwrap();
        let evaluation = Evaluation::new(
            ScalarType::Timestamp,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Date(date),
                PgValue::Timestamp(timestamp),
                PgValue::Text(alloc::string::String::from("a")),
            ])),
        );
        assert_eq!(
            evaluation.values(),
            &Values::Listed(BTreeSet::from([PgValue::Date(date), PgValue::Timestamp(timestamp)]))
        );
        // A `date` keeps only its dates
        let evaluation = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Date(date), PgValue::Timestamp(timestamp)])),
        );
        assert_eq!(evaluation.values(), &Values::Listed(BTreeSet::from([PgValue::Date(date)])));
        // A `boolean` keeps only booleans
        let evaluation = Evaluation::new(
            ScalarType::Boolean,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Boolean(true), PgValue::Integer(1)])),
        );
        assert_eq!(evaluation.values(), &Values::Listed(BTreeSet::from([PgValue::Boolean(true)])));
        // A `numeric` keeps only numbers
        let evaluation = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Numeric(num("1.5")),
                PgValue::Text(alloc::string::String::from("1")),
            ])),
        );
        assert_eq!(
            evaluation.values(),
            &Values::Listed(BTreeSet::from([PgValue::Numeric(num("1.5"))]))
        );
    }

    #[test]
    fn values_meet_and_join() {
        let text = |text: &str| PgValue::Text(alloc::string::String::from(text));
        // A listed meet keeps the shared members
        assert_eq!(
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)]))
                .meet(&Values::Listed(BTreeSet::from([PgValue::Integer(2), text("a")]))),
            Values::Listed(BTreeSet::from([PgValue::Integer(2)]))
        );
        // A listed side filters through the unlisted side
        let unlisted = Unlisted {
            excluded: BTreeSet::from([PgValue::Integer(1)]),
            numbers: Interval::unbounded(),
            length: Interval::point(1),
        };
        assert_eq!(
            Values::Listed(BTreeSet::from([PgValue::Integer(1), text("a")]))
                .meet(&Values::Unlisted(unlisted)),
            Values::Listed(BTreeSet::from([text("a")]))
        );
        // A listed join unions the members
        assert_eq!(
            Values::Listed(BTreeSet::from([PgValue::Integer(1)]))
                .join(&Values::Listed(BTreeSet::from([text("a")]))),
            Values::Listed(BTreeSet::from([PgValue::Integer(1), text("a")]))
        );
        // An unlisted join keeps the common exclusions
        let left = Unlisted {
            excluded: BTreeSet::from([PgValue::Integer(1)]),
            numbers: Interval::unbounded(),
            length: Interval::unbounded(),
        };
        let right = Unlisted {
            excluded: BTreeSet::from([text("a")]),
            numbers: Interval::unbounded(),
            length: Interval::unbounded(),
        };
        assert_eq!(
            Values::Unlisted(left).join(&Values::Unlisted(right)),
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::unbounded(),
                length: Interval::unbounded(),
            })
        );
    }

    #[test]
    fn condition_lists_the_admitted_outcomes() {
        assert_eq!(
            Evaluation::condition(Outcomes::TRUE, false),
            Evaluation::new(
                ScalarType::Boolean,
                false,
                false,
                Values::Listed(BTreeSet::from([PgValue::Boolean(true)])),
            )
        );
        assert_eq!(
            Evaluation::condition(Outcomes::FALSE, false),
            Evaluation::new(
                ScalarType::Boolean,
                false,
                false,
                Values::Listed(BTreeSet::from([PgValue::Boolean(false)])),
            )
        );
        assert_eq!(
            Evaluation::condition(Outcomes::ANY, true),
            Evaluation::new(
                ScalarType::Boolean,
                true,
                true,
                Values::Listed(BTreeSet::from([PgValue::Boolean(true), PgValue::Boolean(false),])),
            )
        );
    }

    /// The ranges of a listed set read the members of each kind, whatever
    /// the other members are.
    #[test]
    fn listed_ranges_read_the_members_of_their_kind() {
        let mixed = Evaluation::new(
            ScalarType::Other,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Text("ab".to_string())])),
        );
        assert_eq!(mixed.number_range(), Interval::point(PgNumeric::from(1)));
        assert_eq!(mixed.length_range(), Interval::point(2));
    }

    /// A point interval whose one value is excluded admits nothing.
    #[test]
    fn an_excluded_point_admits_nothing() {
        let five = || PgNumeric::from(5);
        let point = || Interval { low: Bound::Included(five()), high: Bound::Included(five()) };
        let degenerate = || {
            Evaluation::new(
                ScalarType::Numeric,
                false,
                false,
                Values::Unlisted(Unlisted {
                    excluded: BTreeSet::from([PgValue::Numeric(five())]),
                    numbers: point(),
                    length: Interval::unbounded(),
                }),
            )
        };
        assert!(degenerate().is_impossible());
        let mut guarantees = Guarantees::new();
        guarantees.restrict(Subject::Reference("x"), degenerate());
        assert!(guarantees.is_impossible());
    }
}
