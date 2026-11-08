//! The numeric interval helpers the conservative comparison and arithmetic
//! readings share.

use alloc::collections::BTreeSet;
use core::ops::Bound;

use sql_values::{PgNumeric, PgValue};

use crate::domain::{Interval, Values};

#[must_use]
pub(crate) fn intervals_disjoint(a: &Interval<PgNumeric>, b: &Interval<PgNumeric>) -> bool {
    a.meet(b).is_empty()
}

/// The smallest and the largest of the listed numeric datums.
fn listed_numbers(values: &BTreeSet<PgValue>) -> (Option<PgNumeric>, Option<PgNumeric>) {
    // The set iterates in ascending datum order, so the numeric value it
    // admits first is its minimum and the one it admits last its maximum.
    let mut low = None;
    let mut high = None;
    for value in values {
        if let Some(number) = value.as_numeric() {
            if low.is_none() {
                low = Some(number.clone());
            }
            high = Some(number);
        }
    }
    (low, high)
}

/// The numeric interval a value set lies in, unbounded when unknown.
#[must_use]
pub(crate) fn number_bounds(values: &Values) -> Interval<PgNumeric> {
    match values {
        Values::Listed(listed) => {
            let (low, high) = listed_numbers(listed);
            Interval {
                low: low.map_or(Bound::Unbounded, Bound::Included),
                high: high.map_or(Bound::Unbounded, Bound::Included),
            }
        }
        Values::Unlisted(unlisted) => unlisted.numbers.clone(),
    }
}

/// The character-length interval a value set's text lies in, unbounded when
/// unknown or when it lists no text.
#[must_use]
pub(crate) fn length_bounds(values: &Values) -> Interval<u64> {
    match values {
        Values::Listed(listed) => {
            let mut lengths = listed.iter().filter_map(crate::scalar::text_length);
            let Some(first) = lengths.next() else {
                return Interval::unbounded();
            };
            let (low, high) = lengths
                .fold((first, first), |(low, high), length| (low.min(length), high.max(length)));
            Interval { low: Bound::Included(low), high: Bound::Included(high) }
        }
        Values::Unlisted(unlisted) => unlisted.length.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A non-numeric member does not move the numeric bounds.
    #[test]
    fn mixed_members_take_the_numeric_bounds() {
        let listed = BTreeSet::from([
            PgValue::Text("a".to_string()),
            PgValue::Numeric(PgNumeric::from(1)),
            PgValue::Numeric(PgNumeric::from(2)),
        ]);
        let bounds = number_bounds(&Values::Listed(listed));
        assert_eq!(bounds.low, Bound::Included(PgNumeric::from(1)));
        assert_eq!(bounds.high, Bound::Included(PgNumeric::from(2)));
    }

    /// A listed set holding no text bounds no length.
    #[test]
    fn a_set_without_text_bounds_no_length() {
        let numbers = Values::Listed(BTreeSet::from([PgValue::Integer(1)]));
        assert_eq!(super::length_bounds(&numbers), Interval::unbounded());
    }
}
