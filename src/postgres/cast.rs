//! PostgreSQL's reading of a cast of an evaluated expression.

use alloc::collections::BTreeSet;
use core::ops::Bound;

use sql_values::postgres::{
    CastTarget, CastValue, CharLength, IntegerType, NumericTypmod, PgNumeric, PgValue, cast_value,
};
use sqlparser::ast::{ArrayElemTypeDef, CastKind, CharacterLength, DataType, ExactNumberInfo};

use crate::{
    domain::{Evaluation, Unlisted, Values},
    scalar::{self, ScalarType},
};

/// The result of a cast of `source` to `to` under PostgreSQL's casts and
/// typmods.
pub(super) fn cast(
    source: &Evaluation,
    kind: &CastKind,
    data_type: &DataType,
    to: ScalarType,
    element_type: impl Fn(&DataType) -> ScalarType,
) -> Evaluation {
    if source.is_impossible() {
        return Evaluation::impossible(to).with_error(source.may_raise());
    }
    if !matches!(kind, CastKind::Cast | CastKind::DoubleColon) {
        return Evaluation::unknown(to).with_error(source.may_raise());
    }
    if let DataType::Array(element) = data_type {
        let element = match element {
            // The element's type decides the members' cast; the bracket and
            // size syntax does not.
            ArrayElemTypeDef::AngleBracket(element)
            | ArrayElemTypeDef::SquareBracket(element, _)
            | ArrayElemTypeDef::Parenthesis(element)
            | ArrayElemTypeDef::ParenthesisNotNull(element)
            | ArrayElemTypeDef::Qualified(element, _) => element,
            // An array without a stated element type casts nothing we can
            // fold.
            ArrayElemTypeDef::None => {
                return Evaluation::unknown(ScalarType::Other).with_error(source.may_raise());
            }
        };
        return cast_array(element, element_type(element), source);
    }
    let mut values = BTreeSet::new();
    let mut unknown = false;
    // A typed `NULL` still needs a cast between the two types.
    let mut fold_error = source.may_raise() || (source.is_always_null() && cast_raises(source, to));
    match source.values() {
        Values::Listed(listed) => {
            for datum in listed {
                match scalar::cast(datum, source.ty(), to) {
                    scalar::Cast::Folded(folded) => {
                        values.insert(folded);
                    }
                    scalar::Cast::Raises => fold_error = true,
                    scalar::Cast::Unknown => {
                        unknown = true;
                        fold_error |= cast_raises(source, to);
                    }
                }
            }
        }
        Values::Unlisted(_) => {
            unknown = true;
            fold_error |= cast_raises(source, to);
        }
    }
    let values = if unknown { Values::Unlisted(Unlisted::any()) } else { Values::Listed(values) };
    let mut evaluation = Evaluation::new(to, source.may_be_null(), fold_error, values);
    if to.is_text()
        && let Some(limit) = varchar_limit(data_type)
    {
        evaluation = if CharLength::new(limit).is_some() {
            truncate_text(&evaluation, limit)
        } else {
            // PostgreSQL refuses the type itself.
            Evaluation::raising(to)
        };
    }
    if let Some((precision, scale)) = numeric_modifier(data_type) {
        let typmod = u16::try_from(precision)
            .ok()
            .zip(i16::try_from(scale).ok())
            .and_then(|(precision, scale)| NumericTypmod::new(precision, scale));
        evaluation = match typmod {
            Some(typmod) => {
                let raise = numeric_typmod_raises(source, typmod) || evaluation.may_raise();
                round_numeric(evaluation, typmod).with_error(raise)
            }
            // PostgreSQL refuses the type itself.
            None => Evaluation::raising(to),
        };
    }
    evaluation
}

/// The fold of a cast of a constant array, applying the element cast and
/// its `varchar` bound to every member, unknown when a member's cast is
/// undecided or the array holds no constant members.
fn cast_array(element: &DataType, ty: ScalarType, source: &Evaluation) -> Evaluation {
    let limit = varchar_limit(element);
    if limit.is_some_and(|limit| CharLength::new(limit).is_none()) {
        // PostgreSQL refuses the element type itself.
        return Evaluation::raising(ScalarType::Other);
    }
    // Only an `ARRAY[...]` constructor lists its members, untyped. Any other
    // source is text read as an array literal, which may be malformed, or an
    // array whose members the cast may refuse.
    if source.ty() != ScalarType::Unknown {
        let error = source.may_raise() || !source.is_always_null();
        return Evaluation::new(
            ScalarType::Other,
            source.may_be_null(),
            error,
            Values::Unlisted(Unlisted::any()),
        );
    }
    let Values::Listed(listed) = source.values() else {
        return Evaluation::unknown(ScalarType::Other).with_error(source.may_raise());
    };
    let mut values = BTreeSet::new();
    let mut error = source.may_raise();
    for datum in listed {
        match scalar::cast(datum, source.ty(), ty) {
            scalar::Cast::Folded(folded) => {
                values.insert(match limit {
                    Some(limit) => clip(&folded, limit),
                    None => folded,
                });
            }
            scalar::Cast::Raises => error = true,
            scalar::Cast::Unknown => {
                return Evaluation::unknown(ScalarType::Other).with_error(error);
            }
        }
    }
    Evaluation::new(ScalarType::Other, source.may_be_null(), error, Values::Listed(values))
}

/// The maximum length `varchar` typmods state, when they name one.
fn varchar_limit(data_type: &DataType) -> Option<usize> {
    let (DataType::Varchar(length)
    | DataType::CharacterVarying(length)
    | DataType::CharVarying(length)) = data_type
    else {
        return None;
    };
    match length {
        Some(CharacterLength::IntegerLength { length, .. }) => usize::try_from(*length).ok(),
        _ => None,
    }
}

/// The precision and scale of a `numeric(p, s)` or `numeric(p)` type, the
/// scale of `numeric(p)` being zero.
fn numeric_modifier(data_type: &DataType) -> Option<(u64, i64)> {
    match data_type {
        DataType::Numeric(ExactNumberInfo::PrecisionAndScale(precision, scale))
        | DataType::Decimal(ExactNumberInfo::PrecisionAndScale(precision, scale))
        | DataType::Dec(ExactNumberInfo::PrecisionAndScale(precision, scale)) => {
            Some((*precision, *scale))
        }
        DataType::Numeric(ExactNumberInfo::Precision(precision))
        | DataType::Decimal(ExactNumberInfo::Precision(precision))
        | DataType::Dec(ExactNumberInfo::Precision(precision)) => Some((*precision, 0)),
        _ => None,
    }
}

/// The value cast to `numeric` under `typmod`, `None` when the cast raises.
fn apply_typmod(value: &PgNumeric, typmod: NumericTypmod) -> Option<PgNumeric> {
    match cast_value(CastValue::Numeric(value.clone()), CastTarget::Numeric(Some(typmod))) {
        Ok(CastValue::Numeric(rounded)) => Some(rounded),
        _ => None,
    }
}

/// Whether a cast to `numeric` under `typmod` may raise on the source's
/// values. Rounding to the scale and the precision check are monotone in
/// the magnitude of a finite value, so finite extremes decide it, and an
/// unbounded or infinite extreme may hold an infinity, which raises.
fn numeric_typmod_raises(source: &Evaluation, typmod: NumericTypmod) -> bool {
    let range = source.number_range();
    let fits = |bound: &Bound<PgNumeric>| {
        match bound {
            Bound::Included(value) | Bound::Excluded(value) => {
                !value.is_nan() && !value.is_infinite() && apply_typmod(value, typmod).is_some()
            }
            Bound::Unbounded => false,
        }
    };
    !(fits(&range.low) && fits(&range.high))
}

/// Whether casting a non-constant of the source's type to `to` can raise.
fn cast_raises(source: &Evaluation, to: ScalarType) -> bool {
    let from = source.ty();
    if source.is_always_null() && matches!(from, ScalarType::Unknown | ScalarType::UnknownLiteral) {
        return false;
    }
    match to {
        ScalarType::Text(_) | ScalarType::Unknown | ScalarType::UnknownLiteral => false,
        // PostgreSQL casts only `integer` among the other types to `boolean`.
        ScalarType::Boolean => {
            !matches!(from, ScalarType::Boolean | ScalarType::Integer(IntegerType::Integer))
        }
        ScalarType::Integer(target) => {
            if from.is_number() {
                let (low, high) = target.range().into_inner();
                let range = source.number_range();
                let below = match range.low {
                    core::ops::Bound::Unbounded => true,
                    core::ops::Bound::Included(value) => PgNumeric::from(low) > value,
                    core::ops::Bound::Excluded(value) => PgNumeric::from(low) >= value,
                };
                let above = match range.high {
                    core::ops::Bound::Unbounded => true,
                    core::ops::Bound::Included(value) => PgNumeric::from(high) < value,
                    core::ops::Bound::Excluded(value) => PgNumeric::from(high) <= value,
                };
                below || above
            } else {
                true
            }
        }
        ScalarType::Numeric => {
            if from.is_text() || from == ScalarType::UnknownLiteral {
                true
            } else {
                !from.is_number()
            }
        }
        ScalarType::Date | ScalarType::Timestamp => {
            // An identity cast and `timestamp::date` never raise, while
            // `date::timestamp` refuses a date from 294277-01-01.
            !(from == to || (from == ScalarType::Timestamp && to == ScalarType::Date))
        }
        // An integer always fits a `float`, a `numeric` or `float` may not.
        ScalarType::Float => !matches!(from, ScalarType::Integer(_)),
        ScalarType::Other => true,
    }
}

/// The evaluation with its listed values truncated to at most `limit`
/// characters and its length interval bounded, as an explicit cast to a
/// bounded `varchar` does.
fn truncate_text(evaluation: &Evaluation, limit: usize) -> Evaluation {
    let (null, error, ty) = (evaluation.may_be_null(), evaluation.may_raise(), evaluation.ty());
    match evaluation.values() {
        Values::Listed(listed) => {
            let kept: BTreeSet<PgValue> = listed.iter().map(|value| clip(value, limit)).collect();
            Evaluation::new(ty, null, error, Values::Listed(kept))
        }
        Values::Unlisted(unlisted) => {
            let cap = u64::try_from(limit).unwrap_or(u64::MAX);
            let high = match unlisted.length.high {
                core::ops::Bound::Included(value) if value <= cap => {
                    core::ops::Bound::Included(value)
                }
                core::ops::Bound::Excluded(value) if value <= cap => {
                    core::ops::Bound::Excluded(value)
                }
                _ => core::ops::Bound::Included(cap),
            };
            let mut narrowed = unlisted.clone();
            narrowed.length.high = high;
            Evaluation::new(ty, null, error, Values::Unlisted(narrowed))
        }
    }
}

/// A text value cut to `limit` characters by the `varchar(n)` length
/// coercion, any other value as it is.
fn clip(value: &PgValue, limit: usize) -> PgValue {
    let clipped = match (value, CharLength::new(limit)) {
        (PgValue::Text(text), Some(length)) => {
            cast_value(CastValue::Text(text.clone()), CastTarget::Varchar(Some(length))).ok()
        }
        _ => None,
    };
    match clipped.map(PgValue::try_from) {
        Some(Ok(clipped)) => clipped,
        _ => value.clone(),
    }
}

/// The evaluation with its listed `numeric` values rounded to the typmod
/// scale, raising for a value the typmod's precision cannot hold after
/// rounding.
fn round_numeric(evaluation: Evaluation, typmod: NumericTypmod) -> Evaluation {
    let Values::Listed(values) = evaluation.values() else {
        return evaluation;
    };
    let mut kept = BTreeSet::new();
    let mut error = evaluation.may_raise();
    for value in values {
        let rounded = match value {
            PgValue::Numeric(numeric) => apply_typmod(numeric, typmod),
            _ => None,
        };
        match rounded {
            Some(rounded) => {
                kept.insert(PgValue::Numeric(rounded));
            }
            None => error = true,
        }
    }
    Evaluation::new(evaluation.ty(), evaluation.may_be_null(), error, Values::Listed(kept))
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use core::ops::Bound;

    use sql_values::postgres::{IntegerType, NumericTypmod, PgNumeric, PgValue};
    use sqlparser::ast::{ArrayElemTypeDef, CastKind, CharacterLength, DataType, ExactNumberInfo};

    use crate::{
        domain::{Evaluation, Interval, Unlisted, Values},
        scalar::{ScalarType, TextRules},
    };

    type Element = fn(&DataType) -> ScalarType;
    const ELEMENT: Element = ScalarType::of;

    fn text(value: &str) -> Evaluation {
        Evaluation::known(PgValue::Text(value.to_string()), ScalarType::UnknownLiteral)
    }

    fn num(text: &str) -> Evaluation {
        Evaluation::known(
            PgValue::Numeric(PgNumeric::parse(text).expect("a numeric literal")),
            ScalarType::Numeric,
        )
    }

    /// A nullable, non-raising column of `ty` with unlisted values.
    fn column(ty: ScalarType) -> Evaluation {
        Evaluation::new(ty, true, false, Values::Unlisted(Unlisted::any()))
    }

    /// A `varchar(n)` data type.
    fn varchar(n: u64) -> DataType {
        DataType::Varchar(Some(CharacterLength::IntegerLength { length: n, unit: None }))
    }

    /// A `numeric(p, s)` data type.
    fn numeric(p: u64, s: i64) -> DataType {
        DataType::Numeric(ExactNumberInfo::PrecisionAndScale(p, s))
    }

    /// A Unlisted numeric source whose values lie in `range`.
    fn numeric_in(low: Bound<PgNumeric>, high: Bound<PgNumeric>) -> Evaluation {
        Evaluation::new(
            ScalarType::Numeric,
            true,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval { low, high },
                length: Interval::unbounded(),
            }),
        )
    }

    fn cast(source: &Evaluation, data_type: &DataType, to: ScalarType) -> Evaluation {
        super::cast(source, &CastKind::Cast, data_type, to, ELEMENT)
    }

    /// A cast of an impossible source is impossible.
    #[test]
    fn a_cast_of_an_impossible_source_is_impossible() {
        let source = Evaluation::impossible(ScalarType::Unknown);
        let evaluation = super::cast(
            &source,
            &CastKind::Cast,
            &DataType::Text,
            ScalarType::Text(TextRules::DEFAULT),
            ELEMENT,
        );
        assert!(evaluation.is_impossible());
    }

    /// A non-`CAST`/`::` cast kind is unknown.
    #[test]
    fn a_try_cast_kind_is_unknown() {
        let source = text("5");
        let evaluation = super::cast(
            &source,
            &CastKind::TryCast,
            &DataType::Text,
            ScalarType::Text(TextRules::DEFAULT),
            ELEMENT,
        );
        assert!(evaluation.may_be_null());
    }

    /// A cast to an array of unknown element type is unknown.
    #[test]
    fn a_cast_to_an_untyped_array_is_unknown() {
        let source = text("1");
        let data_type = DataType::Array(ArrayElemTypeDef::None);
        let evaluation = cast(&source, &data_type, ScalarType::Other);
        assert!(evaluation.may_be_null());
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    }

    /// A listed cast that raises on one member carries the error.
    #[test]
    fn a_listed_cast_that_raises_carries_the_error() {
        let source = text("not a boolean");
        let evaluation = cast(&source, &DataType::Boolean, ScalarType::Boolean);
        assert!(evaluation.may_raise());
    }

    /// A listed cast to `numeric` folds the special values `numeric_in` reads.
    #[test]
    fn a_listed_cast_to_a_special_number_folds() {
        let source = text("infinity");
        let evaluation =
            cast(&source, &DataType::Numeric(ExactNumberInfo::None), ScalarType::Numeric);
        assert_eq!(evaluation.constant(), Some(&PgValue::Numeric(PgNumeric::INFINITY)));
    }

    /// A `varchar(n)` cast truncates each listed member to `n` characters.
    #[test]
    fn a_varchar_cast_truncates_listed_members() {
        let source = text("abcdefgh");
        let evaluation = cast(&source, &varchar(3), ScalarType::Text(TextRules::DEFAULT));
        assert_eq!(evaluation.constant(), Some(&PgValue::Text("abc".to_string())));
    }

    /// PostgreSQL refuses a `varchar(0)` type, so every cast to it raises,
    /// an array element cast included.
    #[test]
    fn a_varchar_zero_cast_raises() {
        let text_type = ScalarType::Text(TextRules::DEFAULT);
        assert!(cast(&text("abc"), &varchar(0), text_type).always_raises());
        assert!(cast(&column(text_type), &varchar(0), text_type).always_raises());
        let array = DataType::Array(ArrayElemTypeDef::SquareBracket(Box::new(varchar(0)), None));
        assert!(cast(&text("ab"), &array, ScalarType::Other).always_raises());
    }

    /// A `varchar` bound does not shrink a listed non-text value.
    #[test]
    fn a_varchar_bound_keeps_a_listed_non_text_value() {
        let source =
            Evaluation::known(PgValue::Integer(5), ScalarType::Integer(IntegerType::Integer));
        let evaluation = super::truncate_text(&source, 2);
        assert_eq!(evaluation.constant(), Some(&PgValue::Integer(5)));
    }

    /// A `varchar` bound keeps a length bound already within the cap.
    #[test]
    fn a_varchar_bound_keeps_an_in_range_length_bound() {
        let source = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::unbounded(),
                length: Interval { low: Bound::Unbounded, high: Bound::Included(5) },
            }),
        );
        let evaluation = super::truncate_text(&source, 10);
        assert_eq!(evaluation.length_range().high, Bound::Included(5));
    }

    /// A `varchar` bound keeps an excluded length bound within the cap.
    #[test]
    fn a_varchar_bound_keeps_an_in_range_excluded_bound() {
        let source = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::unbounded(),
                length: Interval { low: Bound::Unbounded, high: Bound::Excluded(5) },
            }),
        );
        let evaluation = super::truncate_text(&source, 10);
        assert_eq!(evaluation.length_range().high, Bound::Excluded(5));
    }

    /// `varchar_limit` reads a `varchar(n)`, `None` otherwise.
    #[test]
    fn varchar_limit_reads_varchar() {
        assert_eq!(super::varchar_limit(&DataType::String(Some(5))), None);
        assert_eq!(super::varchar_limit(&varchar(5)), Some(5));
        assert_eq!(super::varchar_limit(&DataType::Text), None);
        assert_eq!(super::varchar_limit(&DataType::Varchar(None)), None);
    }

    /// A `numeric(p, s)` cast rounds each listed member to `s`.
    #[test]
    fn a_numeric_cast_rounds_to_the_scale() {
        let source = num("12.349");
        let evaluation = cast(&source, &numeric(10, 2), ScalarType::Numeric);
        assert_eq!(
            evaluation.constant(),
            Some(&PgValue::Numeric(PgNumeric::parse("12.35").expect("rounded")))
        );
    }

    /// The modifier `numeric(precision, scale)`.
    fn typmod(precision: u16, scale: i16) -> NumericTypmod {
        NumericTypmod::new(precision, scale).expect("a valid modifier")
    }

    /// A `numeric(10, 0)` cast keeps an integer as is.
    #[test]
    fn a_numeric_zero_scale_cast_keeps_the_value() {
        let evaluation = super::round_numeric(num("12"), typmod(10, 0));
        assert_eq!(evaluation.constant(), Some(&PgValue::Numeric(PgNumeric::from(12))));
    }

    /// A `numeric(p, s)` cast of a non-numeric value raises.
    #[test]
    fn a_numeric_cast_of_a_non_numeric_value_raises() {
        let source =
            Evaluation::known(PgValue::Integer(5), ScalarType::Integer(IntegerType::Integer));
        assert!(super::round_numeric(source, typmod(10, 2)).may_raise());
    }

    /// A `numeric(p, s)` cast past the precision raises, judged after the
    /// rounding, as `9.995::numeric(3, 2)` rounds to `10.00`.
    #[test]
    fn a_numeric_cast_past_the_precision_raises() {
        assert!(super::round_numeric(num("123.45"), typmod(2, 1)).may_raise());
        let rounded_up = cast(&num("9.995"), &numeric(3, 2), ScalarType::Numeric);
        assert!(rounded_up.may_raise());
        assert!(rounded_up.constant().is_none());
    }

    /// `numeric(p)` rounds to a scale of zero.
    #[test]
    fn a_numeric_cast_with_a_precision_alone_rounds_to_integers() {
        let data_type = DataType::Numeric(ExactNumberInfo::Precision(5));
        let evaluation = cast(&num("1.5"), &data_type, ScalarType::Numeric);
        assert_eq!(evaluation.constant(), Some(&PgValue::Numeric(PgNumeric::from(2))));
        assert!(!evaluation.may_raise());
    }

    /// A `numeric` typmod on a non-listed source is left as is.
    #[test]
    fn a_numeric_rounding_of_an_unlisted_source_is_left_alone() {
        let evaluation = super::round_numeric(column(ScalarType::Numeric), typmod(10, 2));
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    }

    /// `numeric_typmod_raises` is true for a value the precision cannot hold,
    /// an unbounded range and an infinite bound, and false when a negative
    /// scale rounds the value into range.
    #[test]
    fn numeric_typmod_raises_on_the_unsound_inputs() {
        let one = num("1");
        assert!(super::numeric_typmod_raises(&one, typmod(2, 5)));
        assert!(!super::numeric_typmod_raises(&one, typmod(10, -1)));
        assert!(super::numeric_typmod_raises(&column(ScalarType::Numeric), typmod(10, 2)));
        let infinite =
            numeric_in(Bound::Included(PgNumeric::from(1)), Bound::Included(PgNumeric::INFINITY));
        assert!(super::numeric_typmod_raises(&infinite, typmod(10, 2)));
    }

    /// A modifier PostgreSQL refuses, as `numeric(0, 0)`, makes the cast raise.
    #[test]
    fn a_refused_numeric_modifier_raises() {
        assert!(cast(&num("1"), &numeric(0, 0), ScalarType::Numeric).always_raises());
    }

    /// `numeric_typmod_raises` is false when the range fits the integer digits.
    #[test]
    fn numeric_typmod_raises_is_false_in_range() {
        let source =
            numeric_in(Bound::Included(PgNumeric::from(1)), Bound::Included(PgNumeric::from(9)));
        assert!(!super::numeric_typmod_raises(&source, typmod(5, 2)));
    }

    /// `cast_raises` is false for an always-`NULL` source.
    #[test]
    fn cast_raises_is_false_for_an_always_null_source() {
        let source = Evaluation::null(ScalarType::Unknown);
        assert!(!super::cast_raises(&source, ScalarType::Boolean));
        // A typed `NULL` still needs a cast between the two types.
        let bigint = Evaluation::null(ScalarType::Integer(IntegerType::Bigint));
        assert!(super::cast_raises(&bigint, ScalarType::Boolean));
    }

    /// Only an `integer` or a `boolean` source never raises casting to boolean.
    #[test]
    fn cast_raises_from_text_to_boolean() {
        assert!(super::cast_raises(
            &column(ScalarType::Text(TextRules::DEFAULT)),
            ScalarType::Boolean
        ));
        assert!(super::cast_raises(&column(ScalarType::UnknownLiteral), ScalarType::Boolean));
        for raising in
            [ScalarType::Numeric, ScalarType::Integer(IntegerType::Bigint), ScalarType::Date]
        {
            assert!(super::cast_raises(&column(raising), ScalarType::Boolean));
        }
        for clear in [ScalarType::Integer(IntegerType::Integer), ScalarType::Boolean] {
            assert!(!super::cast_raises(&column(clear), ScalarType::Boolean));
        }
    }

    /// An unbounded numeric source may raise casting to an integer.
    #[test]
    fn cast_raises_from_unbounded_numeric_to_integer() {
        let source = numeric_in(Bound::Unbounded, Bound::Unbounded);
        assert!(super::cast_raises(&source, ScalarType::Integer(IntegerType::Integer)));
    }

    /// A bounded numeric source's included and excluded bounds are checked.
    #[test]
    fn cast_raises_checks_the_included_and_excluded_bounds() {
        let low = PgNumeric::from(1);
        let high = PgNumeric::from(2);
        let included = numeric_in(Bound::Included(low.clone()), Bound::Included(high.clone()));
        let excluded = numeric_in(Bound::Excluded(low), Bound::Excluded(high));
        let target = ScalarType::Integer(IntegerType::Bigint);
        assert!(!super::cast_raises(&included, target));
        assert!(!super::cast_raises(&excluded, target));
    }

    /// A non-number source may raise casting to an integer.
    #[test]
    fn cast_raises_from_non_number_to_integer() {
        assert!(super::cast_raises(
            &column(ScalarType::Date),
            ScalarType::Integer(IntegerType::Integer)
        ));
    }

    /// A text or unknown source may raise casting to numeric.
    #[test]
    fn cast_raises_from_text_to_numeric() {
        assert!(super::cast_raises(
            &column(ScalarType::Text(TextRules::DEFAULT)),
            ScalarType::Numeric
        ));
        assert!(!super::cast_raises(&column(ScalarType::Numeric), ScalarType::Numeric));
    }

    /// Identity casts and `timestamp::date` do not raise, while a date
    /// column may hold a date from 294277-01-01, which `date::timestamp`
    /// refuses.
    #[test]
    fn cast_raises_between_temporal_types() {
        assert!(super::cast_raises(&column(ScalarType::Date), ScalarType::Timestamp));
        assert!(!super::cast_raises(&column(ScalarType::Date), ScalarType::Date));
        assert!(!super::cast_raises(&column(ScalarType::Timestamp), ScalarType::Date));
        assert!(!super::cast_raises(&column(ScalarType::Timestamp), ScalarType::Timestamp));
    }

    /// Only an integer source never raises casting to float.
    #[test]
    fn cast_raises_to_float() {
        assert!(super::cast_raises(
            &column(ScalarType::Text(TextRules::DEFAULT)),
            ScalarType::Float
        ));
        assert!(super::cast_raises(&column(ScalarType::Numeric), ScalarType::Float));
        assert!(super::cast_raises(&column(ScalarType::Float), ScalarType::Float));
        assert!(!super::cast_raises(
            &column(ScalarType::Integer(IntegerType::Bigint)),
            ScalarType::Float
        ));
    }

    /// The evaluation of an `ARRAY[...]` constructor of the text members.
    fn array(members: &[&str]) -> Evaluation {
        let members = members.iter().map(|member| PgValue::Text((*member).to_string())).collect();
        Evaluation::new(ScalarType::Unknown, false, false, Values::Listed(members))
    }

    /// A cast to an array of a non-listed source is unknown.
    #[test]
    fn an_array_cast_of_an_unlisted_source_is_unknown() {
        let source = column(ScalarType::Unknown);
        let elem = DataType::Int(None);
        let evaluation =
            super::cast_array(&elem, ScalarType::Integer(IntegerType::Integer), &source);
        assert!(evaluation.may_be_null());
    }

    /// An array cast that raises on one member carries the error.
    #[test]
    fn an_array_cast_that_raises_carries_the_error() {
        let source = array(&["not an integer"]);
        let elem = DataType::Int(None);
        let evaluation =
            super::cast_array(&elem, ScalarType::Integer(IntegerType::Integer), &source);
        assert!(evaluation.may_raise());
    }

    /// Text cast to an array is an array literal, which may be malformed,
    /// and only a `NULL` never raises.
    #[test]
    fn an_array_cast_of_text_may_raise() {
        let int = ScalarType::Integer(IntegerType::Integer);
        let evaluation = super::cast_array(&DataType::Int(None), int, &text("{1,2}"));
        assert!(evaluation.may_raise() && !evaluation.always_raises());
        assert!(!evaluation.may_be_null());
        let null = Evaluation::null(ScalarType::Text(TextRules::DEFAULT));
        let evaluation = super::cast_array(&DataType::Int(None), int, &null);
        assert!(!evaluation.may_raise());
    }

    /// A cast that is undecided on a member, as a date whose reading the
    /// `DateStyle` field order decides, is unknown, on its own and in an
    /// array.
    #[test]
    fn a_cast_that_is_undecided_is_unknown() {
        let evaluation = cast(&text("01/02/2026"), &DataType::Date, ScalarType::Date);
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
        let evaluation =
            super::cast_array(&DataType::Date, ScalarType::Date, &array(&["01/02/2026"]));
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    }

    /// An array cast folds each member and applies a `varchar` bound.
    #[test]
    fn an_array_cast_folds_its_members() {
        let elem = varchar(1);
        let evaluation =
            super::cast_array(&elem, ScalarType::Text(TextRules::DEFAULT), &array(&["abcd", "xy"]));
        assert!(evaluation.may_be(&PgValue::Text("a".to_string())));
        assert!(evaluation.may_be(&PgValue::Text("x".to_string())));
        assert!(!evaluation.may_be(&PgValue::Text("abcd".to_string())));
    }
}
