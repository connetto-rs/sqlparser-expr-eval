//! The scalar types the evaluator reads an expression as, and the casts and
//! literal typing it folds [`PgValue`]s with.

use sql_values::{
    CastError, CastTarget, CastValue, IntegerType, PgNumeric, PgValue, cast_value, number_constant,
};
use sqlparser::ast::{DataType, TimezoneInfo};

/// What a text collation lets the evaluator decide about text values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextRules {
    /// Whether `=` on text is decided by identity, as PostgreSQL's
    /// deterministic collations (and the database default) allow.
    pub identity_equality: bool,
    /// Whether ordering is by byte, as the `"C"` and `"POSIX"` collations
    /// allow.
    pub byte_order: bool,
}

impl TextRules {
    /// The database's default collation, which PostgreSQL never lets be
    /// nondeterministic: equality is by identity, ordering is not by byte.
    pub const DEFAULT: TextRules = TextRules { identity_equality: true, byte_order: false };
    /// The `"C"` and `"POSIX"` collations, which order by byte.
    pub const BYTEWISE: TextRules = TextRules { identity_equality: true, byte_order: true };
    /// A collation the evaluator cannot reason about.
    pub const UNKNOWN: TextRules = TextRules { identity_equality: false, byte_order: false };
}

/// The scalar type a value or an expression's result carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// The type is not known, as for an untyped `NULL`.
    Unknown,
    /// A quoted string literal whose type PostgreSQL resolves from its use.
    UnknownLiteral,
    /// `boolean`.
    Boolean,
    /// An integer type.
    Integer(IntegerType),
    /// `numeric`.
    Numeric,
    /// `real` or `double precision`, whose output depends on
    /// `extra_float_digits` and whose values are never folded.
    Float,
    /// A text type, with the collation it carries.
    Text(TextRules),
    /// `date`.
    Date,
    /// `timestamp` without time zone.
    Timestamp,
    /// Any other known type whose values are not modelled.
    Other,
}

impl ScalarType {
    /// The scalar type a PostgreSQL `DataType` spells.
    #[must_use]
    pub fn of(data_type: &DataType) -> Self {
        match data_type {
            DataType::Int2(_) | DataType::SmallInt(_) => ScalarType::Integer(IntegerType::Smallint),
            DataType::Int(_) | DataType::Int4(_) | DataType::Integer(_) => {
                ScalarType::Integer(IntegerType::Integer)
            }
            DataType::Int8(_) | DataType::BigInt(_) => ScalarType::Integer(IntegerType::Bigint),
            DataType::Numeric(_) | DataType::Decimal(_) | DataType::Dec(_) => ScalarType::Numeric,
            DataType::Float(_)
            | DataType::Float4
            | DataType::Float8
            | DataType::Real
            | DataType::Double(_)
            | DataType::DoublePrecision => ScalarType::Float,
            DataType::Text
            | DataType::Varchar(_)
            | DataType::CharacterVarying(_)
            | DataType::CharVarying(_) => ScalarType::Text(TextRules::DEFAULT),
            DataType::Boolean | DataType::Bool => ScalarType::Boolean,
            DataType::Date => ScalarType::Date,
            DataType::Timestamp(_, TimezoneInfo::None | TimezoneInfo::WithoutTimeZone) => {
                ScalarType::Timestamp
            }
            _ => ScalarType::Other,
        }
    }

    /// Whether the type is a number, an integer or `numeric`.
    #[must_use]
    pub fn is_number(self) -> bool {
        matches!(self, ScalarType::Integer(_) | ScalarType::Numeric)
    }

    /// Whether the type is a text type.
    #[must_use]
    pub fn is_text(self) -> bool {
        matches!(self, ScalarType::Text(_))
    }

    /// The inclusive `i64` bounds the type admits, when it is an integer type.
    #[must_use]
    pub fn integer_bounds(self) -> Option<(i64, i64)> {
        match self {
            ScalarType::Integer(integer) => Some(integer.range().into_inner()),
            _ => None,
        }
    }
}

/// The character length of a text value.
pub(crate) fn text_length(value: &PgValue) -> Option<u64> {
    // the character count always fits a `u64`
    value.as_text().map(|text| text.chars().count() as u64)
}

/// Types the literal `text` as PostgreSQL's `make_const` does, applying a
/// leading minus when `negated`: a digits-only literal is `int4` when it
/// fits, `int8` when it fits, and `numeric` otherwise, while a literal with a
/// point or an exponent is `numeric`.
pub(crate) fn number_literal(text: &str, negated: bool) -> Option<(PgValue, ScalarType)> {
    let constant =
        if negated { number_constant(&alloc::format!("-{text}")) } else { number_constant(text) };
    let constant = constant.ok()?;
    let ty = match constant {
        CastValue::Integer(_) => ScalarType::Integer(IntegerType::Integer),
        CastValue::Bigint(_) => ScalarType::Integer(IntegerType::Bigint),
        _ => ScalarType::Numeric,
    };
    let Cast::Folded(datum) = folded(constant) else { return None };
    Some((datum, ty))
}

/// A cast the evaluator can fold, raise, or leave undecided.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Cast {
    /// The cast folds to this value.
    Folded(PgValue),
    /// The cast raises an error.
    Raises,
    /// The cast's result is not decided by the value.
    Unknown,
}

/// Casts `datum` of type `from` to type `to`, following PostgreSQL's casts.
/// A date or a timestamp prints through `DateStyle`, so its cast to text is
/// left undecided, as is a cast to a type the evaluator does not model.
pub(crate) fn cast(datum: &PgValue, from: ScalarType, to: ScalarType) -> Cast {
    if from == to {
        return Cast::Folded(datum.clone());
    }
    let target = match to {
        ScalarType::Boolean => CastTarget::Boolean,
        ScalarType::Integer(IntegerType::Smallint) => CastTarget::Smallint,
        ScalarType::Integer(IntegerType::Integer) => CastTarget::Integer,
        ScalarType::Integer(IntegerType::Bigint) => CastTarget::Bigint,
        ScalarType::Numeric => CastTarget::Numeric(None),
        // A date or a timestamp prints through `DateStyle`.
        ScalarType::Text(_) if matches!(datum, PgValue::Date(_) | PgValue::Timestamp(_)) => {
            return Cast::Unknown;
        }
        ScalarType::Text(_) => CastTarget::Text,
        ScalarType::Date => CastTarget::Date,
        ScalarType::Timestamp => CastTarget::Timestamp(None),
        ScalarType::Unknown
        | ScalarType::UnknownLiteral
        | ScalarType::Float
        | ScalarType::Other => {
            return Cast::Unknown;
        }
    };
    let Some(source) = cast_source(datum, from) else { return Cast::Unknown };
    match cast_value(source, target) {
        Ok(value) => folded(value),
        Err(CastError::Unsupported) => Cast::Unknown,
        Err(_) => Cast::Raises,
    }
}

/// The `sql-values` value of `datum` read as its type `from`, an integer at
/// the width its type names, `bigint` when the type names none, and `None`
/// for an integer past that width, which no value of the type holds.
fn cast_source(datum: &PgValue, from: ScalarType) -> Option<CastValue> {
    let width = match (datum, from) {
        (PgValue::Integer(value), ScalarType::Numeric) => {
            return Some(CastValue::Numeric(PgNumeric::from(*value)));
        }
        (_, ScalarType::Integer(width)) => width,
        _ => IntegerType::Bigint,
    };
    datum.clone().into_cast_value(width).ok()
}

/// The value a cast produced, a `char(n)` as the `text` it casts to, and
/// undecided for a `timestamptz`, whose value the evaluator does not model.
fn folded(value: CastValue) -> Cast {
    match PgValue::try_from(value) {
        Ok(value) => Cast::Folded(value),
        Err(CastValue::Bpchar(text)) => Cast::Folded(PgValue::Text(text.as_text().into())),
        Err(_) => Cast::Unknown,
    }
}

/// The value rounded half away from zero, as `numeric::bigint` rounds it,
/// `None` for `NaN`, an infinity or a value past `bigint`.
pub(crate) fn rounded_i64(value: &PgNumeric) -> Option<i64> {
    match cast_value(CastValue::Numeric(value.clone()), CastTarget::Bigint) {
        Ok(CastValue::Bigint(rounded)) => Some(rounded),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use sql_values::{PgBpchar, PgDate, PgTimestamp, PgTimestampTz};

    use super::*;

    fn text(value: &str) -> PgValue {
        PgValue::Text(String::from(value))
    }

    fn numeric(value: &str) -> PgNumeric {
        PgNumeric::parse(value).unwrap()
    }

    fn date(year: i32, month: u8, day: u8) -> PgDate {
        PgDate::from_ymd(year, month, day).unwrap()
    }

    fn midnight(year: i32, month: u8, day: u8) -> PgTimestamp {
        date(year, month, day).to_timestamp().unwrap()
    }

    #[test]
    fn number_literals_are_typed_as_make_const_types_them() {
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let int8 = ScalarType::Integer(IntegerType::Bigint);
        assert_eq!(
            number_literal("2147483647", false),
            Some((PgValue::Integer(2_147_483_647), int4))
        );
        assert_eq!(
            number_literal("2147483648", false),
            Some((PgValue::Integer(2_147_483_648), int8))
        );
        // The negation folds into the constant before it is typed.
        assert_eq!(
            number_literal("2147483648", true),
            Some((PgValue::Integer(-2_147_483_648), int4))
        );
        assert_eq!(number_literal("0xff", false), Some((PgValue::Integer(255), int4)));
        let (datum, ty) = number_literal("9223372036854775808", false).unwrap();
        assert!(matches!(datum, PgValue::Numeric(_)));
        assert_eq!(ty, ScalarType::Numeric);
        assert_eq!(
            number_literal("1e3", false),
            Some((PgValue::Numeric(numeric("1000")), ScalarType::Numeric))
        );
        assert_eq!(number_literal("1.5.3", false), None);
        assert_eq!(number_literal("0x", false), None);
    }

    #[test]
    fn casts_read_the_source_at_its_declared_type() {
        let int2 = ScalarType::Integer(IntegerType::Smallint);
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let int8 = ScalarType::Integer(IntegerType::Bigint);
        let text_type = ScalarType::Text(TextRules::DEFAULT);
        // An identity cast keeps the value, and a narrowing cast checks it.
        assert_eq!(cast(&PgValue::Integer(5), int4, int4), Cast::Folded(PgValue::Integer(5)));
        assert_eq!(cast(&PgValue::Integer(5), int4, int2), Cast::Folded(PgValue::Integer(5)));
        // `int4` alone casts to boolean.
        assert_eq!(
            cast(&PgValue::Integer(1), int4, ScalarType::Boolean),
            Cast::Folded(PgValue::Boolean(true))
        );
        assert_eq!(cast(&PgValue::Integer(1), int2, ScalarType::Boolean), Cast::Raises);
        assert_eq!(cast(&PgValue::Integer(1), int8, ScalarType::Boolean), Cast::Raises);
        assert_eq!(
            cast(&PgValue::Boolean(true), ScalarType::Boolean, int4),
            Cast::Folded(PgValue::Integer(1))
        );
        assert_eq!(cast(&PgValue::Boolean(true), ScalarType::Boolean, int2), Cast::Raises);
        assert_eq!(cast(&PgValue::Integer(32768), int4, int2), Cast::Raises);
        // An integer under a `numeric` type is a `numeric`, which no boolean
        // cast takes.
        assert_eq!(
            cast(&PgValue::Integer(1), ScalarType::Numeric, ScalarType::Boolean),
            Cast::Raises
        );
        assert_eq!(
            cast(&PgValue::Numeric(numeric("2.5")), ScalarType::Numeric, int4),
            Cast::Folded(PgValue::Integer(3))
        );
        assert_eq!(
            cast(&PgValue::Integer(42), int4, ScalarType::Numeric),
            Cast::Folded(PgValue::Numeric(numeric("42")))
        );
        assert_eq!(
            cast(&PgValue::Numeric(PgNumeric::NAN), ScalarType::Numeric, int4),
            Cast::Raises
        );
        assert_eq!(cast(&text(" 42 "), text_type, int4), Cast::Folded(PgValue::Integer(42)));
        assert_eq!(cast(&text("4.2"), text_type, int4), Cast::Raises);
        assert_eq!(
            cast(&text("of"), text_type, ScalarType::Boolean),
            Cast::Folded(PgValue::Boolean(false))
        );
        assert_eq!(
            cast(&text("-inf"), text_type, ScalarType::Numeric),
            Cast::Folded(PgValue::Numeric(PgNumeric::NEG_INFINITY))
        );
    }

    #[test]
    fn temporal_casts_fold_unless_a_setting_decides_them() {
        let text_type = ScalarType::Text(TextRules::DEFAULT);
        assert_eq!(
            cast(&text("Jan 29 2024"), text_type, ScalarType::Date),
            Cast::Folded(PgValue::Date(date(2024, 1, 29)))
        );
        assert_eq!(cast(&text("2023-02-29"), text_type, ScalarType::Date), Cast::Raises);
        // The `DateStyle` field order decides `01/02/2024`, and the current
        // time decides `today`.
        assert_eq!(cast(&text("01/02/2024"), text_type, ScalarType::Date), Cast::Unknown);
        assert_eq!(cast(&text("today"), text_type, ScalarType::Date), Cast::Unknown);
        assert_eq!(
            cast(&PgValue::Date(date(2026, 1, 1)), ScalarType::Date, ScalarType::Timestamp),
            Cast::Folded(PgValue::Timestamp(midnight(2026, 1, 1)))
        );
        assert_eq!(
            cast(&PgValue::Date(date(294_277, 1, 1)), ScalarType::Date, ScalarType::Timestamp),
            Cast::Raises
        );
        assert_eq!(
            cast(
                &PgValue::Timestamp(midnight(2026, 1, 1)),
                ScalarType::Timestamp,
                ScalarType::Date
            ),
            Cast::Folded(PgValue::Date(date(2026, 1, 1)))
        );
        assert_eq!(
            cast(&PgValue::Date(date(2026, 1, 1)), ScalarType::Date, text_type),
            Cast::Unknown
        );
        assert_eq!(
            cast(
                &PgValue::Date(date(2026, 1, 1)),
                ScalarType::Date,
                ScalarType::Integer(IntegerType::Integer)
            ),
            Cast::Raises
        );
        assert_eq!(
            cast(
                &PgValue::Integer(1),
                ScalarType::Integer(IntegerType::Integer),
                ScalarType::Float
            ),
            Cast::Unknown
        );
    }

    #[test]
    fn casts_to_text_print_through_the_output_function() {
        let text_type = ScalarType::Text(TextRules::DEFAULT);
        let int8 = ScalarType::Integer(IntegerType::Bigint);
        assert_eq!(
            cast(&PgValue::Boolean(true), ScalarType::Boolean, text_type),
            Cast::Folded(text("true"))
        );
        assert_eq!(cast(&PgValue::Integer(42), int8, text_type), Cast::Folded(text("42")));
        assert_eq!(
            cast(&PgValue::Numeric(numeric("1.50")), ScalarType::Numeric, text_type),
            Cast::Folded(text("1.50"))
        );
        assert_eq!(cast(&text("42"), text_type, int8), Cast::Folded(PgValue::Integer(42)));
    }

    #[test]
    fn numerics_convert_to_i64_as_bigint_does() {
        assert_eq!(rounded_i64(&numeric("2.5")), Some(3));
        assert_eq!(rounded_i64(&numeric("-2.5")), Some(-3));
        assert_eq!(rounded_i64(&PgNumeric::NAN), None);
        assert_eq!(rounded_i64(&numeric("9223372036854775808")), None);
    }

    #[test]
    fn scalar_type_of_maps_every_datatype() {
        use sqlparser::ast::ExactNumberInfo;

        assert_eq!(
            ScalarType::of(&DataType::SmallInt(None)),
            ScalarType::Integer(IntegerType::Smallint)
        );
        assert_eq!(
            ScalarType::of(&DataType::Int2(None)),
            ScalarType::Integer(IntegerType::Smallint)
        );
        assert_eq!(ScalarType::of(&DataType::Int(None)), ScalarType::Integer(IntegerType::Integer));
        assert_eq!(
            ScalarType::of(&DataType::Integer(None)),
            ScalarType::Integer(IntegerType::Integer)
        );
        assert_eq!(ScalarType::of(&DataType::Int8(None)), ScalarType::Integer(IntegerType::Bigint));
        assert_eq!(
            ScalarType::of(&DataType::BigInt(None)),
            ScalarType::Integer(IntegerType::Bigint)
        );
        assert_eq!(ScalarType::of(&DataType::Numeric(ExactNumberInfo::None)), ScalarType::Numeric);
        assert_eq!(
            ScalarType::of(&DataType::Decimal(ExactNumberInfo::Precision(10))),
            ScalarType::Numeric
        );
        assert_eq!(
            ScalarType::of(&DataType::Dec(ExactNumberInfo::PrecisionAndScale(10, 2))),
            ScalarType::Numeric
        );
        assert_eq!(ScalarType::of(&DataType::Float(ExactNumberInfo::None)), ScalarType::Float);
        assert_eq!(ScalarType::of(&DataType::Float4), ScalarType::Float);
        assert_eq!(ScalarType::of(&DataType::Float8), ScalarType::Float);
        assert_eq!(ScalarType::of(&DataType::Real), ScalarType::Float);
        assert_eq!(ScalarType::of(&DataType::Double(ExactNumberInfo::None)), ScalarType::Float);
        assert_eq!(ScalarType::of(&DataType::DoublePrecision), ScalarType::Float);
        assert_eq!(
            ScalarType::of(&DataType::CharVarying(None)),
            ScalarType::Text(TextRules::DEFAULT)
        );
        assert_eq!(ScalarType::of(&DataType::Boolean), ScalarType::Boolean);
        assert_eq!(ScalarType::of(&DataType::Bool), ScalarType::Boolean);
        assert_eq!(ScalarType::of(&DataType::Date), ScalarType::Date);
        assert_eq!(
            ScalarType::of(&DataType::Timestamp(None, TimezoneInfo::None)),
            ScalarType::Timestamp
        );
        assert_eq!(
            ScalarType::of(&DataType::Timestamp(None, TimezoneInfo::WithoutTimeZone)),
            ScalarType::Timestamp
        );
        // A type the evaluator does not model stays `Other`
        assert_eq!(ScalarType::of(&DataType::Time(None, TimezoneInfo::None)), ScalarType::Other);
        // PostgreSQL has no `string` type
        assert_eq!(ScalarType::of(&DataType::String(Some(3))), ScalarType::Other);
    }

    #[test]
    fn scalar_type_predicates_name_the_number_and_text_types() {
        assert!(ScalarType::Integer(IntegerType::Integer).is_number());
        assert!(ScalarType::Numeric.is_number());
        assert!(!ScalarType::Boolean.is_number());
        assert!(ScalarType::Text(TextRules::DEFAULT).is_text());
        assert!(!ScalarType::Boolean.is_text());
        assert_eq!(
            ScalarType::Integer(IntegerType::Smallint).integer_bounds(),
            Some((i64::from(i16::MIN), i64::from(i16::MAX)))
        );
        assert_eq!(ScalarType::Boolean.integer_bounds(), None);
    }

    #[test]
    fn every_cast_value_maps_to_a_value() {
        assert_eq!(
            folded(CastValue::Bpchar(PgBpchar::new(String::from("ab  ")))),
            Cast::Folded(text("ab"))
        );
        let instant = PgTimestampTz::from_pg_micros(0).unwrap();
        assert_eq!(folded(CastValue::TimestampTz(instant)), Cast::Unknown);
    }

    #[test]
    fn text_length_counts_characters() {
        assert_eq!(text_length(&text("héllo")), Some(5));
        assert_eq!(text_length(&PgValue::Integer(1)), None);
    }
}
