//! The scalar types the evaluator reads an expression as.

use sql_values::IntegerType;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
