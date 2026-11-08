//! Literals and typed string literals, forward.

use sql_values::PgValue;
use sqlparser::ast::{CastKind, DataType, Expr, TypedString, Value, ValueWithSpan};

use super::{
    super::{
        context::Context,
        dialect::EvalDialect,
        domain::Evaluation,
        scalar::{self, Cast, ScalarType},
    },
    State, forward,
};

/// The forward evaluation of a literal, typed the way PostgreSQL types it.
pub(super) fn value(value: &ValueWithSpan) -> Evaluation {
    match &value.value {
        Value::Number(number, _) => {
            match scalar::number_literal(number, false) {
                Some((datum, ty)) => Evaluation::known(datum, ty),
                None => Evaluation::unknown(ScalarType::Unknown),
            }
        }
        Value::Null => Evaluation::null(ScalarType::Unknown),
        Value::Boolean(true) => Evaluation::known(PgValue::Boolean(true), ScalarType::Boolean),
        Value::Boolean(false) => Evaluation::known(PgValue::Boolean(false), ScalarType::Boolean),
        Value::SingleQuotedString(text)
        | Value::EscapedStringLiteral(text)
        | Value::NationalStringLiteral(text) => {
            Evaluation::known(PgValue::Text(text.clone()), ScalarType::UnknownLiteral)
        }
        Value::DollarQuotedString(dollar) => {
            Evaluation::known(PgValue::Text(dollar.value.clone()), ScalarType::UnknownLiteral)
        }
        _ => Evaluation::unknown(ScalarType::Unknown),
    }
}

/// The forward evaluation of a typed string literal, such as `DATE
/// '2024-01-01'`, which PostgreSQL reads with the type's input function.
pub(super) fn typed_string<C: Context>(context: &mut C, typed: &TypedString) -> Evaluation {
    let ty = context.data_type(&typed.data_type);
    let Value::SingleQuotedString(text) = &typed.value.value else {
        return Evaluation::unknown(ty);
    };
    match scalar::cast(&PgValue::Text(text.clone()), ScalarType::UnknownLiteral, ty) {
        Cast::Folded(datum) => Evaluation::known(datum, ty),
        Cast::Raises => Evaluation::raising(ty),
        Cast::Unknown => Evaluation::unknown(ty),
    }
}

/// The forward evaluation of a cast, asking the dialect for its reading of
/// the cast.
pub(super) fn cast<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    kind: &CastKind,
    expr: &Expr,
    data_type: &DataType,
) -> Result<Evaluation, C::Error> {
    let to = state.context.data_type(data_type);
    let source = forward(state, expr)?;
    Ok(state.dialect.cast(
        source,
        kind,
        data_type,
        to,
        |element| state.context.data_type(element),
        state.options,
    ))
}

#[cfg(test)]
mod tests {
    use sql_values::{IntegerType, PgValue};
    use sqlparser::{
        ast::{DataType, TimezoneInfo, TypedString, Value, ValueWithSpan},
        tokenizer::Span,
    };

    use crate::{scalar::ScalarType, test_context::ColumnContext};

    fn vspan(value: Value) -> ValueWithSpan {
        ValueWithSpan { value, span: Span::empty() }
    }

    fn typed(data_type: DataType, value: Value) -> TypedString {
        TypedString { data_type, value: vspan(value), uses_odbc_syntax: false }
    }

    fn ctx() -> ColumnContext {
        ColumnContext::new()
    }

    /// A number PostgreSQL cannot type is unknown, not a value.
    #[test]
    fn an_untypeable_number_stays_unknown() {
        let evaluation = super::value(&vspan(Value::Number("0x".to_string(), false)));
        assert_eq!(evaluation.ty(), ScalarType::Unknown);
        assert!(evaluation.may_raise());
    }

    /// A dollar-quoted string is an untyped literal of its body.
    #[test]
    fn a_dollar_quoted_string_is_an_untyped_literal() {
        let evaluation =
            super::value(&vspan(Value::DollarQuotedString(sqlparser::ast::DollarQuotedString {
                value: "hello".to_string(),
                tag: None,
            })));
        assert_eq!(evaluation.constant(), Some(&PgValue::Text("hello".to_string())));
        assert_eq!(evaluation.ty(), ScalarType::UnknownLiteral);
    }

    /// A placeholder is unknown to the evaluator.
    #[test]
    fn a_placeholder_is_unknown() {
        let evaluation = super::value(&vspan(Value::Placeholder("?".to_string())));
        assert_eq!(evaluation.ty(), ScalarType::Unknown);
        assert!(evaluation.may_raise());
    }

    /// A valid `DATE` literal folds to its date; an invalid one raises.
    #[test]
    fn a_typed_date_folds_and_an_invalid_one_raises() {
        let good = super::typed_string(
            &mut ctx(),
            &typed(DataType::Date, Value::SingleQuotedString("2024-01-01".to_string())),
        );
        assert_eq!(good.ty(), ScalarType::Date);
        assert!(!good.may_raise());
        assert!(matches!(good.constant(), Some(PgValue::Date(_))));

        let bad = super::typed_string(
            &mut ctx(),
            &typed(DataType::Date, Value::SingleQuotedString("2024-13-40".to_string())),
        );
        assert_eq!(bad.ty(), ScalarType::Date);
        assert!(bad.always_raises());
    }

    /// A valid `TIMESTAMP` literal folds; an invalid one raises.
    #[test]
    fn a_typed_timestamp_folds_and_an_invalid_one_raises() {
        let good = super::typed_string(
            &mut ctx(),
            &typed(
                DataType::Timestamp(None, TimezoneInfo::None),
                Value::SingleQuotedString("2024-01-01 01:02:03".to_string()),
            ),
        );
        assert_eq!(good.ty(), ScalarType::Timestamp);
        assert!(!good.may_raise());
        assert!(matches!(good.constant(), Some(PgValue::Timestamp(_))));

        let bad = super::typed_string(
            &mut ctx(),
            &typed(
                DataType::Timestamp(None, TimezoneInfo::None),
                Value::SingleQuotedString("2024-01-01 25:00:00".to_string()),
            ),
        );
        assert_eq!(bad.ty(), ScalarType::Timestamp);
        assert!(bad.always_raises());
    }

    /// A typed literal reads through the type's input function, and one of a
    /// type whose values are not modelled is left unknown.
    #[test]
    fn a_typed_literal_reads_through_the_input_function() {
        let integer = super::typed_string(
            &mut ctx(),
            &typed(DataType::Int(None), Value::SingleQuotedString(" 5 ".to_string())),
        );
        assert_eq!(integer.ty(), ScalarType::Integer(IntegerType::Integer));
        assert_eq!(integer.constant(), Some(&PgValue::Integer(5)));
        let float = super::typed_string(
            &mut ctx(),
            &typed(DataType::Real, Value::SingleQuotedString("1.5".to_string())),
        );
        assert_eq!(float.ty(), ScalarType::Float);
        assert!(float.may_raise());
    }

    /// A typed literal that is not a string is left unknown.
    #[test]
    fn a_non_string_typed_literal_is_unknown() {
        let evaluation = super::typed_string(
            &mut ctx(),
            &typed(DataType::Date, Value::Number("5".to_string(), false)),
        );
        assert_eq!(evaluation.ty(), ScalarType::Date);
        assert!(evaluation.may_raise());
    }
}
