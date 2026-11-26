//! Literals, typed string literals, and the casts and collations applied
//! to an expression, forward.

use sql_values::postgres::PgValue;
use sqlparser::ast::{CastKind, DataType, Expr, ObjectName, TypedString, Value, ValueWithSpan};

use super::{
    super::{
        context::Context,
        dialect::EvalDialect,
        domain::Evaluation,
        scalar::{self, ArrayElement, Cast, ScalarType, TextRules},
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
    match scalar::cast(&PgValue::Text(text.clone()), &ScalarType::UnknownLiteral, &ty) {
        Cast::Folded(datum) => Evaluation::known(datum, ty),
        Cast::Raises => Evaluation::raising(ty),
        Cast::Null | Cast::Unknown => Evaluation::unknown(ty),
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
    // PostgreSQL casts each element of a cast `ARRAY[...]` to the element
    // type, which also types an empty one.
    if let (Expr::Array(array), DataType::Array(definition)) =
        (crate::support::unnest(expr), data_type)
        && let Some(element) = scalar::array_element(definition)
        && matches!(kind, CastKind::Cast | CastKind::DoubleColon)
    {
        return super::arrays::cast_constructor(state, kind, array, data_type, element);
    }
    let source = forward(state, expr)?;
    cast_evaluation(state, kind, source, data_type)
}

/// The dialect's reading of a cast of the already-evaluated `source`, met
/// with the checks of `data_type`.
pub(super) fn cast_evaluation<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    kind: &CastKind,
    source: Evaluation,
    data_type: &DataType,
) -> Result<Evaluation, C::Error> {
    let converted = conversion(state, kind, source, data_type);
    checked(state.context, data_type, converted)
}

/// The dialect's reading of a cast of `source` to `data_type` before the
/// type's checks.
pub(super) fn conversion<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    kind: &CastKind,
    source: Evaluation,
    data_type: &DataType,
) -> Evaluation {
    let to = state.context.data_type(data_type);
    state.dialect.cast(
        source,
        kind,
        data_type,
        to,
        |element| state.context.data_type(element),
        state.options,
    )
}

/// `converted`, the value a cast or a typed literal of `data_type` converts
/// to, met with what the context says the type's checks make of it.
pub(super) fn checked<C: Context>(
    context: &mut C,
    data_type: &DataType,
    converted: Evaluation,
) -> Result<Evaluation, C::Error> {
    if !context.cast_checks(data_type)? {
        return Ok(converted);
    }
    let raises = converted.may_raise();
    let checked = context.checked_cast(data_type, converted)?;
    Ok(if raises { checked.with_error(true) } else { checked })
}

/// Whether `node` is a cast or a typed literal whose type checks the
/// converted value as the expression runs, which the planner never folds.
/// An array's input function runs its elements' checks as an untyped
/// literal is parsed, so a cast of one to an array checks nothing later.
pub(super) fn checks_at_run_time<C: Context>(
    context: &mut C,
    node: &Expr,
) -> Result<bool, C::Error> {
    let (data_type, untyped) = match node {
        Expr::Cast { expr, data_type, .. } => {
            let untyped = match crate::support::unnest(expr) {
                Expr::Value(literal) => value(literal).is_untyped(),
                _ => false,
            };
            (data_type, untyped)
        }
        Expr::TypedString(typed) => (&typed.data_type, true),
        _ => return Ok(false),
    };
    if untyped && matches!(data_type, DataType::Array(_)) {
        return Ok(false);
    }
    context.cast_checks(data_type)
}

/// The forward evaluation of `expr COLLATE collation`, the operand's values
/// under the collation the context names, marked explicit. An untyped
/// literal or `NULL` reads as text, and an operand of another type than
/// text may raise, as an engine may refuse to collate it.
pub(super) fn collate<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
    collation: &ObjectName,
) -> Result<Evaluation, C::Error> {
    let operand = forward(state, expr)?;
    let rules = TextRules { explicit: true, ..state.context.collation(collation)? };
    Ok(match operand.ty() {
        ty if ty.is_text() || operand.is_untyped() => operand.with_type(ScalarType::Text(rules)),
        ScalarType::Array(ArrayElement::Text(_)) => {
            operand.with_type(ScalarType::Array(ArrayElement::Text(rules)))
        }
        _ => operand.with_error(true),
    })
}

#[cfg(test)]
mod tests {
    use sql_values::postgres::{IntegerType, PgValue};
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
        assert_eq!(*evaluation.ty(), ScalarType::Unknown);
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
        assert_eq!(*evaluation.ty(), ScalarType::UnknownLiteral);
    }

    /// A placeholder is unknown to the evaluator.
    #[test]
    fn a_placeholder_is_unknown() {
        let evaluation = super::value(&vspan(Value::Placeholder("?".to_string())));
        assert_eq!(*evaluation.ty(), ScalarType::Unknown);
        assert!(evaluation.may_raise());
    }

    /// A valid `DATE` literal folds to its date; an invalid one raises.
    #[test]
    fn a_typed_date_folds_and_an_invalid_one_raises() {
        let good = super::typed_string(
            &mut ctx(),
            &typed(DataType::Date, Value::SingleQuotedString("2024-01-01".to_string())),
        );
        assert_eq!(*good.ty(), ScalarType::Date);
        assert!(!good.may_raise());
        assert!(matches!(good.constant(), Some(PgValue::Date(_))));

        let bad = super::typed_string(
            &mut ctx(),
            &typed(DataType::Date, Value::SingleQuotedString("2024-13-40".to_string())),
        );
        assert_eq!(*bad.ty(), ScalarType::Date);
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
        assert_eq!(*good.ty(), ScalarType::Timestamp);
        assert!(!good.may_raise());
        assert!(matches!(good.constant(), Some(PgValue::Timestamp(_))));

        let bad = super::typed_string(
            &mut ctx(),
            &typed(
                DataType::Timestamp(None, TimezoneInfo::None),
                Value::SingleQuotedString("2024-01-01 25:00:00".to_string()),
            ),
        );
        assert_eq!(*bad.ty(), ScalarType::Timestamp);
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
        assert_eq!(*integer.ty(), ScalarType::Integer(IntegerType::Integer));
        assert_eq!(integer.constant(), Some(&PgValue::Integer(5)));
        let float = super::typed_string(
            &mut ctx(),
            &typed(DataType::Real, Value::SingleQuotedString("1.5".to_string())),
        );
        assert_eq!(*float.ty(), ScalarType::Float);
        assert!(float.may_raise());
    }

    /// A typed literal that is not a string is left unknown.
    #[test]
    fn a_non_string_typed_literal_is_unknown() {
        let evaluation = super::typed_string(
            &mut ctx(),
            &typed(DataType::Date, Value::Number("5".to_string(), false)),
        );
        assert_eq!(*evaluation.ty(), ScalarType::Date);
        assert!(evaluation.may_raise());
    }
}
