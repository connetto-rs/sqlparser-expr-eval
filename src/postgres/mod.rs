//! PostgreSQL's answers to the [`EvalDialect`] questions.

mod arithmetic;
mod calls;
mod cast;
mod comparison;

use alloc::{collections::BTreeSet, format};

use sql_values::postgres::PgValue;
use sqlparser::{
    ast::{
        BinaryOperator, CastKind, DataType, Expr, Function, FunctionArg, FunctionArgExpr,
        FunctionArguments, Value,
    },
    dialect::PostgreSqlDialect,
};

use crate::{
    Comparison,
    context::{Builtin, Callee},
    dialect::EvalDialect,
    domain::{Evaluation, Values},
    scalar::{ScalarType, TextRules},
    support,
};

/// The settings of the PostgreSQL dialect that change its answers.
#[non_exhaustive]
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct PostgreSqlOptions {
    /// Whether the nonstandard form `expr = NULL` reads as `expr IS NULL`,
    /// the way the
    /// [`transform_null_equals` setting](https://www.postgresql.org/docs/current/runtime-config-compatible.html#GUC-TRANSFORM-NULL-EQUALS)
    /// rewrites it.
    pub transform_null_equals: bool,
}

impl EvalDialect for PostgreSqlDialect {
    type Options = PostgreSqlOptions;

    fn comparison(
        &self,
        left: Evaluation,
        op: &BinaryOperator,
        right: Evaluation,
        options: &PostgreSqlOptions,
    ) -> Comparison {
        let _ = options;
        comparison::comparison(left, op, right)
    }

    fn concat(
        &self,
        left: &Evaluation,
        right: &Evaluation,
        options: &PostgreSqlOptions,
    ) -> Option<Evaluation> {
        let _ = options;
        // Fold when both sides list values whose output is known.
        let (Values::Listed(left_values), Values::Listed(right_values)) =
            (left.values(), right.values())
        else {
            return None;
        };
        if left_values.is_empty() || right_values.is_empty() {
            return None;
        }
        let mut out = BTreeSet::new();
        let mut known = true;
        'outer: for a in left_values {
            for b in right_values {
                if let (Some(x), Some(y)) = (concat_text(a), concat_text(b)) {
                    out.insert(PgValue::Text(format!("{x}{y}")));
                } else {
                    known = false;
                    break 'outer;
                }
            }
        }
        known.then(|| {
            let null = left.may_be_null() || right.may_be_null();
            let error = left.may_raise() || right.may_raise();
            Evaluation::new(ScalarType::Text(TextRules::DEFAULT), null, error, Values::Listed(out))
        })
    }

    fn arithmetic(
        &self,
        left: &Evaluation,
        op: &BinaryOperator,
        right: &Evaluation,
        options: &PostgreSqlOptions,
    ) -> Evaluation {
        let _ = options;
        arithmetic::arithmetic(left, op, right)
    }

    fn unary_plus_preserves(&self, inner: &Evaluation, options: &PostgreSqlOptions) -> bool {
        let _ = options;
        inner.ty().is_number()
    }

    fn cast(
        &self,
        source: Evaluation,
        kind: &CastKind,
        data_type: &DataType,
        to: ScalarType,
        element_type: impl Fn(&DataType) -> ScalarType,
        options: &PostgreSqlOptions,
    ) -> Evaluation {
        let _ = options;
        cast::cast(&source, kind, data_type, to, element_type)
    }

    fn named_call(
        &self,
        name: &str,
        arguments: &[Evaluation],
        options: &PostgreSqlOptions,
    ) -> Option<Evaluation> {
        let _ = options;
        let error = arguments.iter().any(Evaluation::may_raise);
        calls::name_recognised(name, arguments, error)
    }

    fn null_equals_operand<'a>(
        &self,
        left: &'a Expr,
        right: &'a Expr,
        options: &PostgreSqlOptions,
    ) -> Option<&'a Expr> {
        if !options.transform_null_equals {
            return None;
        }
        let null_literal = |expr: &Expr| {
            matches!(
                support::unnest(expr),
                Expr::Value(value) if matches!(value.value, Value::Null)
            )
        };
        if null_literal(right) {
            Some(left)
        } else if null_literal(left) {
            Some(right)
        } else {
            None
        }
    }

    fn null_only_on_null_input(&self, callees: &[Callee], options: &PostgreSqlOptions) -> bool {
        let _ = options;
        callees.iter().all(|callee| {
            matches!(
                callee.builtin,
                Some(
                    Builtin::Length
                        | Builtin::OctetLength
                        | Builtin::Lower
                        | Builtin::Upper
                        | Builtin::Btrim
                        | Builtin::Abs
                )
            )
        })
    }

    fn counted_length_argument(
        &self,
        call: &Function,
        callees: &[Callee],
        options: &PostgreSqlOptions,
    ) -> Option<usize> {
        let _ = options;
        if call.over.is_some() {
            return None;
        }
        let FunctionArguments::List(list) = &call.args else {
            return None;
        };
        let [FunctionArg::Unnamed(FunctionArgExpr::Expr(_))] = list.args.as_slice() else {
            return None;
        };
        let builtin = !callees.is_empty()
            && callees.iter().all(|callee| callee.builtin == Some(Builtin::Length));
        builtin.then_some(0)
    }
}

/// The text `||` reads a value as, which for a `boolean` is its text cast.
fn concat_text(value: &PgValue) -> Option<alloc::string::String> {
    match value {
        PgValue::Boolean(flag) => Some(if *flag { "true" } else { "false" }.into()),
        _ => value.output().map(|output| format!("{output}")),
    }
}

#[cfg(test)]
mod tests {
    use sqlparser::dialect::PostgreSqlDialect;

    use super::PostgreSqlOptions;
    use crate::{
        domain::{Evaluation, Outcomes, Unlisted, Values},
        evaluate, implied,
        scalar::{ScalarType, TextRules},
        test_context::{ColumnContext, parse},
    };

    /// A nullable, non-raising text column.
    fn nullable_text() -> Evaluation {
        Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        )
    }

    fn context() -> ColumnContext {
        ColumnContext::new().with_column("x", nullable_text())
    }

    fn read(sql: &str, transform: bool) -> Evaluation {
        let options = PostgreSqlOptions { transform_null_equals: transform };
        evaluate(
            &parse(sql),
            &PostgreSqlDialect {},
            &options,
            &mut context(),
            &crate::guarantees::Guarantees::new(),
        )
        .expect("the context resolves")
    }

    #[test]
    fn equals_null_is_a_comparison_when_off() {
        let evaluation = read("x = NULL", false);
        assert!(evaluation.is_always_null());
    }

    #[test]
    fn equals_null_reads_as_is_null_when_on() {
        let evaluation = read("x = NULL", true);
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
        assert!(!evaluation.is_always_null());
    }

    #[test]
    fn reversed_equals_null_reads_as_is_null_when_on() {
        let evaluation = read("NULL = x", true);
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
        assert!(!evaluation.is_always_null());
    }

    #[test]
    fn parenthesized_null_still_counts_when_on() {
        let evaluation = read("(NULL) = x", true);
        assert!(evaluation.may_be_true());
        assert!(!evaluation.is_always_null());
    }

    #[test]
    fn a_cast_of_null_is_not_the_literal() {
        let evaluation = read("NULL::text = x", true);
        assert!(evaluation.is_always_null());
    }

    #[test]
    fn other_operators_are_unaffected() {
        assert!(read("x <> NULL", true).is_always_null());
        assert!(read("x < NULL", true).is_always_null());
    }

    #[test]
    fn in_a_null_member_is_unaffected() {
        let evaluation = read("x IN (NULL)", true);
        assert!(!evaluation.may_be_true());
    }

    #[test]
    fn implied_reads_the_null_rows_when_on() {
        let options = PostgreSqlOptions { transform_null_equals: true };
        let guarantees = implied(
            &parse("x = NULL"),
            Outcomes::TRUE,
            &PostgreSqlDialect {},
            &options,
            &mut context(),
            &crate::guarantees::Guarantees::new(),
        )
        .expect("the context resolves");
        let x = guarantees.reference(&"x".to_string()).expect("the condition names x");
        assert!(x.may_be_null());
    }

    #[test]
    fn concat_folds_the_listed_outputs() {
        // `'a' || 'b'` is the single value `'ab'`
        let evaluation = read("'a' || 'b'", false);
        assert_eq!(evaluation.ty(), ScalarType::Text(TextRules::DEFAULT));
        assert!(!evaluation.may_be_null());
        assert!(!evaluation.may_raise());
        assert_eq!(
            evaluation.values(),
            &Values::Listed(alloc::collections::BTreeSet::from([
                sql_values::postgres::PgValue::Text(alloc::string::String::from("ab"))
            ]))
        );
    }

    #[test]
    fn concat_reads_a_boolean_as_its_text_cast() {
        // `'1' || false` is `1false`
        let evaluation = read("'1' || false", false);
        assert_eq!(
            evaluation.constant(),
            Some(&sql_values::postgres::PgValue::Text("1false".into()))
        );
    }

    #[test]
    fn concat_does_not_fold_when_a_side_is_unlisted() {
        // `x` may hold text the read cannot enumerate
        let evaluation = read("'a' || x", false);
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
        assert!(evaluation.may_be_null());
        assert!(!evaluation.may_raise());
    }

    #[test]
    fn concat_does_not_fold_when_a_side_holds_no_value() {
        // `1 / 0` raises, so the right side holds no value the fold can use
        let evaluation = read("'a' || (1 / 0)", false);
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
        assert!(evaluation.may_raise());
        assert!(!evaluation.may_be_null());
    }

    #[test]
    fn concat_does_not_fold_a_date_output() {
        // A date has no output the fold can join
        let evaluation = read("'a' || DATE '2024-01-01'", false);
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
        assert!(!evaluation.may_raise());
    }

    #[test]
    fn unary_plus_keeps_the_number() {
        // `+1` is `1`, as the sign over a number changes nothing
        let evaluation = read("+1", false);
        assert_eq!(
            evaluation.values(),
            &Values::Listed(alloc::collections::BTreeSet::from([
                sql_values::postgres::PgValue::Integer(1)
            ]))
        );
        assert!(!evaluation.may_raise());
    }
}
