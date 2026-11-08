//! PostgreSQL facts read through the public API, and the guarantees the
//! backward pass places on the columns a condition names.

use sqlparser::{
    ast::{Expr, Function, ObjectNamePart, Query, SelectItem, SetExpr, Statement},
    dialect::PostgreSqlDialect,
    parser::{Parser, ParserError},
};
use sqlparser_expr_eval::{
    Builtin, Callee, CalleeKind, Context, Evaluation, Guarantees, Interval, NullInput, Outcomes,
    PostgreSqlOptions, QueryEvaluation, Resolved, Rows, ScalarType, TextRules, Unlisted, Values,
    Volatility, evaluate, implied,
    sql_values::{IntegerType, PgValue},
};

/// A context that resolves a fixed set of columns and one nested query.
struct Ctx {
    columns: Vec<(String, Evaluation)>,
    callees: Vec<(String, Callee)>,
    query: Option<QueryEvaluation>,
}

impl Ctx {
    fn with_columns(columns: Vec<(String, Evaluation)>) -> Self {
        Self { columns, callees: Vec::new(), query: None }
    }

    fn with_callee(mut self, name: &str, callee: Callee) -> Self {
        self.callees.push((name.to_string(), callee));
        self
    }

    fn with_query(mut self, query: QueryEvaluation) -> Self {
        self.query = Some(query);
        self
    }
}

/// A nullable, non-raising column of `ty` with unlisted values.
fn column(ty: ScalarType, null: bool) -> Evaluation {
    Evaluation::new(ty, null, false, Values::Unlisted(Unlisted::any()))
}

impl Context for Ctx {
    type Reference = String;
    type Error = ParserError;

    fn reference(
        &mut self,
        expression: &Expr,
    ) -> Result<Option<Resolved<Self::Reference>>, Self::Error> {
        let name = match expression {
            Expr::Identifier(ident) => Some(ident.value.clone()),
            Expr::CompoundIdentifier(parts) if parts.len() == 1 => Some(parts[0].value.clone()),
            _ => None,
        };
        let Some(name) = name else {
            return Ok(None);
        };
        Ok(self.columns.iter().find(|(key, _)| *key == name).map(|(_, evaluation)| {
            Resolved { reference: name.clone(), evaluation: evaluation.clone() }
        }))
    }

    fn callees(&mut self, call: &Function) -> Result<Vec<Callee>, Self::Error> {
        let Some(name) = call.name.0.last().and_then(ObjectNamePart::as_ident) else {
            return Ok(Vec::new());
        };
        let name = name.value.to_lowercase();
        Ok(self
            .callees
            .iter()
            .filter(|(callee_name, _)| *callee_name == name)
            .map(|(_, callee)| callee.clone())
            .collect())
    }

    fn query(
        &mut self,
        _query: &Query,
        _guarantees: &Guarantees<Self::Reference>,
    ) -> Result<QueryEvaluation, Self::Error> {
        Ok(self.query.clone().unwrap_or_else(QueryEvaluation::unknown))
    }
}

/// The expression of `SELECT expression`, parsed as PostgreSQL.
#[expect(clippy::expect_used, clippy::panic, reason = "a malformed test expression fails the test")]
fn parse(expression: &str) -> Expr {
    let statements =
        Parser::parse_sql(&PostgreSqlDialect {}, &format!("SELECT {expression}")).expect("parses");
    let Statement::Query(query) = &statements[0] else {
        panic!("the test statement is a query");
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        panic!("the test query is a select");
    };
    let [SelectItem::UnnamedExpr(expr)] = select.projection.as_slice() else {
        panic!("the test select projects one expression");
    };
    expr.clone()
}

/// Reads `sql` under PostgreSQL with no settings.
#[expect(clippy::expect_used, reason = "a context error fails the test")]
fn read(mut context: Ctx, sql: &str) -> Evaluation {
    let dialect = PostgreSqlDialect {};
    let options = PostgreSqlOptions::default();
    evaluate(&parse(sql), &dialect, &options, &mut context, &Guarantees::new())
        .expect("the context resolves")
}

/// Reads `sql` and returns the guarantee it places under the given outcomes.
#[expect(clippy::expect_used, reason = "a context error fails the test")]
fn restrict(mut context: Ctx, sql: &str, outcomes: Outcomes) -> Guarantees<String> {
    let dialect = PostgreSqlDialect {};
    let options = PostgreSqlOptions::default();
    implied(&parse(sql), outcomes, &dialect, &options, &mut context, &Guarantees::new())
        .expect("the context resolves")
}

/// A context with a nullable text `x` and a non-null integer `id`.
fn context() -> Ctx {
    Ctx::with_columns(vec![
        ("x".into(), column(ScalarType::Text(TextRules::DEFAULT), true)),
        ("id".into(), column(ScalarType::Integer(IntegerType::Integer), false)),
    ])
}

#[test]
fn int4_max_literal_is_typed_int4() {
    let evaluation = read(context(), "2147483647");
    assert_eq!(evaluation.ty(), ScalarType::Integer(IntegerType::Integer));
    assert!(evaluation.constant().is_some());
    assert!(!evaluation.may_be_null());
}

#[test]
fn an_int4_overflow_literal_is_typed_int8() {
    let evaluation = read(context(), "2147483648");
    assert_eq!(evaluation.ty(), ScalarType::Integer(IntegerType::Bigint));
}

#[test]
fn an_int4_overflow_always_raises() {
    let evaluation = read(context(), "2147483647 + 1");
    assert!(evaluation.always_raises());
}

#[test]
fn a_constant_division_by_zero_always_raises() {
    let evaluation = read(context(), "1 / 0");
    assert!(evaluation.always_raises());
}

#[test]
fn an_integer_division_folds_to_the_floor() {
    let evaluation = read(context(), "7 / 2");
    assert_eq!(evaluation.constant(), Some(&PgValue::Integer(3)));
}

#[test]
fn a_numeric_division_keeps_ten_extra_digits() {
    let evaluation = read(context(), "7.0 / 2");
    let constant = evaluation.constant().expect("a numeric division is a constant");
    let PgValue::Numeric(numeric) = constant else {
        panic!("the constant is numeric");
    };
    assert_eq!(numeric.to_string(), "3.5000000000000000");
}

#[test]
fn a_concat_with_a_null_operand_is_always_null() {
    let evaluation = read(context(), "'a' || NULL");
    assert!(evaluation.is_always_null());
}

#[test]
fn nullif_of_equal_arguments_is_always_null() {
    let evaluation = read(context(), "nullif(1, 1)");
    assert!(evaluation.is_always_null());
}

#[test]
fn or_with_null_keeps_the_truth() {
    let evaluation = read(context(), "true OR NULL");
    assert!(evaluation.may_be_true());
    assert!(!evaluation.may_be_false());
    assert!(!evaluation.may_be_null());
}

#[test]
fn a_case_with_a_null_condition_skips_the_branch() {
    let evaluation = read(context(), "CASE WHEN NULL THEN 1 ELSE 2 END");
    assert_eq!(evaluation.constant(), Some(&PgValue::Integer(2)));
}

#[test]
fn equality_with_a_null_literal_is_always_null() {
    let evaluation = read(context(), "x = NULL");
    assert!(evaluation.is_always_null());
}

#[test]
fn an_in_over_only_a_null_member_is_never_true() {
    let evaluation = read(context(), "x IN (NULL)");
    assert!(!evaluation.may_be_true());
}

#[test]
fn an_equality_true_restriction_keeps_the_value() {
    let guarantees = restrict(context(), "x = 'read'", Outcomes::TRUE);
    let x = guarantees.reference(&"x".to_string()).expect("the condition names x");
    assert!(!x.may_be_null());
    assert_eq!(x.values(), &Values::Listed([PgValue::Text("read".into())].into()));
}

#[test]
fn an_in_true_restriction_keeps_the_members() {
    let guarantees = restrict(context(), "x IN ('read', 'write')", Outcomes::TRUE);
    let x = guarantees.reference(&"x".to_string()).expect("the condition names x");
    assert!(!x.may_be_null());
    let Values::Listed(listed) = x.values() else {
        panic!("the restriction lists the members");
    };
    assert_eq!(listed.len(), 2);
}

#[test]
fn an_is_null_true_restriction_keeps_only_null() {
    let guarantees = restrict(context(), "x IS NULL", Outcomes::TRUE);
    let x = guarantees.reference(&"x".to_string()).expect("the condition names x");
    assert!(x.may_be_null());
}

#[test]
fn an_is_not_null_true_restriction_drops_null() {
    let guarantees = restrict(context(), "x IS NOT NULL", Outcomes::TRUE);
    let x = guarantees.reference(&"x".to_string()).expect("the condition names x");
    assert!(!x.may_be_null());
}

#[test]
fn a_scalar_subquery_over_one_row_keeps_the_value_unknown() {
    let evaluation = column(ScalarType::Text(TextRules::DEFAULT), false);
    let context =
        Ctx::with_columns(vec![("x".into(), column(ScalarType::Text(TextRules::DEFAULT), true))])
            .with_query(QueryEvaluation::new(Rows::ONE, Some(vec![evaluation]), false));
    let evaluation = read(context, "(SELECT note FROM t) = 'read'");
    assert!(evaluation.may_be_true());
    assert!(evaluation.may_be_false());
}

/// A builtin callee that folds `builtin` and returns `returns`.
fn builtin_callee(builtin: Builtin, returns: ScalarType) -> Callee {
    Callee::new(
        Some(builtin),
        NullInput::Strict,
        Volatility::Immutable,
        CalleeKind::Normal,
        returns,
        false,
        0,
    )
}

fn length_callee() -> Callee {
    builtin_callee(Builtin::Length, ScalarType::Integer(IntegerType::Integer))
}

#[test]
fn a_length_comparison_counts_the_characters() {
    // `length(x) = 3` holds only for three-character `x`, which are not
    // `NULL`
    let guarantees =
        restrict(context().with_callee("length", length_callee()), "length(x) = 3", Outcomes::TRUE);
    let x = guarantees.reference(&"x".to_string()).expect("the condition names x");
    assert!(!x.may_be_null());
    let Values::Unlisted(unlisted) = x.values() else {
        panic!("the restriction stays unlisted");
    };
    assert_eq!(
        unlisted.length,
        Interval { low: core::ops::Bound::Included(3), high: core::ops::Bound::Included(3) }
    );
    assert!(unlisted.numbers.is_unbounded());
}

#[test]
fn a_window_length_comparison_counts_nothing() {
    // A windowed call does not count its argument, so `x` is left alone
    let guarantees = restrict(
        context().with_callee("length", length_callee()),
        "length(x) OVER () = 3",
        Outcomes::TRUE,
    );
    assert_eq!(guarantees.reference(&"x".to_string()), None);
}

#[test]
fn a_call_with_no_arguments_counts_nothing() {
    // `current_timestamp` spells no argument list to count
    let guarantees = restrict(context(), "current_timestamp = x", Outcomes::TRUE);
    let x = guarantees.reference(&"x".to_string()).expect("the condition names x");
    assert!(!x.may_be_null());
    let Values::Unlisted(unlisted) = x.values() else {
        panic!("the restriction stays unlisted");
    };
    // No length constraint: the hull a fresh text reading carries
    assert_eq!(
        unlisted.length,
        Interval { low: core::ops::Bound::Included(0), high: core::ops::Bound::Unbounded }
    );
}

#[test]
fn a_length_with_two_arguments_counts_nothing() {
    // More than one argument is not the `length(text)` form
    let guarantees = restrict(
        context().with_callee("length", length_callee()),
        "length(x, 'a') = 3",
        Outcomes::TRUE,
    );
    assert_eq!(guarantees.reference(&"x".to_string()), None);
}

#[test]
fn a_null_upper_result_comes_from_a_null_input() {
    // `upper(x) = 'A'` being `NULL` with a non-`NULL` right side means
    // `upper(x)` was `NULL`, and PostgreSQL's `upper` returns `NULL` exactly
    // on a `NULL` argument, so `x` was `NULL`
    let guarantees = restrict(
        context().with_callee(
            "upper",
            builtin_callee(Builtin::Upper, ScalarType::Text(TextRules::DEFAULT)),
        ),
        "upper(x) = 'A'",
        Outcomes::NULL,
    );
    let x = guarantees.reference(&"x".to_string()).expect("the result names x");
    assert!(x.may_be_null());
    assert!(x.values().is_empty());
}

#[test]
fn a_null_concat_result_comes_from_no_input_reason() {
    // `concat` returns `NULL` for no argument, so a `NULL` result says
    // nothing about `x`
    let guarantees = restrict(
        context().with_callee(
            "concat",
            builtin_callee(Builtin::Concat, ScalarType::Text(TextRules::DEFAULT)),
        ),
        "concat(x)",
        Outcomes::NULL,
    );
    assert_eq!(guarantees.reference(&"x".to_string()), None);
}
