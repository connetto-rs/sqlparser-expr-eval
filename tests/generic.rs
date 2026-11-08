//! The conservative answers every non-PostgreSQL dialect shares, read
//! through the public API.

extern crate alloc;

use alloc::collections::BTreeSet;

use sqlparser::{
    ast::{Expr, Function, ObjectNamePart, SelectItem, SetExpr, Statement},
    dialect::PostgreSqlDialect,
    parser::{Parser, ParserError},
};
use sqlparser_expr_eval::{
    Builtin, Callee, CalleeKind, Context, Evaluation, Guarantees, NullInput, Outcomes, Resolved,
    ScalarType, Unlisted, Values, Volatility, evaluate, implied,
    sql_values::{IntegerType, PgValue},
};

/// A context that resolves a fixed set of columns and callees.
struct Ctx {
    columns: Vec<(String, Evaluation)>,
    callees: Vec<(String, Callee)>,
}

impl Ctx {
    fn with_callee(mut self, name: &str, callee: Callee) -> Self {
        self.callees.push((name.to_string(), callee));
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
}

/// The expression of `SELECT expression`.
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

/// Reads `sql` under the conservative generic dialect, with no settings.
#[expect(clippy::expect_used, reason = "a context error fails the test")]
fn read(mut context: Ctx, sql: &str) -> Evaluation {
    evaluate(
        &parse(sql),
        &sqlparser::dialect::GenericDialect,
        &(),
        &mut context,
        &Guarantees::new(),
    )
    .expect("the context resolves")
}

/// A context with a nullable integer `x`.
fn context() -> Ctx {
    Ctx {
        columns: vec![("x".into(), column(ScalarType::Integer(IntegerType::Integer), true))],
        callees: Vec::new(),
    }
}

/// Reads `sql`'s guarantee under the given outcomes, in the conservative
/// dialect.
#[expect(clippy::expect_used, reason = "a context error fails the test")]
fn imply(mut context: Ctx, sql: &str, outcomes: Outcomes) -> Guarantees<String> {
    implied(
        &parse(sql),
        outcomes,
        &sqlparser::dialect::GenericDialect,
        &(),
        &mut context,
        &Guarantees::new(),
    )
    .expect("the context resolves")
}

/// A strict callee returning `returns`.
fn strict_callee(returns: ScalarType) -> Callee {
    Callee::new(
        None,
        NullInput::Strict,
        Volatility::Immutable,
        CalleeKind::Normal,
        returns,
        false,
        0,
    )
}

/// A callee that folds the built-in `builtin` and returns `returns`.
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

#[test]
fn arithmetic_stays_unknown_but_may_raise() {
    let evaluation = read(context(), "1 + 2");
    assert!(evaluation.may_raise());
    assert!(!evaluation.may_be_null());
    assert!(matches!(evaluation.values(), Values::Unlisted(_)));
}

#[test]
fn a_constant_division_by_zero_may_be_null_and_may_raise() {
    let evaluation = read(context(), "1 / 0");
    assert!(evaluation.may_be_null());
    assert!(evaluation.may_raise());
}

#[test]
fn a_constant_modulo_by_zero_may_be_null_and_may_raise() {
    let evaluation = read(context(), "1 % 0");
    assert!(evaluation.may_be_null());
    assert!(evaluation.may_raise());
}

#[test]
fn text_concatenation_folds_nothing() {
    let evaluation = read(context(), "'a' || 'b'");
    assert!(matches!(evaluation.values(), Values::Unlisted(_)));
}

#[test]
fn a_text_literal_against_a_number_is_unknown() {
    let evaluation = read(context(), "'1' = 1");
    assert!(evaluation.may_be_true());
    assert!(evaluation.may_be_false());
}

#[test]
fn boolean_literals_compare_by_value() {
    let evaluation = read(context(), "true = false");
    assert!(evaluation.may_be_false());
    assert!(!evaluation.may_be_true());
    assert!(!evaluation.may_be_null());
}

#[test]
fn an_integer_cast_of_a_number_keeps_the_value_unknown() {
    let evaluation = read(context(), "x::int");
    assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    assert!(evaluation.may_be_null());
}

#[test]
fn a_positive_sign_keeps_the_inner_unknown() {
    let evaluation = read(context(), "+1");
    assert!(matches!(evaluation.values(), Values::Unlisted(_)));
}

#[test]
fn a_null_cast_keeps_no_value() {
    // The default cast of a `NULL` source is `NULL`, raising nothing
    let evaluation = read(context(), "NULL::int");
    assert!(evaluation.is_always_null());
    assert!(!evaluation.may_raise());
}

#[test]
fn null_in_a_comparison_reads_as_null() {
    // The conservative dialect reads `x = NULL` as a comparison with a
    // `NULL` literal, which is always `NULL`
    let evaluation = read(context(), "x = NULL");
    assert!(evaluation.is_always_null());
}

#[test]
fn a_call_with_no_callees_reads_unknown() {
    let evaluation = read(context(), "f('a')");
    assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    assert!(evaluation.may_raise());
}

#[test]
fn a_strict_boolean_call_nulls_for_no_input_reason() {
    // The conservative dialect knows no builtin returning `NULL` exactly on
    // a `NULL` argument, so a `NULL` result says nothing about `x`
    let guarantees = imply(
        context().with_callee("f", strict_callee(ScalarType::Boolean)),
        "f(x)",
        Outcomes::NULL,
    );
    assert_eq!(guarantees.reference(&"x".to_string()), None);
}

#[test]
fn a_length_comparison_counts_nothing() {
    // The conservative dialect counts no argument, so the comparison says
    // nothing about `x`
    let guarantees = imply(
        context().with_callee(
            "length",
            builtin_callee(Builtin::Length, ScalarType::Integer(IntegerType::Integer)),
        ),
        "length(x) = 3",
        Outcomes::TRUE,
    );
    assert_eq!(guarantees.reference(&"x".to_string()), None);
}

/// `= ALL` over a `NULL` member reads `NULL` when the value equals the rest,
/// never `FALSE`.
#[test]
fn an_all_over_a_null_member_reads_null_when_it_equals_the_rest() {
    let listed = |value: i64| {
        Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(value)])),
        )
    };
    let context = Ctx { columns: vec![("x".to_string(), listed(1))], callees: Vec::new() };
    let result = read(context, "x = ALL (ARRAY[1, NULL])");
    assert!(result.is_always_null());
}

/// `= ANY` over a `NULL` member reads exactly `NULL`, never `TRUE`, when the
/// value is apart from the rest.
#[test]
fn an_any_over_a_null_member_reads_null_when_it_is_apart() {
    let listed = |value: i64| {
        Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(value)])),
        )
    };
    let context = Ctx { columns: vec![("x".to_string(), listed(2))], callees: Vec::new() };
    let result = read(context, "x = ANY (ARRAY[1, NULL])");
    assert!(result.is_always_null());
}

/// A cast to a bounded `numeric` may refuse a value past its precision.
#[test]
fn a_bounded_numeric_cast_may_raise() {
    assert!(read(context(), "1.5::numeric(4,1)").may_raise());
    assert!(!read(context(), "1.5::numeric").may_raise());
}

/// A typed `NULL` cast to another type may name a cast that does not exist.
#[test]
fn a_typed_null_cast_may_raise() {
    assert!(read(context(), "NULL::int::date").may_raise());
    assert!(!read(context(), "NULL::date").may_raise());
}

/// A non-strict cast may answer `NULL` where the conversion fails.
#[test]
fn a_try_cast_may_be_null() {
    let evaluation = read(context(), "TRY_CAST(1 AS date)");
    assert!(evaluation.may_be_null() && evaluation.may_raise());
}
