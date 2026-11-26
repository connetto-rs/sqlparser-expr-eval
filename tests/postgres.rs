//! PostgreSQL facts read through the public API, the guarantees the
//! backward pass places on the columns a condition names, and the relations
//! it keeps between two compared operands.

use std::{collections::HashSet, sync::Arc};

use sqlparser::dialect::PostgreSqlDialect;
use sqlparser_expr_eval::{
    Builtin, Callee, CalleeKind, Collation, EnumType, Evaluation, Guarantees, Interval, NullInput,
    NullResult, Order, OtherType, Outcomes, PostgreSqlOptions, QueryEvaluation, Relation, Rows,
    ScalarType, Subject, TextRules, Values, Volatility, evaluate, implied,
    sql_values::postgres::{IntegerType, PgNumeric, PgValue},
};

mod common;

use common::{Ctx, builtin_callee, column, parse};

impl Ctx {
    /// The context with the nested query reading as `query`.
    fn with_query(mut self, query: QueryEvaluation) -> Self {
        self.query = Some(query);
        self
    }

    /// The context with the collation `name` carrying `rules`.
    fn with_collation(mut self, name: &str, rules: TextRules) -> Self {
        self.collations.push((name.to_string(), rules));
        self
    }
}

/// Reads `sql` under PostgreSQL with no settings.
fn read(mut context: Ctx, sql: &str) -> Evaluation {
    let dialect = PostgreSqlDialect {};
    let options = PostgreSqlOptions::default();
    evaluate(&parse(sql), &dialect, &options, &mut context, &Guarantees::new())
        .expect("the context resolves")
}

/// Reads `sql` and returns the guarantee it places under the given outcomes.
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
    assert_eq!(*evaluation.ty(), ScalarType::Integer(IntegerType::Integer));
    assert!(evaluation.constant().is_some());
    assert!(!evaluation.may_be_null());
}

#[test]
fn an_int4_overflow_literal_is_typed_int8() {
    let evaluation = read(context(), "2147483648");
    assert_eq!(*evaluation.ty(), ScalarType::Integer(IntegerType::Bigint));
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
fn a_zero_divisor_raises_on_every_row() {
    for sql in ["id / 0", "id % 0", "id / (2 - 2)", "id / 0.0", "id * 1.0 / 0"] {
        assert!(read(context(), sql).always_raises(), "{sql}");
    }
    let comparison = read(context(), "id / 0 = 1");
    assert!(comparison.always_raises());
    assert!(!comparison.may_be_true());
}

#[test]
fn a_zero_divisor_answers_null_for_a_null_operand() {
    for sql in ["n / 0", "x / (n * 0)"] {
        let evaluation = read(integers(), sql);
        assert!(evaluation.may_be_null(), "{sql}");
        assert!(evaluation.may_raise(), "{sql}");
        assert!(evaluation.values().is_empty(), "{sql}");
    }
}

#[test]
fn a_zero_divisor_keeps_a_nan_dividend() {
    let numbers = || {
        Ctx::with_columns(vec![
            ("amount".into(), column(ScalarType::Numeric, false)),
            ("ratio".into(), column(ScalarType::Float, false)),
        ])
    };
    let nan = PgValue::Numeric(PgNumeric::NAN);
    let constant = read(numbers(), "CAST('NaN' AS numeric) / 0");
    assert_eq!(constant.constant(), Some(&nan));
    for sql in ["amount / 0", "amount % 0", "ratio / 0"] {
        let evaluation = read(numbers(), sql);
        assert!(evaluation.may_raise(), "{sql}");
        assert!(evaluation.may_be(&nan), "{sql}");
        assert!(!evaluation.may_be(&PgValue::Numeric(PgNumeric::from(1_i64))), "{sql}");
    }
}

#[test]
fn a_divisor_that_may_be_zero_leaves_the_quotient_open() {
    let evaluation = read(integers(), "x / y");
    assert!(evaluation.may_raise());
    assert!(evaluation.may_be(&PgValue::Integer(1)));
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

/// A catalog function the context declares to return `NULL` only on a
/// `NULL` argument, such as `now()`, which takes none.
fn now_callee() -> Callee {
    Callee::new(
        None,
        NullInput::Strict,
        NullResult::OnlyOnNullInput,
        Volatility::Stable,
        CalleeKind::Normal,
        catalog("timestamptz"),
        false,
        0,
    )
}

#[test]
fn a_function_returning_null_only_on_null_input_without_arguments_is_never_null() {
    for sql in ["now() IS NULL", "pg_catalog.now() IS NULL", "CURRENT_TIMESTAMP IS NULL"] {
        let context = context().with_callee("now", now_callee());
        assert_eq!(read(context, sql).outcomes(), Outcomes::FALSE, "{sql}");
    }
    let guarantees =
        restrict(context().with_callee("now", now_callee()), "now() IS NULL", Outcomes::TRUE);
    assert!(guarantees.is_impossible());
}

const INTEGER: ScalarType = ScalarType::Integer(IntegerType::Integer);

/// Integers `x`, `y` and `z` that are never `NULL` and a nullable integer
/// `n`.
fn integers() -> Ctx {
    Ctx::with_columns(vec![
        ("x".into(), column(INTEGER, false)),
        ("y".into(), column(INTEGER, false)),
        ("z".into(), column(INTEGER, false)),
        ("n".into(), column(INTEGER, true)),
    ])
}

/// The outcomes `sql` may take under `guarantees`.
fn outcomes_under(mut context: Ctx, sql: &str, guarantees: &Guarantees<String>) -> Outcomes {
    let dialect = PostgreSqlDialect {};
    let options = PostgreSqlOptions::default();
    evaluate(&parse(sql), &dialect, &options, &mut context, guarantees)
        .expect("the context resolves")
        .outcomes()
}

/// The outcomes `sql` may take with nothing guaranteed.
fn outcomes(context: Ctx, sql: &str) -> Outcomes {
    outcomes_under(context, sql, &Guarantees::new())
}

fn reference(name: &str) -> Subject<String> {
    Subject::Reference(name.to_string())
}

#[test]
fn contradicting_comparisons_of_one_pair_are_decided() {
    for (sql, expected) in [
        ("x < y AND y < x", Outcomes::FALSE),
        ("x < y AND x >= y", Outcomes::FALSE),
        ("x = y AND x <> y", Outcomes::FALSE),
        ("x <= y AND x >= y AND x <> y", Outcomes::FALSE),
        ("x < y OR x >= y", Outcomes::TRUE),
        ("x < n OR x >= n", Outcomes::NOT_FALSE),
        ("NOT (x < y) AND y > x", Outcomes::FALSE),
        ("x < x", Outcomes::FALSE),
        ("n < n", Outcomes::NOT_TRUE),
        ("n <> n", Outcomes::NOT_TRUE),
        ("n IS NOT NULL AND n <> n", Outcomes::FALSE),
        ("n <= n", Outcomes::NOT_FALSE),
        ("x BETWEEN y AND z AND z < y", Outcomes::FALSE),
    ] {
        assert_eq!(outcomes(integers(), sql), expected, "{sql}");
    }
}

#[test]
fn an_operand_computed_from_columns_keeps_no_relation() {
    // `y + z` reads columns, so it is no subject, and a negative `z` makes
    // the conjunction true.
    assert!(outcomes(integers(), "x < y AND y + z < x").truth);
}

#[test]
fn a_check_guarantee_decides_a_later_comparison() {
    let check = restrict(integers(), "x < y", Outcomes::NOT_FALSE);
    assert_eq!(outcomes_under(integers(), "x >= y", &check), Outcomes::FALSE);
    assert_eq!(outcomes_under(integers(), "y > x", &check), Outcomes::TRUE);
}

#[test]
fn a_relation_alone_narrows_no_values() {
    // `numeric` has no type bounds for the comparison to narrow towards.
    let numbers = || {
        Ctx::with_columns(vec![
            ("p".into(), column(ScalarType::Numeric, false)),
            ("q".into(), column(ScalarType::Numeric, false)),
        ])
    };
    let guarantees = restrict(numbers(), "p < q", Outcomes::TRUE);
    for name in ["p", "q"] {
        let evaluation = guarantees.reference(&name.to_string()).expect("the comparison names it");
        assert!(evaluation.number_range().is_unbounded(), "{name}");
    }
    assert_eq!(
        guarantees.relation(Order::Number, &reference("p"), &reference("q")),
        Relation { less: true, equal: false, greater: false }
    );
    assert_eq!(outcomes_under(numbers(), "q > p", &guarantees), Outcomes::TRUE);
}

#[test]
fn relations_compose_along_a_chain() {
    assert_eq!(outcomes(integers(), "x < y AND y < z AND z < x"), Outcomes::FALSE);
    let chain = restrict(integers(), "x < y AND y <= z", Outcomes::TRUE);
    assert_eq!(outcomes_under(integers(), "x < z", &chain), Outcomes::TRUE);
    let both_ways = restrict(integers(), "x <= y AND y <= x", Outcomes::TRUE);
    assert_eq!(outcomes_under(integers(), "x = y", &both_ways), Outcomes::TRUE);
}

/// What a comparison operator answers on two values.
type Holds = fn(&i32, &i32) -> bool;

/// The six comparison operators with what each answers on two values.
const OPERATORS: [(&str, Holds); 6] = [
    ("<", i32::lt),
    ("<=", i32::le),
    ("=", i32::eq),
    ("<>", i32::ne),
    (">=", i32::ge),
    (">", i32::gt),
];

/// The outcome of `left AND right`, or of `left OR right`, in three-valued
/// logic.
fn combine(left: Option<bool>, right: Option<bool>, and: bool) -> Option<bool> {
    match (left, right) {
        (Some(left), Some(right)) => Some(if and { left && right } else { left || right }),
        (Some(decided), None) | (None, Some(decided)) if decided != and => Some(decided),
        _ => None,
    }
}

#[test]
fn every_pair_of_comparisons_takes_the_outcomes_of_every_row() {
    for (left_null, right_null) in [(false, false), (true, false), (false, true), (true, true)] {
        let context = || {
            Ctx::with_columns(vec![
                ("a".into(), column(INTEGER, left_null)),
                ("b".into(), column(INTEGER, right_null)),
            ])
        };
        let values =
            |null: bool| [Some(1), Some(2), Some(3)].into_iter().chain(null.then_some(None));
        for (first, first_holds) in OPERATORS {
            for (second, second_holds) in OPERATORS {
                for and in [true, false] {
                    let mut expected = Outcomes::NONE;
                    for a in values(left_null) {
                        for b in values(right_null) {
                            let compare = |holds: Holds| a.zip(b).map(|(a, b)| holds(&a, &b));
                            expected = expected.union(
                                match combine(compare(first_holds), compare(second_holds), and) {
                                    Some(true) => Outcomes::TRUE,
                                    Some(false) => Outcomes::FALSE,
                                    None => Outcomes::NULL,
                                },
                            );
                        }
                    }
                    let connective = if and { "AND" } else { "OR" };
                    let sql = format!("a {first} b {connective} a {second} b");
                    assert_eq!(
                        outcomes(context(), &sql),
                        expected,
                        "{sql} with a nullable {left_null} and b nullable {right_null}"
                    );
                }
            }
        }
    }
}

/// Text columns `a`, `b` and `c` under the default collation and `named`
/// under another one, none of them ever `NULL`.
fn texts() -> Ctx {
    let named = TextRules { collation: Collation::Named(1), ..TextRules::DEFAULT };
    Ctx::with_columns(vec![
        ("a".into(), column(ScalarType::Text(TextRules::DEFAULT), false)),
        ("b".into(), column(ScalarType::Text(TextRules::DEFAULT), false)),
        ("c".into(), column(ScalarType::Text(TextRules::DEFAULT), false)),
        ("named".into(), column(ScalarType::Text(named), false)),
    ])
}

#[test]
fn text_relations_compose_only_under_one_collation() {
    assert_eq!(outcomes(texts(), "a < b AND b < c AND c < a"), Outcomes::FALSE);
    // `named < a` and `c <= named` compare under the named collation and
    // `a < c` under the default one, and two collations may order one pair
    // of strings either way.
    assert!(outcomes(texts(), "named < a AND a < c AND c <= named").truth);
    // Under "C" 'B' sorts before 'a', and under a linguistic default after.
    assert!(outcomes(texts(), r#"a COLLATE "C" < b AND b < a"#).truth);
}

/// Text columns `icu` and `bytes` under two different collations other than
/// the default, `nullable` under the collation of `bytes` and `plain` under
/// the default, and `unknown` under a collation the context cannot identify.
/// The collation `"und-x-icu"` is the one of `icu`, `"POSIX"` the one of
/// `bytes`, and `"C"` the built-in one.
fn conflicting_texts() -> Ctx {
    let icu = TextRules { collation: Collation::Named(0), ..TextRules::DEFAULT };
    let bytes = TextRules {
        identity_equality: true,
        byte_order: true,
        collation: Collation::Named(1),
        explicit: false,
    };
    Ctx::with_columns(vec![
        ("icu".into(), column(ScalarType::Text(icu), false)),
        ("bytes".into(), column(ScalarType::Text(bytes), false)),
        ("nullable".into(), column(ScalarType::Text(bytes), true)),
        ("plain".into(), column(ScalarType::Text(TextRules::DEFAULT), false)),
        ("unknown".into(), column(ScalarType::Text(TextRules::UNKNOWN), false)),
    ])
    .with_collation("und-x-icu", icu)
    .with_collation("POSIX", bytes)
    .with_collation("C", TextRules::C)
}

#[test]
fn comparing_two_implicit_collations_raises() {
    // PostgreSQL cannot determine the collation of a comparison between two
    // different implicit collations and raises on every pair of values.
    for sql in [
        "icu < bytes AND bytes < icu",
        "icu = bytes",
        "icu <> bytes",
        "icu BETWEEN bytes AND bytes",
        "icu IN (bytes)",
        "icu IS DISTINCT FROM bytes",
    ] {
        let evaluation = read(conflicting_texts(), sql);
        assert!(evaluation.may_raise(), "{sql}");
        assert!(!evaluation.may_be_true() && !evaluation.may_be_false(), "{sql}");
    }
    // An array whose members the evaluator does not list may hold none to
    // compare, but any member it holds raises.
    assert!(read(conflicting_texts(), "icu = ANY (ARRAY[bytes])").may_raise());
    // A nullable operand raises on the rows where it holds a value and
    // reads `NULL` on the others.
    let evaluation = read(conflicting_texts(), "icu < nullable");
    assert!(evaluation.may_raise());
    assert_eq!(evaluation.outcomes(), Outcomes::NULL);
    // The default collation and an untyped literal each settle the
    // comparison's collation, and an unknown collation may be either side's.
    for sql in ["icu < plain", "icu < 'x'", "icu < unknown"] {
        assert!(!read(conflicting_texts(), sql).may_raise(), "{sql}");
    }
}

#[test]
fn sql_value_functions_take_their_type_and_never_raise() {
    let name = || ScalarType::Text(TextRules::C);
    for (sql, ty, null) in [
        ("current_user", name(), false),
        ("CURRENT_ROLE", name(), false),
        ("session_user", name(), false),
        ("user", name(), false),
        ("current_catalog", name(), false),
        ("current_schema", name(), true),
        ("system_user", ScalarType::Text(TextRules::DEFAULT), true),
        ("current_date", ScalarType::Date, false),
        ("current_time", catalog("timetz"), false),
        ("current_time(3)", catalog("timetz"), false),
        ("current_timestamp", catalog("timestamptz"), false),
        ("current_timestamp(3)", catalog("timestamptz"), false),
        ("localtime", catalog("time"), false),
        ("localtime(3)", catalog("time"), false),
        ("localtimestamp", ScalarType::Timestamp, false),
        ("localtimestamp(3)", ScalarType::Timestamp, false),
    ] {
        let evaluation = read(context(), sql);
        assert_eq!(*evaluation.ty(), ty, "{sql}");
        assert_eq!(evaluation.may_be_null(), null, "{sql}");
        assert!(!evaluation.may_raise(), "{sql}");
    }
}

#[test]
fn a_text_column_compares_with_the_current_user_without_raising() {
    let compared = read(context(), "x = current_user");
    assert!(!compared.may_raise());
    let admitted = read(context(), "x = current_user OR true");
    assert_eq!(admitted.outcomes(), Outcomes::TRUE);
    assert!(!admitted.may_raise());
}

#[test]
fn the_current_user_raises_against_another_named_collation() {
    // `name` carries the "C" collation, which conflicts with any other
    // implicit collation but the default.
    assert!(read(conflicting_texts(), "icu = current_user").may_raise());
    assert!(read(conflicting_texts(), "current_user < bytes").may_raise());
    assert!(!read(conflicting_texts(), "plain = current_user").may_raise());
}

#[test]
fn an_explicit_collation_settles_the_comparison() {
    // An explicit collation wins over the implicit one of the other operand,
    // so the comparison has a collation and never raises, even under a
    // collation the context cannot identify.
    for sql in [
        r#"icu < bytes COLLATE "C""#,
        r#"icu COLLATE "C" < bytes"#,
        r#"icu COLLATE "und-x-icu" = bytes"#,
        r#"icu < bytes COLLATE "unlisted""#,
    ] {
        let evaluation = read(conflicting_texts(), sql);
        assert!(!evaluation.may_raise(), "{sql}");
        assert_eq!(evaluation.outcomes(), Outcomes { null: false, ..Outcomes::ANY }, "{sql}");
    }
    let evaluation = read(conflicting_texts(), r#"(icu < bytes COLLATE "C") IS NOT NULL"#);
    assert_eq!((evaluation.outcomes(), evaluation.may_raise()), (Outcomes::TRUE, false));
    // PostgreSQL refuses two different explicit collations.
    let mismatch = r#"icu COLLATE "und-x-icu" < bytes COLLATE "C""#;
    assert!(read(conflicting_texts(), mismatch).always_raises());
    // The comparison runs under the explicit collation, which orders by
    // byte here although the other side carries the default collation.
    let ordered = read(conflicting_texts(), r#"'B' COLLATE "C" < CAST('a' AS TEXT)"#);
    assert_eq!((ordered.outcomes(), ordered.may_raise()), (Outcomes::TRUE, false));
}

#[test]
fn a_collation_applies_to_text_alone() {
    let null = read(conflicting_texts(), r#"NULL COLLATE "C""#);
    assert!(null.is_always_null());
    assert!(matches!(null.ty(), ScalarType::Text(rules) if rules.explicit));
    let literal = read(conflicting_texts(), r#"'a' COLLATE "C""#);
    assert_eq!(literal.constant(), Some(&PgValue::Text("a".into())));
    assert!(!literal.may_raise());
    assert!(read(conflicting_texts(), r#"1 COLLATE "C""#).may_raise());
}

#[test]
fn a_deciding_constant_folds_the_chain_before_it_runs() {
    // PostgreSQL's planner reduces a chain holding `true` under `OR`, or
    // `false` under `AND`, to that constant, so the other terms never run.
    for (sql, decided) in [
        ("icu < bytes OR true", Outcomes::TRUE),
        ("true OR icu < bytes", Outcomes::TRUE),
        ("icu < bytes OR 1 = 1", Outcomes::TRUE),
        ("icu < bytes OR (plain < icu OR NOT false)", Outcomes::TRUE),
        ("icu < bytes AND false", Outcomes::FALSE),
    ] {
        let evaluation = read(conflicting_texts(), sql);
        assert_eq!((evaluation.outcomes(), evaluation.may_raise()), (decided, false), "{sql}");
    }
    // The planner folds every constant before the deciding one, and a
    // constant that raises raises there, but never reaches one after it.
    assert!(read(context(), "1 / 0 = 1 OR true").may_raise());
    assert!(read(context(), "id > 1 / 0 OR true").may_raise());
    assert!(!read(context(), "true OR 1 / 0 = 1").may_raise());
    // A term that reads a column is not folded, whatever it evaluates to.
    assert!(read(conflicting_texts(), "icu < bytes OR plain = plain").may_raise());
}

/// An enum type `name` declaring `project`, `note` and `folder` in that
/// order.
fn kind_type(name: &str) -> ScalarType {
    ScalarType::Enum(Arc::new(EnumType::new(name, ["project", "note", "folder"])))
}

/// A context typing `resource_kind` and `mirror_kind`, two enums declaring
/// the same labels, with a non-null `kind` of `resource_kind` holding
/// `labels`, every label when `labels` is `None`, and a non-null `mirror`
/// of `mirror_kind`.
fn kinds(labels: Option<&[&str]>) -> Ctx {
    let kind = match labels {
        Some(labels) => {
            let labels = labels.iter().map(|label| PgValue::Text((*label).into())).collect();
            Evaluation::new(kind_type("resource_kind"), false, false, Values::Listed(labels))
        }
        None => column(kind_type("resource_kind"), false),
    };
    let mut context = Ctx::with_columns(vec![
        ("kind".into(), kind),
        ("mirror".into(), column(kind_type("mirror_kind"), false)),
    ]);
    context.types = vec![
        ("resource_kind".into(), kind_type("resource_kind")),
        ("mirror_kind".into(), kind_type("mirror_kind")),
    ];
    context
}

#[test]
fn an_enum_column_narrowed_to_one_label_decides_an_equality() {
    let note = || kinds(Some(&["note"]));
    let other = read(note(), "kind = 'project'");
    assert_eq!((other.outcomes(), other.may_raise()), (Outcomes::FALSE, false));
    let same = read(note(), "kind = 'note'");
    assert_eq!((same.outcomes(), same.may_raise()), (Outcomes::TRUE, false));
    let unnarrowed = read(kinds(None), "kind = 'project'");
    assert_eq!(unnarrowed.outcomes(), Outcomes { null: false, ..Outcomes::ANY });
    assert!(!unnarrowed.may_raise());
}

#[test]
fn an_enum_check_narrows_the_column_to_its_labels() {
    for check in ["kind = 'project'", "kind IN ('project', 'folder')", "kind <> 'note'"] {
        let guarantees = restrict(kinds(None), check, Outcomes::NOT_FALSE);
        let note = outcomes_under(kinds(None), "kind = 'note'", &guarantees);
        assert_eq!(note, Outcomes::FALSE, "{check}");
    }
}

#[test]
fn an_enum_cast_reads_a_declared_label_and_raises_on_any_other() {
    let note = read(kinds(None), "'note'::resource_kind");
    assert_eq!(note.constant(), Some(&PgValue::Text("note".into())));
    assert_eq!(*note.ty(), kind_type("resource_kind"));
    assert!(read(kinds(None), "'bogus'::resource_kind").always_raises());
    assert!(read(kinds(None), "kind = 'bogus'").always_raises());
    let printed = read(kinds(Some(&["folder"])), "kind::text");
    assert_eq!(printed.constant(), Some(&PgValue::Text("folder".into())));
    assert!(read(kinds(None), "'note'::resource_kind::mirror_kind").always_raises());
    assert!(read(kinds(None), "1::resource_kind").always_raises());
}

#[test]
fn enum_labels_order_by_declaration() {
    // `project` sorts after `folder` as text and before it as a label.
    let project = || kinds(Some(&["project"]));
    assert_eq!(read(project(), "kind < 'folder'").outcomes(), Outcomes::TRUE);
    assert_eq!(read(project(), "kind > 'note'").outcomes(), Outcomes::FALSE);
    let narrowed = read(kinds(Some(&["note", "folder"])), "kind >= 'note'");
    assert_eq!(narrowed.outcomes(), Outcomes::TRUE);
}

#[test]
fn an_enum_compares_with_no_other_type() {
    // PostgreSQL has no operator between two enums or an enum and text, and
    // no implicit cast to reach one.
    assert!(read(kinds(None), "kind = mirror").always_raises());
    assert!(read(kinds(None), "kind = 'note'::text").always_raises());
    assert!(read(kinds(None), "kind = 'note'::resource_kind").outcomes().truth);
}

#[test]
fn an_enum_cast_raises_only_on_text_that_may_hold_no_label() {
    let mut context = kinds(None);
    context.columns.push(("name".into(), column(ScalarType::Text(TextRules::DEFAULT), false)));
    let label = read(context, "name::resource_kind");
    assert!(label.may_raise() && !label.always_raises());
    // Any type the evaluator does not model may declare a cast with an enum.
    let unmodelled = read(kinds(None), "kind::unmodelled_type");
    assert!(!unmodelled.always_raises());
    assert_eq!(unmodelled.constant(), None);
}

#[test]
fn an_enum_too_large_to_list_stays_undecided() {
    let wide = ScalarType::Enum(Arc::new(EnumType::new(
        "wide_kind",
        (0..65).map(|label| format!("label_{label}")),
    )));
    let wide_context = || {
        let mut context = Ctx::with_columns(vec![("kind".into(), column(wide.clone(), false))]);
        context.types = vec![("wide_kind".into(), wide.clone())];
        context
    };
    let equality = read(wide_context(), "kind = 'label_0'");
    assert_eq!(equality.outcomes(), Outcomes { null: false, ..Outcomes::ANY });
    assert!(!read(wide_context(), "kind::wide_kind").may_raise());
}

/// The unmodelled built-in type `name`.
fn catalog(name: &'static str) -> ScalarType {
    ScalarType::Other(Some(OtherType::catalog(name)))
}

/// A context with two non-null `uuid` columns, `id` and `other_id`, a
/// non-null `jsonb` `doc`, a nullable `boolean` `flag`, a non-null
/// `geometry` `shape`, and non-null `bpchar`, `bit`, `timestamptz` and
/// `interval` columns named after their types.
fn identified() -> Ctx {
    Ctx::with_columns(vec![
        ("id".into(), column(catalog("uuid"), false)),
        ("other_id".into(), column(catalog("uuid"), false)),
        ("doc".into(), column(catalog("jsonb"), false)),
        ("flag".into(), column(ScalarType::Boolean, true)),
        ("shape".into(), column(ScalarType::Other(Some(OtherType::new("geometry"))), false)),
        ("pad".into(), column(catalog("bpchar"), false)),
        ("bits".into(), column(catalog("bit"), false)),
        ("moment".into(), column(catalog("timestamptz"), false)),
        ("span".into(), column(catalog("interval"), false)),
    ])
}

#[test]
fn one_unmodelled_type_hashes_alike_however_built() {
    let types: HashSet<ScalarType> = [
        catalog("uuid"),
        ScalarType::Other(Some(OtherType::new(String::from("uuid")))),
        catalog("jsonb"),
    ]
    .into_iter()
    .collect();
    assert_eq!(types.len(), 2);
}

#[test]
fn an_unmodelled_type_keeps_its_identity_where_values_meet() {
    for sql in ["COALESCE(id, other_id)", "CASE WHEN flag THEN id ELSE other_id END"] {
        let met = read(identified(), sql);
        assert_eq!(*met.ty(), catalog("uuid"), "{sql}");
        assert!(!met.may_raise(), "{sql}");
    }
    // PostgreSQL matches no `uuid` with a `jsonb`.
    for sql in ["COALESCE(id, doc)", "CASE WHEN flag THEN id ELSE doc END"] {
        let met = read(identified(), sql);
        assert_eq!(*met.ty(), ScalarType::Other(None), "{sql}");
        assert!(met.may_raise(), "{sql}");
    }
    assert_eq!(*read(identified(), "'10.0.0.1'::INET").ty(), catalog("inet"));
}

#[test]
fn a_cast_of_an_unmodelled_value_to_its_own_type_never_raises() {
    for sql in [
        "id::uuid",
        "id::pg_catalog.UUID",
        "CAST(shape AS geometry)",
        "pad::bpchar",
        "moment::timestamptz(3)",
        "span::interval(0)",
    ] {
        assert!(!read(identified(), sql).may_raise(), "{sql}");
    }
    // Another type, or a length the type may refuse, may raise.
    for sql in [
        "doc::uuid",
        "id::jsonb",
        "shape::geometry(Point, 4326)",
        "pad::char(0)",
        "bits::bit(0)",
        "bits::varbit(0)",
    ] {
        assert!(read(identified(), sql).may_raise(), "{sql}");
    }
}

#[test]
fn an_operator_over_a_constant_raising_operand_always_raises() {
    for sql in ["1 / 0 > n", "n < 1 / 0", "(1 / 0) + n > 0", "NULL::int > 1 / 0", "n * 0 + 1 / 0"] {
        assert!(read(integers(), sql).always_raises(), "{sql}");
    }
}

#[test]
fn an_operator_over_a_raising_operand_beside_a_column_always_raises() {
    for sql in [
        "x / 0 > n",
        "n <> x / 0",
        "x / 0 + n",
        "x / 0 + n + 1",
        "x / 0 > n + 1",
        "x / 0 > -n",
        "x / 0 || CAST(n AS text)",
        "CAST(n AS text) || x / 0",
    ] {
        assert!(read(integers(), sql).always_raises(), "{sql}");
    }
}

#[test]
fn an_operator_beside_a_null_constant_folds_to_null() {
    for sql in [
        "x / 0 > NULL::int",
        "NULL::int > x / 0",
        "x / 0 + NULL::int",
        "x / 0 || NULL::text",
        "(x / 0 + NULL) > x / 0",
        "x / 0 > CASE WHEN true THEN NULL ELSE n END",
    ] {
        let evaluation = read(integers(), sql);
        assert!(evaluation.may_be_null(), "{sql}");
        assert!(!evaluation.always_raises(), "{sql}");
    }
}

#[test]
fn an_operator_beside_a_parameter_planned_as_a_constant_folds_to_null() {
    let parameters = || {
        let mut context = integers();
        context.plan_constants = vec!["n".into()];
        context
    };
    for sql in ["x / 0 > n", "x / 0 + n * 2"] {
        let evaluation = read(parameters(), sql);
        assert!(evaluation.may_be_null(), "{sql}");
        assert!(!evaluation.always_raises(), "{sql}");
    }
    assert!(read(parameters(), "1 / 0 > n").always_raises());
}

#[test]
fn an_operator_beside_an_operand_without_outcome_has_none() {
    let mut context = integers();
    context.columns.push(("none".into(), Evaluation::impossible(INTEGER)));
    assert!(read(context, "1 / 0 + none").is_impossible());
}
