//! PostgreSQL facts read through the public API, the guarantees the
//! backward pass places on the columns a condition names, and the relations
//! it keeps between two compared operands.

use sqlparser::dialect::PostgreSqlDialect;
use sqlparser_expr_eval::{
    Builtin, Callee, Collation, Evaluation, Guarantees, Interval, Order, Outcomes,
    PostgreSqlOptions, QueryEvaluation, Relation, Rows, ScalarType, Subject, TextRules, Values,
    evaluate, implied,
    sql_values::postgres::{IntegerType, PgValue},
};

mod common;

use common::{Ctx, builtin_callee, column, parse};

impl Ctx {
    /// The context with the nested query reading as `query`.
    fn with_query(mut self, query: QueryEvaluation) -> Self {
        self.query = Some(query);
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
fn conflicting_texts() -> Ctx {
    let icu = TextRules { collation: Collation::Named(0), ..TextRules::DEFAULT };
    let bytes =
        TextRules { identity_equality: true, byte_order: true, collation: Collation::Named(1) };
    Ctx::with_columns(vec![
        ("icu".into(), column(ScalarType::Text(icu), false)),
        ("bytes".into(), column(ScalarType::Text(bytes), false)),
        ("nullable".into(), column(ScalarType::Text(bytes), true)),
        ("plain".into(), column(ScalarType::Text(TextRules::DEFAULT), false)),
        ("unknown".into(), column(ScalarType::Text(TextRules::UNKNOWN), false)),
    ])
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
