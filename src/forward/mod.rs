//! The forward pass: what an expression evaluates to over every row.

pub(crate) mod fold;
mod functions;
mod literals;

use alloc::collections::BTreeSet;
use core::cell::Cell;

use sqlparser::ast::Expr;

use super::{
    context::Context,
    dialect::EvalDialect,
    domain::{Evaluation, Values},
    guarantees::Guarantees,
    scalar::ScalarType,
    support,
};

/// How many times one top-level evaluation reads the right operand of an
/// `AND` or an `OR` once per outcome of its left operand, which correlates
/// the two sides and keeps the work bounded.
pub(super) const PATH_BUDGET: u32 = 64;

/// How deep the passes may recurse before a node answers unknown, so that a
/// deep nesting keeps the stack inside a small thread. A parser-built tree
/// stays under the parser's own recursion limit, so this only bounds
/// hand-built trees. Measured as the largest depth whose frames fit a
/// default 2 MiB debug thread on the widest frame, the unary-minus chain
/// whose `negate` reads its values.
pub(super) const DEPTH_LIMIT: u32 = 32;

/// Reads `expression` through `context` into what it can evaluate to,
/// meeting the result at every node with the guarantee that node's subject
/// satisfies under `guarantees`, under the dialect's behaviour and
/// settings.
///
/// # Errors
///
/// Returns the context's error when a reference, a call or a subquery fails
/// to resolve.
pub fn evaluate<D: EvalDialect, C: Context>(
    expression: &Expr,
    dialect: &D,
    options: &D::Options,
    context: &mut C,
    guarantees: &Guarantees<C::Reference>,
) -> Result<Evaluation, C::Error> {
    evaluate_within(expression, dialect, options, context, guarantees, &Cell::new(PATH_BUDGET))
}

/// [`evaluate`] drawing on the path budget of an enclosing pass.
pub(super) fn evaluate_within<D: EvalDialect, C: Context>(
    expression: &Expr,
    dialect: &D,
    options: &D::Options,
    context: &mut C,
    guarantees: &Guarantees<C::Reference>,
    budget: &Cell<u32>,
) -> Result<Evaluation, C::Error> {
    let mut state = State { dialect, options, context, guarantees, budget, depth: 0 };
    forward(&mut state, expression)
}

/// What the forward pass walks over: the dialect and its settings, the
/// context, the guarantees it meets at every node and the path budget it
/// shares with the backward pass.
pub(super) struct State<'a, D: EvalDialect, C: Context> {
    pub(super) dialect: &'a D,
    pub(super) options: &'a D::Options,
    pub(super) context: &'a mut C,
    pub(super) guarantees: &'a Guarantees<C::Reference>,
    pub(super) budget: &'a Cell<u32>,
    /// How deep the pass has walked, for the depth limit.
    pub(super) depth: u32,
}

/// The forward evaluation of one node, meeting its subject's guarantee.
fn forward<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
) -> Result<Evaluation, C::Error> {
    let mut expression = expression;
    while let Expr::Nested(inner) = expression {
        expression = inner;
    }
    if state.depth >= DEPTH_LIMIT {
        // Past the depth limit the node answers unknown, which is a safe
        // over-approximation.
        return Ok(Evaluation::unknown(ScalarType::Unknown));
    }
    if let Some(resolved) = support::resolve_reference(state.context, expression)? {
        let evaluation = resolved.evaluation;
        return Ok(match state.guarantees.reference(&resolved.reference) {
            Some(guarantee) => evaluation.meet(guarantee),
            None => evaluation,
        });
    }
    // An error abandons the pass, so only the success path restores the
    // depth.
    state.depth += 1;
    let evaluation = evaluate_node(state, expression)?;
    state.depth -= 1;
    Ok(match state.guarantees.expression(expression) {
        Some(guarantee) => evaluation.meet(guarantee),
        None => evaluation,
    })
}

/// The forward evaluation of a non-reference, non-nested node, nested
/// expressions being unwound by [`forward`].
fn evaluate_node<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
) -> Result<Evaluation, C::Error> {
    Ok(match expression {
        Expr::Value(value) => literals::value(value),
        Expr::TypedString(typed) => literals::typed_string(state.context, typed),
        Expr::UnaryOp { op, expr } => fold::unary(state, *op, expr)?,
        Expr::BinaryOp { left, op, right } => fold::binary(state, left, op, right)?,
        Expr::Cast { kind, expr, data_type, .. } => literals::cast(state, kind, expr, data_type)?,
        Expr::Function(call) => functions::call(state, call)?,
        Expr::Case { operand, conditions, else_result, .. } => {
            functions::case(state, operand.as_deref(), conditions, else_result.as_deref())?
        }
        Expr::IsNull(expr) => functions::is_null(state, expr, false)?,
        Expr::IsNotNull(expr) => functions::is_null(state, expr, true)?,
        Expr::IsUnknown(expr) => functions::is_unknown(state, expr, false)?,
        Expr::IsNotUnknown(expr) => functions::is_unknown(state, expr, true)?,
        Expr::IsTrue(expr) => functions::is_true_false(state, expr, true, false)?,
        Expr::IsNotTrue(expr) => functions::is_true_false(state, expr, true, true)?,
        Expr::IsFalse(expr) => functions::is_true_false(state, expr, false, false)?,
        Expr::IsNotFalse(expr) => functions::is_true_false(state, expr, false, true)?,
        Expr::IsDistinctFrom(left, right) => functions::is_distinct(state, left, right, false)?,
        Expr::IsNotDistinctFrom(left, right) => functions::is_distinct(state, left, right, true)?,
        Expr::In { expr, target, negated, .. } => {
            functions::in_target(state, expr, target, *negated)?
        }
        Expr::Between { expr, negated, low, high } => {
            fold::between(state, expr, low, high, *negated)?
        }
        Expr::Like { negated, expr, pattern, .. }
        | Expr::ILike { negated, expr, pattern, .. }
        | Expr::SimilarTo { negated, expr, pattern, .. } => {
            functions::like(state, expr, pattern, *negated)?
        }
        Expr::AnyOp { left, compare_op, right, .. } => {
            functions::any_all(state, left, compare_op, right, true)?
        }
        Expr::AllOp { left, compare_op, right, .. } => {
            functions::any_all(state, left, compare_op, right, false)?
        }
        Expr::Exists { subquery, negated } => functions::exists(state, subquery, *negated)?,
        Expr::Subquery(query) => functions::scalar_subquery(state, query)?,
        Expr::Array(array) => {
            let mut values = BTreeSet::new();
            let mut has_null = false;
            let mut error = false;
            for element in &array.elem {
                let evaluation = forward(state, element)?;
                error |= evaluation.may_raise();
                match evaluation.constant() {
                    Some(datum) => {
                        values.insert(datum.clone());
                    }
                    None if evaluation.is_always_null() => has_null = true,
                    None => {
                        // A non-constant member leaves the array's members
                        // unknown.
                        return Ok(Evaluation::unknown(ScalarType::Unknown));
                    }
                }
            }
            // A `NULL` member nulls every `ANY` and `ALL` reading of the
            // array, so it stands for the array's `NULL` possibility.
            Evaluation::new(ScalarType::Unknown, has_null, error, Values::Listed(values))
        }
        _ => Evaluation::unknown(ScalarType::Unknown),
    })
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;

    use sql_values::{IntegerType, PgValue};
    use sqlparser::dialect::PostgreSqlDialect;

    use super::{
        super::{
            context::{Builtin, Callee, NullInput, Rows},
            domain::{Evaluation, Unlisted, Values},
            guarantees::Guarantees,
            postgres::PostgreSqlOptions,
            scalar::{ScalarType, TextRules},
            test_context::{self, ColumnContext, read, read_default},
        },
        evaluate,
    };

    #[test]
    fn integer_overflow_always_raises() {
        let evaluation = read(ColumnContext::new(), "2147483647 + 1");
        assert!(evaluation.always_raises());
    }

    #[test]
    fn division_by_zero_raises() {
        let evaluation = read(ColumnContext::new(), "1 / 0");
        assert!(evaluation.always_raises());
    }

    #[test]
    fn integer_division_folds() {
        let evaluation = read(ColumnContext::new(), "7 / 2");
        assert_eq!(evaluation.constant(), Some(&PgValue::Integer(3)));
    }

    #[test]
    fn numeric_division_folds_at_scale() {
        let evaluation = read(ColumnContext::new(), "7.0 / 2");
        let constant = evaluation.constant().expect("a numeric division is a constant");
        let PgValue::Numeric(numeric) = constant else { panic!("the constant is numeric") };
        assert_eq!(numeric.to_string(), "3.5000000000000000");
    }

    #[test]
    fn a_left_deep_division_chain_folds_pairwise() {
        let evaluation = read(ColumnContext::new(), "8 / 4 / 2");
        assert_eq!(evaluation.constant(), Some(&PgValue::Integer(1)));
    }

    #[test]
    fn a_left_deep_concat_chain_folds_pairwise() {
        let evaluation = read(ColumnContext::new(), "'a' || 'b' || 'c'");
        assert_eq!(evaluation.constant(), Some(&PgValue::Text("abc".to_string())));
    }

    #[test]
    fn a_comparison_inside_an_additive_spine_stops_the_walk() {
        let context = ColumnContext::new().with_column(
            "x",
            Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                true,
                false,
                Values::Unlisted(Unlisted::any()),
            ),
        );
        // PostgreSQL has no `+` for booleans, so the pair raises
        let evaluation = read(context, "(x < 5) + 3");
        assert!(evaluation.may_raise());
    }

    #[test]
    fn arithmetic_with_an_untyped_side_keeps_the_unknown_reading() {
        let context =
            ColumnContext::new().with_column("x", Evaluation::unknown(ScalarType::Unknown));
        let evaluation = read(context, "x + 1");
        assert!(evaluation.is_uninformative());
    }

    #[test]
    fn a_concat_with_an_untyped_side_keeps_the_unknown_reading() {
        let context =
            ColumnContext::new().with_column("x", Evaluation::unknown(ScalarType::Unknown));
        let evaluation = read(context, "x || 'a'");
        assert!(evaluation.is_uninformative());
    }

    #[test]
    fn date_adding_a_listed_numeric_shift_cannot_fold() {
        let date = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Date(
                sql_values::parse_date("2000-01-01").unwrap(),
            )])),
        );
        let days = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Numeric(sql_values::PgNumeric::from(3))])),
        );
        let context = ColumnContext::new().with_column("d", date).with_column("n", days);
        // A numeric shift stands for the same day count but cannot be folded
        // against a listed date, so the shift stays open
        let evaluation = read(context, "d + n");
        assert_eq!(evaluation.ty(), ScalarType::Date);
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    }

    #[test]
    fn concat_operator_with_null_is_always_null() {
        let evaluation = read(ColumnContext::new(), "'a' || NULL");
        assert!(evaluation.is_always_null());
    }

    /// An array holding a non-constant member leaves its members unknown.
    #[test]
    fn an_array_with_a_non_constant_member_is_unknown() {
        let evaluation = read(
            ColumnContext::new()
                .with_column("x", Evaluation::unknown(ScalarType::Text(TextRules::DEFAULT))),
            "ARRAY[x]",
        );
        assert_eq!(evaluation.ty(), ScalarType::Unknown);
        assert!(!matches!(evaluation.values(), Values::Listed(_)));
    }

    #[test]
    fn concat_call_skips_nulls() {
        let mut callee =
            test_context::builtin_callee(Builtin::Concat, ScalarType::Text(TextRules::DEFAULT));
        callee.null_input = NullInput::CalledOnNull(Vec::new());
        let evaluation =
            read(ColumnContext::new().with_callee("concat", callee), "concat('a', NULL)");
        assert_eq!(evaluation.constant(), Some(&PgValue::Text("a".to_string())));
    }

    #[test]
    fn nullif_equal_arguments_is_null() {
        let evaluation = read(ColumnContext::new(), "nullif(1, 1)");
        assert!(evaluation.is_always_null());
    }

    #[test]
    fn or_with_null_keeps_the_truth() {
        let evaluation = read(ColumnContext::new(), "true OR NULL");
        assert!(evaluation.may_be_true());
        assert!(!evaluation.may_be_false());
        assert!(!evaluation.may_be_null());
    }

    #[test]
    fn and_with_null_keeps_the_falsity() {
        let evaluation = read(ColumnContext::new(), "false AND NULL");
        assert!(evaluation.may_be_false());
        assert!(!evaluation.may_be_true());
        assert!(!evaluation.may_be_null());
    }

    #[test]
    fn case_skips_the_null_condition() {
        let evaluation = read(ColumnContext::new(), "CASE WHEN NULL THEN 1 ELSE 2 END");
        assert_eq!(evaluation.constant(), Some(&PgValue::Integer(2)));
    }

    /// A context with an integer `id`, nullable when `null`.
    fn id_column(null: bool) -> ColumnContext {
        let id = ScalarType::Integer(IntegerType::Integer);
        let values = Values::Unlisted(super::super::domain::Unlisted::any());
        ColumnContext::new().with_column("id", Evaluation::new(id, null, false, values))
    }

    #[test]
    fn not_in_a_null_member_never_true() {
        let evaluation = read(id_column(true), "id NOT IN (1, NULL)");
        assert!(!evaluation.may_be_true());
    }

    #[test]
    fn in_a_null_member_never_true() {
        let evaluation = read(id_column(true), "id IN (NULL)");
        assert!(!evaluation.may_be_true());
    }

    #[test]
    fn equality_with_null_is_always_null() {
        let evaluation = read(id_column(true), "id = NULL");
        assert!(evaluation.is_always_null());
    }

    #[test]
    fn is_null_of_a_never_null_column_is_false() {
        let evaluation = read(id_column(false), "id IS NULL");
        assert!(evaluation.may_be_false());
        assert!(!evaluation.may_be_true());
    }

    #[test]
    fn standard_concat_stays_unknown() {
        let evaluation = read_default(ColumnContext::new(), "'a' || 'b'");
        let Values::Unlisted(unlisted) = evaluation.values() else { panic!("stays unlisted") };
        assert_eq!(unlisted.length.high, core::ops::Bound::Unbounded);
        assert!(unlisted.excluded.is_empty());
    }

    #[test]
    fn standard_arithmetic_stays_unknown() {
        let evaluation = read_default(ColumnContext::new(), "1 + 2");
        assert!(
            evaluation.values().is_empty() || {
                matches!(evaluation.values(), Values::Unlisted(_))
            }
        );
    }

    #[test]
    fn standard_comparison_of_boolean_literals() {
        let evaluation = read_default(ColumnContext::new(), "1 = 2");
        assert!(evaluation.may_be_false());
        assert!(!evaluation.may_be_true());
        assert!(!evaluation.may_be_null());
    }

    #[test]
    fn an_array_cast_folds_its_elements_to_the_element_type() {
        let evaluation = read(ColumnContext::new(), "ARRAY['read', 'write']::text[]");
        let Values::Listed(listed) = evaluation.values() else { panic!("the cast is listed") };
        let expected: BTreeSet<PgValue> =
            [PgValue::Text("read".to_string()), PgValue::Text("write".to_string())].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_bounded_varchar_array_cast_truncates_its_elements() {
        let evaluation = read(ColumnContext::new(), "ARRAY['read', 'write']::varchar(2)[]");
        let Values::Listed(listed) = evaluation.values() else { panic!("the cast is listed") };
        let expected: BTreeSet<PgValue> =
            [PgValue::Text("re".to_string()), PgValue::Text("wr".to_string())].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn standard_division_by_zero_may_be_null_and_may_raise() {
        let evaluation = read_default(ColumnContext::new(), "1 / 0");
        assert!(evaluation.may_be_null());
        assert!(evaluation.may_raise());
    }

    #[test]
    fn standard_modulo_by_zero_may_be_null_and_may_raise() {
        let evaluation = read_default(ColumnContext::new(), "1 % 0");
        assert!(evaluation.may_be_null());
        assert!(evaluation.may_raise());
    }

    #[test]
    fn standard_arithmetic_may_raise_and_folds_nothing() {
        let evaluation = read_default(ColumnContext::new(), "1 + 2");
        assert!(evaluation.may_raise());
        assert!(!evaluation.may_be_null());
        assert!(
            evaluation.values().is_empty() || matches!(evaluation.values(), Values::Unlisted(_))
        );
    }

    #[test]
    fn standard_division_by_a_certain_divisor_stays_strict() {
        let evaluation = read_default(ColumnContext::new(), "1 / 2");
        assert!(!evaluation.may_be_null());
        assert!(evaluation.may_raise());
    }

    #[test]
    fn standard_text_literal_against_a_number_stays_unknown() {
        let evaluation = read_default(ColumnContext::new(), "'1' = 1");
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
    }

    #[test]
    fn standard_text_literals_compare_unknown() {
        let evaluation = read_default(ColumnContext::new(), "'a' = 'A'");
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
    }

    #[test]
    fn standard_boolean_literals_compare_by_value() {
        let evaluation = read_default(ColumnContext::new(), "true = false");
        assert!(evaluation.may_be_false());
        assert!(!evaluation.may_be_true());
        assert!(!evaluation.may_be_null());
    }

    #[test]
    fn standard_date_literals_compare_unknown() {
        let evaluation =
            read_default(ColumnContext::new(), "DATE '2026-01-01' < DATE '2026-01-02'");
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
    }

    #[test]
    fn a_called_on_null_callee_keeps_the_values() {
        let strict = test_context::strict_callee(ScalarType::Text(TextRules::DEFAULT));
        let called_on_null =
            Callee { null_input: NullInput::CalledOnNull(Vec::new()), token: 1, ..strict.clone() };
        let context =
            ColumnContext::new().with_callee("f", strict).with_callee("f", called_on_null);
        let evaluation = read(context, "f(NULL)");
        assert!(evaluation.may_be_null());
        assert!(!evaluation.is_always_null());
    }

    #[test]
    fn a_builtin_folds_only_when_every_callee_folds() {
        let mut folding = test_context::strict_callee(ScalarType::Integer(IntegerType::Integer));
        folding.builtin = Some(Builtin::Length);
        let opaque = test_context::strict_callee(ScalarType::Integer(IntegerType::Integer));
        let context =
            ColumnContext::new().with_callee("length", folding).with_callee("length", opaque);
        let evaluation = read(context, "length('abc')");
        assert!(evaluation.constant().is_none());
        assert!(evaluation.may_be_null());
    }

    #[test]
    fn a_builtin_still_folds_with_equal_callees() {
        let mut folding = test_context::strict_callee(ScalarType::Integer(IntegerType::Integer));
        folding.builtin = Some(Builtin::Length);
        let mut equal = test_context::strict_callee(ScalarType::Integer(IntegerType::Integer));
        equal.builtin = Some(Builtin::Length);
        equal.token = 1;
        let context =
            ColumnContext::new().with_callee("length", folding).with_callee("length", equal);
        let evaluation = read(context, "length('abc')");
        assert_eq!(evaluation.constant(), Some(&PgValue::Integer(3)));
    }

    #[test]
    fn an_all_over_a_null_member_is_null_when_it_equals_the_rest() {
        let context = ColumnContext::new().with_column(
            "x",
            Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                false,
                false,
                Values::Listed(alloc::collections::BTreeSet::from([PgValue::Integer(1)])),
            ),
        );
        let evaluation = read(context, "x = ALL (ARRAY[1, NULL])");
        assert!(evaluation.is_always_null());
    }

    #[test]
    fn an_all_over_a_null_member_is_null_when_some_value_equals_the_rest() {
        let context = ColumnContext::new().with_column(
            "x",
            Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                false,
                false,
                Values::Listed(alloc::collections::BTreeSet::from([
                    PgValue::Integer(1),
                    PgValue::Integer(2),
                ])),
            ),
        );
        let evaluation = read(context, "x = ALL (ARRAY[1, NULL])");
        assert!(evaluation.may_be_null());
        assert!(evaluation.may_be_false());
        assert!(!evaluation.may_be_true());
    }

    /// A possibly empty subquery reads `FALSE` to `ANY` and `TRUE` to `ALL`
    /// on its empty rows, past what the rows it returns read.
    #[test]
    fn a_quantifier_over_a_sometimes_empty_result_takes_its_empty_reading() {
        let column = |value: i64| {
            Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                false,
                false,
                Values::Listed(BTreeSet::from([PgValue::Integer(value)])),
            )
        };
        // `1 = ANY` and `2 = ALL` over the member `1` alone read the
        // opposite of the empty reading.
        for (x, sql) in [(1, "x = ANY (SELECT v FROM q)"), (2, "x = ALL (SELECT v FROM q)")] {
            let context = ColumnContext::new()
                .with_column("x", column(x))
                .with_query(Rows { none: true, one: true, many: true }, column(1));
            let evaluation = read(context, sql);
            assert!(evaluation.may_be_true(), "{sql}");
            assert!(evaluation.may_be_false(), "{sql}");
            assert!(!evaluation.may_be_null(), "{sql}");
        }
    }

    #[test]
    fn an_any_over_a_null_member_is_null_when_it_is_apart() {
        let context = ColumnContext::new().with_column(
            "x",
            Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                false,
                false,
                Values::Listed(alloc::collections::BTreeSet::from([PgValue::Integer(2)])),
            ),
        );
        let evaluation = read(context, "x = ANY (ARRAY[1, NULL])");
        assert!(evaluation.is_always_null());
    }

    #[test]
    fn alternative_callee_bodies_join_their_answers() {
        let callee = test_context::strict_callee(ScalarType::Text(TextRules::DEFAULT));
        let other = Callee { token: 1, ..callee.clone() };
        let context = ColumnContext::new()
            .with_callee("f", callee)
            .with_callee("f", other)
            .with_body(
                0,
                Evaluation::known(PgValue::Text("x".into()), ScalarType::Text(TextRules::DEFAULT)),
            )
            .with_body(
                1,
                Evaluation::known(PgValue::Text("y".into()), ScalarType::Text(TextRules::DEFAULT)),
            );
        let evaluation = read(context, "f('a')");
        assert!(evaluation.may_be(&PgValue::Text("x".into())));
        assert!(evaluation.may_be(&PgValue::Text("y".into())));
        assert!(!evaluation.is_always_null());
    }

    fn always(sql: &str, value: &PgValue) {
        let evaluation = read(ColumnContext::new(), sql);
        assert_eq!(evaluation.constant(), Some(value));
    }

    #[test]
    fn a_null_operand_reads_the_null_outcome_of_is_true() {
        assert!(read(ColumnContext::new(), "(NULL) IS TRUE").may_be_false());
        always("(NULL) IS NOT TRUE", &PgValue::Boolean(true));
        assert!(read(ColumnContext::new(), "(NULL) IS FALSE").may_be_false());
        always("(NULL) IS NOT FALSE", &PgValue::Boolean(true));
    }

    #[test]
    fn distinct_from_reads_the_null_rows() {
        always("NULL IS DISTINCT FROM 1", &PgValue::Boolean(true));
        always("NULL IS NOT DISTINCT FROM 1", &PgValue::Boolean(false));
        always("NULL IS DISTINCT FROM NULL", &PgValue::Boolean(false));
        always("NULL IS NOT DISTINCT FROM NULL", &PgValue::Boolean(true));
    }

    #[test]
    fn a_date_plus_days_folds_to_a_date() {
        let evaluation = read(ColumnContext::new(), "DATE '2026-01-01' + 5");
        assert_eq!(evaluation.ty(), ScalarType::Date);
        assert!(evaluation.constant().is_some());
        assert!(!evaluation.may_be_null());
    }

    #[test]
    fn an_integer_product_folds() {
        always("5 * 3", &PgValue::Integer(15));
    }

    #[test]
    fn a_bounded_varchar_cast_truncates_the_value() {
        always("'abcdef'::varchar(3)", &PgValue::Text("abc".to_string()));
    }

    #[test]
    fn a_boolean_cast_to_text_spells_the_word() {
        always("true::text", &PgValue::Text("true".to_string()));
        always("false::text", &PgValue::Text("false".to_string()));
    }

    #[test]
    fn a_text_cast_to_an_integer_folds() {
        always("'123'::int", &PgValue::Integer(123));
    }

    #[test]
    fn a_numeric_typmod_rounds_to_the_scale() {
        let evaluation = read(ColumnContext::new(), "1.9999::numeric(10, 4)");
        assert_eq!(evaluation.ty(), ScalarType::Numeric);
        assert!(evaluation.constant().is_some());
    }

    #[test]
    fn a_coalesce_past_the_nulls_folds_to_the_next_value() {
        always("coalesce(null, null, 5)", &PgValue::Integer(5));
    }

    #[test]
    fn a_coalesce_of_a_constant_folds_to_it() {
        always("coalesce(5, 6)", &PgValue::Integer(5));
    }

    #[test]
    fn a_coalesce_of_all_nulls_may_be_null() {
        let evaluation = read(ColumnContext::new(), "coalesce(null, null)");
        assert!(evaluation.may_be_null());
    }

    #[test]
    fn a_greatest_folds_to_the_maximum() {
        always("greatest(1, 9)", &PgValue::Integer(9));
    }

    #[test]
    fn a_least_folds_to_the_minimum() {
        always("least(9, 1)", &PgValue::Integer(1));
    }

    #[test]
    fn date_literals_compare_by_order() {
        assert!(read(ColumnContext::new(), "DATE '2026-01-01' < DATE '2026-01-02'").may_be_true());
    }

    #[test]
    fn integer_literals_compare_by_value() {
        always("1 = 2", &PgValue::Boolean(false));
        always("1 < 2", &PgValue::Boolean(true));
    }

    use sqlparser::{
        ast::{BinaryOperator, Expr, Ident, UnaryOperator, Value, ValueWithSpan},
        tokenizer::Span,
    };

    /// The boolean column `name` as a read expression.
    fn column(name: &str) -> Expr {
        Expr::Identifier(Ident::new(name.to_string()))
    }

    fn boolean_unknown() -> Evaluation {
        Evaluation::new(ScalarType::Boolean, true, false, Values::Unlisted(Unlisted::any()))
    }

    fn integer_unknown() -> Evaluation {
        Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        )
    }

    fn number(token: &str) -> Expr {
        Expr::Value(ValueWithSpan {
            value: Value::Number(token.to_string(), false),
            span: Span::empty(),
        })
    }

    /// The smallest nesting depth that overflows a 2 MiB stack in a
    /// recursive implementation holding one `Evaluation` per frame, measured
    /// by descending a recursive stack with that frame.
    const DEPTH: usize = 5_672;

    /// Runs `evaluate` on a 2 MiB thread and returns its result. The deep
    /// expression is leaked: a left-nested tree's recursive destructor would
    /// itself overflow the small stack, so it must not run there.
    fn on_small_stack(expression: Expr, mut context: ColumnContext) -> Evaluation {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                let result = evaluate(
                    &expression,
                    &PostgreSqlDialect {},
                    &PostgreSqlOptions::default(),
                    &mut context,
                    &Guarantees::new(),
                );
                std::mem::forget(expression);
                result
            })
            .expect("the thread starts")
            .join()
            .expect("the thread keeps the stack small")
            .expect("the context resolves")
    }

    #[test]
    fn a_5_672_term_and_or_or_chain_evaluates_on_a_small_stack() {
        for op in [BinaryOperator::And, BinaryOperator::Or] {
            let terms = DEPTH;
            let mut expression = column("c0");
            for i in 1..terms {
                expression = Expr::BinaryOp {
                    left: Box::new(expression),
                    op: op.clone(),
                    right: Box::new(column(&format!("c{i}"))),
                };
            }
            let mut context = ColumnContext::new();
            for i in 0..terms {
                context = context.with_column(&format!("c{i}"), boolean_unknown());
            }
            let evaluation = on_small_stack(expression, context);
            assert!(evaluation.may_be_true(), "{op}");
            assert!(evaluation.may_be_false(), "{op}");
            assert!(evaluation.may_be_null(), "{op}");
        }
    }

    #[test]
    fn a_5_672_nested_nots_evaluate_on_a_small_stack() {
        let mut expression = column("x");
        for _ in 0..DEPTH {
            expression = Expr::UnaryOp { op: UnaryOperator::Not, expr: Box::new(expression) };
        }
        let evaluation =
            on_small_stack(expression, ColumnContext::new().with_column("x", boolean_unknown()));
        assert!(!evaluation.is_impossible());
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_null());
    }

    #[test]
    fn a_5_672_nested_parentheses_evaluate_on_a_small_stack() {
        let mut expression = column("x");
        for _ in 0..DEPTH {
            expression = Expr::Nested(Box::new(expression));
        }
        let evaluation =
            on_small_stack(expression, ColumnContext::new().with_column("x", boolean_unknown()));
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_null());
    }

    #[test]
    fn a_5_672_nested_additions_evaluate_on_a_small_stack() {
        let one = number("1");
        let mut expression = column("x");
        for _ in 0..DEPTH {
            expression = Expr::BinaryOp {
                left: Box::new(expression),
                op: BinaryOperator::Plus,
                right: Box::new(one.clone()),
            };
        }
        let evaluation =
            on_small_stack(expression, ColumnContext::new().with_column("x", integer_unknown()));
        assert!(!evaluation.is_impossible());
        assert_eq!(evaluation.constant(), None);
    }

    #[test]
    fn a_5_672_nested_negations_evaluate_on_a_small_stack() {
        let mut expression = column("x");
        for _ in 0..DEPTH {
            expression = Expr::UnaryOp { op: UnaryOperator::Minus, expr: Box::new(expression) };
        }
        let evaluation =
            on_small_stack(expression, ColumnContext::new().with_column("x", integer_unknown()));
        assert!(!evaluation.is_impossible());
        assert_eq!(evaluation.constant(), None);
    }
}
