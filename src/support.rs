//! Helpers shared by the forward and the backward evaluation passes.

use alloc::{collections::BTreeSet, string::String, vec::Vec};
use core::ops::Bound;

use sql_values::{PgNumeric, PgValue};
use sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, InTarget, ObjectName, Value,
    ValueWithSpan,
};

use super::{
    context::{CalleeKind, Context, Resolved, Volatility},
    domain::{Evaluation, Interval, Outcomes, Values},
    guarantees::Subject,
    scalar::ScalarType,
};

/// Whether the expression is one the context could resolve: an identifier, a
/// compound identifier or a positional parameter.
#[must_use]
pub(crate) fn is_reference(expression: &Expr) -> bool {
    matches!(
        expression,
        Expr::Identifier(_)
            | Expr::CompoundIdentifier(_)
            | Expr::Value(ValueWithSpan { value: Value::Placeholder(_), .. })
    )
}

/// Resolves `expression` through the context when it is a reference,
/// `None` otherwise.
pub(crate) fn resolve_reference<C: Context>(
    context: &mut C,
    expression: &Expr,
) -> Result<Option<Resolved<C::Reference>>, C::Error> {
    if is_reference(expression) { context.reference(expression) } else { Ok(None) }
}

/// The subject a guarantee about `expression` is keyed under, asking the
/// context to resolve a reference, `None` when no guarantee may be keyed
/// under it.
///
/// An expression other than a reference is a subject only when every
/// evaluation in one statement gives it one value. It is built from literals,
/// operators, casts, `CASE`, lists and calls of immutable or stable ordinary
/// functions alone, so it reads no column and holds no subquery. Two calls of
/// `random()` differ, and a column named alike in a subquery may be another
/// column.
pub(crate) fn subject<C: Context>(
    context: &mut C,
    expression: &Expr,
) -> Result<Option<Subject<C::Reference>>, C::Error> {
    let expression = unnest(expression);
    if let Some(resolved) = resolve_reference(context, expression)? {
        return Ok(Some(Subject::Reference(resolved.reference)));
    }
    // The walk keeps its own stack, so a long left-deep chain costs no
    // recursion.
    let mut stack = vec![expression];
    while let Some(node) = stack.pop() {
        if !push_fixed_children(context, &mut stack, node)? {
            return Ok(None);
        }
    }
    Ok(Some(Subject::Expression(expression.clone())))
}

/// Pushes the children of `node` when `node` itself gives one value per
/// statement, answering `false` for any other node.
fn push_fixed_children<'a, C: Context>(
    context: &mut C,
    stack: &mut Vec<&'a Expr>,
    node: &'a Expr,
) -> Result<bool, C::Error> {
    match node {
        Expr::Value(_) | Expr::TypedString(_) => {}
        Expr::Nested(value)
        | Expr::Cast { expr: value, .. }
        | Expr::UnaryOp { expr: value, .. }
        | Expr::IsNull(value)
        | Expr::IsNotNull(value)
        | Expr::IsTrue(value)
        | Expr::IsNotTrue(value)
        | Expr::IsFalse(value)
        | Expr::IsNotFalse(value)
        | Expr::IsUnknown(value)
        | Expr::IsNotUnknown(value) => stack.push(value),
        Expr::BinaryOp { left, right, .. }
        | Expr::IsDistinctFrom(left, right)
        | Expr::IsNotDistinctFrom(left, right) => stack.extend([&**left, &**right]),
        Expr::Between { expr, low, high, .. } => stack.extend([&**expr, &**low, &**high]),
        Expr::In { expr, target: InTarget::List(list), .. } => {
            stack.push(expr);
            stack.extend(list);
        }
        Expr::Tuple(values) => stack.extend(values),
        Expr::Array(array) => stack.extend(&array.elem),
        Expr::Case { operand, conditions, else_result, .. } => {
            stack.extend(operand.as_deref());
            stack.extend(else_result.as_deref());
            for when in conditions {
                stack.extend([&when.condition, &when.result]);
            }
        }
        Expr::Function(call) => return push_fixed_call(context, stack, call),
        _ => return Ok(false),
    }
    Ok(true)
}

/// Pushes the arguments of `call` when it is a plain call whose every callee
/// is an immutable or stable ordinary function.
fn push_fixed_call<'a, C: Context>(
    context: &mut C,
    stack: &mut Vec<&'a Expr>,
    call: &'a Function,
) -> Result<bool, C::Error> {
    let arguments = match &call.args {
        FunctionArguments::None => &[][..],
        FunctionArguments::List(list)
            if list.clauses.is_empty() && list.duplicate_treatment.is_none() =>
        {
            &list.args[..]
        }
        _ => return Ok(false),
    };
    if call.over.is_some()
        || call.filter.is_some()
        || !call.within_group.is_empty()
        || !matches!(call.parameters, FunctionArguments::None)
    {
        return Ok(false);
    }
    for argument in arguments {
        let Some(value) = argument_expression(argument) else {
            return Ok(false);
        };
        stack.push(value);
    }
    let callees = context.callees(call)?;
    Ok(!callees.is_empty()
        && callees.iter().all(|callee| {
            callee.kind == CalleeKind::Normal && callee.volatility != Volatility::Volatile
        }))
}

/// `expression` without the parentheses around it.
#[must_use]
pub(crate) fn unnest(mut expression: &Expr) -> &Expr {
    while let Expr::Nested(inner) = expression {
        expression = inner;
    }
    expression
}

/// The terms of the flat left-nested `AND` or `OR` chain of `left` and
/// `right`, read left to right, stopping at a different operator.
#[must_use]
pub(crate) fn and_or_terms<'a>(
    left: &'a Expr,
    right: &'a Expr,
    and: bool,
) -> alloc::vec::Vec<&'a Expr> {
    let op =
        if and { sqlparser::ast::BinaryOperator::And } else { sqlparser::ast::BinaryOperator::Or };
    let mut terms = alloc::vec::Vec::new();
    let (mut chain_left, mut chain_right) = (unnest(left), unnest(right));
    loop {
        terms.push(chain_right);
        let Expr::BinaryOp { left, op: inner, right } = chain_left else {
            break;
        };
        if *inner != op {
            break;
        }
        (chain_left, chain_right) = (unnest(left), unnest(right));
    }
    terms.push(chain_left);
    terms.reverse();
    terms
}

/// The evaluation narrowed to the values and the `NULL` possibility that
/// `outcomes` allow, as a condition restricted to them.
#[must_use]
pub(crate) fn prune(evaluation: &Evaluation, outcomes: Outcomes) -> Evaluation {
    let values = match evaluation.values() {
        Values::Listed(listed) => {
            let kept: BTreeSet<PgValue> = listed
                .iter()
                .filter(|value| {
                    match value {
                        PgValue::Boolean(true) => outcomes.truth,
                        PgValue::Boolean(false) => outcomes.falsity,
                        _ => true,
                    }
                })
                .cloned()
                .collect();
            Values::Listed(kept)
        }
        // A boolean set always lists its members, so an unlisted value
        // keeps its exclusions.
        Values::Unlisted(_) => evaluation.values().clone(),
    };
    Evaluation::new(
        evaluation.ty(),
        evaluation.may_be_null() && outcomes.null,
        evaluation.may_raise(),
        values,
    )
}

/// The outcomes of `left AND right`, three-valued.
#[must_use]
pub(crate) fn and_outcomes(left: Outcomes, right: Outcomes) -> Outcomes {
    Outcomes {
        truth: left.truth && right.truth,
        falsity: left.falsity || (right.falsity && (left.truth || left.null)),
        null: (left.null && right.truth) || (right.null && left.truth) || (left.null && right.null),
    }
}

/// The outcomes of `left OR right`, three-valued.
#[must_use]
pub(crate) fn or_outcomes(left: Outcomes, right: Outcomes) -> Outcomes {
    Outcomes {
        truth: left.truth || right.truth,
        falsity: left.falsity && right.falsity,
        null: (left.null && right.falsity)
            || (right.null && left.falsity)
            || (left.null && right.null),
    }
}

/// The outcomes of `NOT left`, three-valued.
#[must_use]
pub(crate) fn not_outcomes(left: Outcomes) -> Outcomes {
    Outcomes { truth: left.falsity, falsity: left.truth, null: left.null }
}

/// The one-outcome readings of an outcome set, in reading order.
pub(crate) const OUTCOMES: [Outcomes; 3] = [Outcomes::TRUE, Outcomes::FALSE, Outcomes::NULL];

/// The outcomes of `left AND right` over the outcome sets, three-valued.
#[must_use]
pub(crate) fn and_outcomes_set(left: Outcomes, right: Outcomes) -> Outcomes {
    product_outcomes(Outcomes::NONE, left, right, and_outcomes)
}

/// The outcomes of `left OR right` over the outcome sets, three-valued.
#[must_use]
pub(crate) fn or_outcomes_set(left: Outcomes, right: Outcomes) -> Outcomes {
    product_outcomes(Outcomes::NONE, left, right, or_outcomes)
}

/// The outcomes `combine` gives the members of the two sets, in every
/// pairing.
fn product_outcomes(
    mut out: Outcomes,
    left: Outcomes,
    right: Outcomes,
    combine: fn(Outcomes, Outcomes) -> Outcomes,
) -> Outcomes {
    for a in OUTCOMES {
        if a.intersection(left).is_empty() {
            continue;
        }
        for b in OUTCOMES {
            if b.intersection(right).is_empty() {
                continue;
            }
            out = out.union(combine(a, b));
        }
    }
    out
}

/// A condition taking exactly the outcomes `outcomes` and raising `error`.
#[must_use]
pub(crate) fn condition(outcomes: Outcomes, error: bool) -> Evaluation {
    Evaluation::condition(outcomes, error)
}

/// Whether two numeric intervals admit no common value.
#[must_use]
pub(crate) fn intervals_disjoint(a: &Interval<PgNumeric>, b: &Interval<PgNumeric>) -> bool {
    a.meet(b).is_empty()
}

/// The smallest and the largest of the listed numeric datums.
fn listed_numbers(values: &BTreeSet<PgValue>) -> (Option<PgNumeric>, Option<PgNumeric>) {
    // The set iterates in ascending datum order, so the numeric value it
    // admits first is its minimum and the one it admits last its maximum.
    let mut low = None;
    let mut high = None;
    for value in values {
        if let Some(number) = value.as_numeric() {
            if low.is_none() {
                low = Some(number.clone());
            }
            high = Some(number);
        }
    }
    (low, high)
}

/// The numeric interval a value set lies in, unbounded when unknown.
#[must_use]
pub(crate) fn number_bounds(values: &Values) -> Interval<PgNumeric> {
    match values {
        Values::Listed(listed) => {
            let (low, high) = listed_numbers(listed);
            Interval {
                low: low.map_or(Bound::Unbounded, Bound::Included),
                high: high.map_or(Bound::Unbounded, Bound::Included),
            }
        }
        Values::Unlisted(unlisted) => unlisted.numbers.clone(),
    }
}

/// The character-length interval a value set's text lies in, unbounded when
/// unknown or when it lists no text.
#[must_use]
pub(crate) fn length_bounds(values: &Values) -> Interval<u64> {
    match values {
        Values::Listed(listed) => {
            let mut lengths = listed.iter().filter_map(crate::scalar::text_length);
            let Some(first) = lengths.next() else {
                return Interval::unbounded();
            };
            let (low, high) = lengths
                .fold((first, first), |(low, high), length| (low.min(length), high.max(length)));
            Interval { low: Bound::Included(low), high: Bound::Included(high) }
        }
        Values::Unlisted(unlisted) => unlisted.length.clone(),
    }
}

/// The unqualified, lowercased name of a call with a single unquoted name
/// segment, the only spelling that names a keyword construct such as
/// `COALESCE`. A qualified or quoted name names a routine of that name.
#[must_use]
pub(crate) fn simple_name(name: &ObjectName) -> Option<String> {
    let [part] = name.0.as_slice() else {
        return None;
    };
    part.as_ident()
        .filter(|ident| ident.quote_style.is_none())
        .map(|ident| ident.value.to_lowercase())
}

/// The argument expressions of a call, in order, `None` where a wildcard or
/// an unsupported form takes the argument's place.
#[must_use]
pub(crate) fn call_arguments(function: &Function) -> Vec<Option<Expr>> {
    let FunctionArguments::List(list) = &function.args else {
        return Vec::new();
    };
    list.args.iter().map(|argument| argument_expression(argument).cloned()).collect()
}

/// The expression an argument passes, positionally or by name, `None` for a
/// wildcard or another form.
#[must_use]
pub(crate) fn argument_expression(argument: &FunctionArg) -> Option<&Expr> {
    match argument {
        FunctionArg::Unnamed(FunctionArgExpr::Expr(expr))
        | FunctionArg::Named { arg: FunctionArgExpr::Expr(expr), .. }
        | FunctionArg::ExprNamed { arg: FunctionArgExpr::Expr(expr), .. } => Some(expr),
        _ => None,
    }
}

/// The numeric value of a character count, `None` past `i64`.
#[must_use]
pub(crate) fn count_numeric(count: u64) -> Option<PgNumeric> {
    i64::try_from(count).ok().map(PgNumeric::from)
}

/// The values of a listed set, admitted by `interval`.
#[must_use]
pub(crate) fn filter_listed_by_interval(
    values: &BTreeSet<PgValue>,
    interval: &Interval<PgNumeric>,
) -> BTreeSet<PgValue> {
    values
        .iter()
        .filter(|value| value.as_numeric().is_some_and(|number| interval.contains(&number)))
        .cloned()
        .collect()
}

/// Whether two value sets could hold a common value on one row.
#[must_use]
pub(crate) fn values_share(left: &Evaluation, right: &Evaluation) -> bool {
    match (left.values(), right.values()) {
        (Values::Listed(l), Values::Listed(r)) => {
            if l.is_empty() || r.is_empty() {
                return false;
            }
            l.iter().any(|value| r.contains(value))
        }
        (Values::Listed(l), Values::Unlisted(u)) | (Values::Unlisted(u), Values::Listed(l)) => {
            if l.is_empty() {
                return false;
            }
            l.iter().any(|value| u.admits(value))
        }
        (Values::Unlisted(l), Values::Unlisted(r)) => {
            if left.ty().is_number() && right.ty().is_number() {
                !intervals_disjoint(&l.numbers, &r.numbers)
            } else if left.ty().is_text() && right.ty().is_text() {
                !l.length.is_empty() && !r.length.is_empty()
            } else {
                !left.values().is_empty() && !right.values().is_empty()
            }
        }
    }
}

/// Whether the type could stand for a condition.
#[must_use]
pub(crate) fn could_be_boolean(ty: ScalarType) -> bool {
    matches!(ty, ScalarType::Boolean | ScalarType::Unknown | ScalarType::UnknownLiteral)
}

/// `evaluation` read as a condition: an untyped literal through the
/// `boolean` input, and a value of another type as any `boolean` an engine
/// may coerce it to, or a raise.
#[must_use]
pub(crate) fn as_condition(evaluation: Evaluation) -> Evaluation {
    match evaluation.ty() {
        ScalarType::Boolean | ScalarType::Unknown => evaluation,
        _ if evaluation.is_impossible() => Evaluation::impossible(ScalarType::Boolean),
        ScalarType::UnknownLiteral => {
            crate::forward::fold::reinterpret_unknown(evaluation, ScalarType::Boolean)
        }
        _ => {
            Evaluation::new(
                ScalarType::Boolean,
                evaluation.may_be_null(),
                true,
                Values::Unlisted(crate::domain::Unlisted::any()),
            )
        }
    }
}

/// Whether an engine may refuse a unary sign over `evaluation`, as over a
/// non-number type or an untyped `NULL`.
#[must_use]
pub(crate) fn sign_may_raise(evaluation: &Evaluation) -> bool {
    let ty = evaluation.ty();
    let untyped_null = ty == ScalarType::Unknown && evaluation.is_always_null();
    untyped_null || !(ty.is_number() || matches!(ty, ScalarType::Float | ScalarType::Unknown))
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use core::ops::Bound;

    use sql_values::{IntegerType, PgNumeric, PgValue};
    use sqlparser::ast::{Expr, Function};

    use super::{
        and_outcomes, intervals_disjoint, is_reference, not_outcomes, number_bounds, or_outcomes,
        prune, subject, unnest, values_share,
    };
    use crate::{
        context::{Callee, CalleeKind, Context, NullInput, Resolved, Volatility},
        domain::{Evaluation, Interval, Outcomes, Unlisted, Values},
        guarantees::Subject,
        scalar::{ScalarType, TextRules},
        test_context::{ColumnContext, parse, strict_callee},
    };

    /// A context whose callee lookup always fails.
    struct FailingCallees;

    impl Context for FailingCallees {
        type Reference = String;
        type Error = String;

        fn reference(
            &mut self,
            _expression: &Expr,
        ) -> Result<Option<Resolved<Self::Reference>>, Self::Error> {
            Ok(None)
        }

        fn callees(&mut self, _call: &Function) -> Result<Vec<Callee>, Self::Error> {
            Err("no callee table".to_string())
        }
    }

    fn volatile_callee() -> Callee {
        Callee {
            builtin: None,
            null_input: NullInput::Strict,
            volatility: Volatility::Volatile,
            kind: CalleeKind::Normal,
            returns: ScalarType::Float,
            returns_set: false,
            token: 0,
        }
    }

    fn listed(ty: ScalarType, values: &[PgValue]) -> Evaluation {
        Evaluation::new(ty, false, false, Values::Listed(values.iter().cloned().collect()))
    }

    fn integers(values: &[i64]) -> Vec<PgValue> {
        values.iter().map(|value| PgValue::Integer(*value)).collect()
    }

    #[test]
    fn a_reference_subject_comes_from_the_context() {
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let mut context =
            ColumnContext::new().with_column("x", Evaluation::known(PgValue::Integer(1), int4));
        assert!(matches!(
            subject(&mut context, &parse("x")).expect("resolves"),
            Some(Subject::Reference(_))
        ));
        // a positional parameter is a reference only when the context names it
        assert!(is_reference(&parse("$1")));
        assert!(matches!(
            subject(&mut context, &parse("$1")).expect("resolves"),
            Some(Subject::Expression(_))
        ));
        assert!(!is_reference(&parse("1 + 2")));
    }

    #[test]
    fn a_stable_expression_is_a_subject_and_a_column_is_not() {
        let text = ScalarType::Text(TextRules::DEFAULT);
        let mut context = ColumnContext::new().with_callee("upper", strict_callee(text));
        // a call of a stable function over constants reads no column
        assert!(matches!(
            subject(&mut context, &parse("upper('a') + 1")).expect("resolves"),
            Some(Subject::Expression(_))
        ));
        // the same shape that reads a column is no subject at all
        let mut context =
            context.with_column("x", Evaluation::known(PgValue::Text("a".to_string()), text));
        assert!(subject(&mut context, &parse("upper(x)")).expect("resolves").is_none());
        // a volatile callee changes between two evaluations of one statement
        let mut context = ColumnContext::new().with_callee("random", volatile_callee());
        assert!(subject(&mut context, &parse("random()")).expect("resolves").is_none());
        // an aggregate and a window are two values per row, not one
        let aggregate = Callee { kind: CalleeKind::Aggregate, ..volatile_callee() };
        let mut context = ColumnContext::new().with_callee("sum", aggregate);
        assert!(subject(&mut context, &parse("sum(1)")).expect("resolves").is_none());
        let mut context = ColumnContext::new().with_callee("upper", strict_callee(text));
        assert!(subject(&mut context, &parse("upper('a') OVER ()")).expect("resolves").is_none());
        // a callee the context cannot read is no subject
        let mut context = ColumnContext::new();
        assert!(subject(&mut context, &parse("nosuch('a')")).expect("resolves").is_none());
        // a subquery names another query's columns
        assert!(subject(&mut context, &parse("EXISTS (SELECT 1)")).expect("resolves").is_none());
        let mut context = FailingCallees;
        // a reference the context cannot name is still no expression subject
        assert!(subject(&mut context, &parse("x")).expect("resolves").is_none());
        assert!(subject(&mut context, &parse("nosuch('a')")).is_err());
    }

    #[test]
    fn every_constant_form_is_a_subject_and_others_are_not() {
        let text = ScalarType::Text(TextRules::DEFAULT);
        let mut context = ColumnContext::new()
            .with_callee("upper", strict_callee(text))
            .with_callee("now", strict_callee(text));
        for sql in [
            "DATE '2026-01-01'",
            "CAST(upper('a') AS int)",
            "-1",
            "upper('a') IS NULL",
            "upper('a') IS NOT NULL",
            "upper('a') IS TRUE",
            "upper('a') IS NOT TRUE",
            "upper('a') IS FALSE",
            "upper('a') IS NOT FALSE",
            "upper('a') IS UNKNOWN",
            "upper('a') IS NOT UNKNOWN",
            "1 IS DISTINCT FROM 2",
            "1 IS NOT DISTINCT FROM 2",
            "2 BETWEEN 1 AND 3",
            "1 IN (1, 2)",
            "(1, 2)",
            "ARRAY[1, 2]",
            "CASE 1 WHEN 1 THEN 'a' ELSE 'b' END",
            "CASE WHEN true THEN 'a' END",
            "now()",
            "upper(arg => 'a')",
        ] {
            assert!(
                matches!(subject(&mut context, &parse(sql)), Ok(Some(Subject::Expression(_)))),
                "{sql}"
            );
        }
        // every refusal is a shape that may differ between rows or evaluations
        for sql in [
            "1 IN (SELECT 1)",
            "(SELECT 1)",
            "upper('a') FILTER (WHERE true)",
            "upper(DISTINCT 'a')",
            "upper('a' ORDER BY 1)",
            "upper(*)",
            "upper(upper('a') + x)",
            "percentile_cont(0.5) WITHIN GROUP (ORDER BY 1)",
            "1 LIKE 2",
            "array(SELECT 1)",
        ] {
            assert!(matches!(subject(&mut context, &parse(sql)), Ok(None)), "{sql}");
        }
    }

    #[test]
    fn unnest_strips_the_parentheses() {
        assert!(matches!(
            subject(
                &mut ColumnContext::new(),
                &parse("((1 + 2))")
            )
            .expect("resolves"),
            Some(Subject::Expression(expr)) if matches!(&unnest(&expr), Expr::BinaryOp { .. })
        ));
    }

    #[test]
    fn prune_keeps_only_the_admitted_values() {
        let boolean = Evaluation::new(
            ScalarType::Boolean,
            true,
            false,
            Values::Listed(BTreeSet::from([PgValue::Boolean(false), PgValue::Boolean(true)])),
        );
        let pruned = prune(&boolean, Outcomes { truth: true, falsity: false, null: true });
        assert_eq!(pruned.values(), &Values::Listed(BTreeSet::from([PgValue::Boolean(true)])));
        assert!(pruned.may_be_null());
        let pruned = prune(&boolean, Outcomes { truth: false, falsity: true, null: false });
        assert_eq!(pruned.values(), &Values::Listed(BTreeSet::from([PgValue::Boolean(false)])));
        assert!(!pruned.may_be_null());
        // a non-boolean listed set keeps its members and only the null
        // possibility is restricted
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let numbers = Evaluation::new(
            int4,
            true,
            false,
            Values::Listed(integers(&[1, 2]).into_iter().collect()),
        );
        let pruned = prune(&numbers, Outcomes::NULL);
        assert_eq!(pruned.values(), &Values::Listed(integers(&[1, 2]).into_iter().collect()));
        assert!(pruned.may_be_null());
        // an unlisted set keeps its exclusions
        let unlisted =
            Evaluation::new(ScalarType::Numeric, true, false, Values::Unlisted(Unlisted::any()));
        let pruned = prune(&unlisted, Outcomes::NULL);
        assert!(matches!(pruned.values(), Values::Unlisted(_)));
        assert!(pruned.may_be_null());
    }

    #[test]
    fn the_three_valued_combinators_follow_postgres() {
        assert_eq!(and_outcomes(Outcomes::TRUE, Outcomes::NULL), Outcomes::NULL);
        assert_eq!(
            and_outcomes(Outcomes { truth: true, falsity: true, null: true }, Outcomes::NULL),
            Outcomes { truth: false, falsity: true, null: true }
        );
        assert_eq!(
            or_outcomes(Outcomes { truth: false, falsity: true, null: true }, Outcomes::TRUE),
            Outcomes::TRUE
        );
        assert_eq!(or_outcomes(Outcomes::FALSE, Outcomes::NULL), Outcomes::NULL);
        assert_eq!(or_outcomes(Outcomes::NULL, Outcomes::NULL), Outcomes::NULL);
        assert_eq!(not_outcomes(Outcomes::NULL), Outcomes::NULL);
        assert_eq!(not_outcomes(Outcomes::TRUE), Outcomes::FALSE);
        assert_eq!(
            not_outcomes(Outcomes { truth: false, falsity: true, null: true }),
            Outcomes { truth: true, falsity: false, null: true }
        );
    }

    #[test]
    fn intervals_disjoint_checks_the_meet() {
        let a = Interval {
            low: Bound::Included(PgNumeric::from(1)),
            high: Bound::Included(PgNumeric::from(3)),
        };
        let b = Interval {
            low: Bound::Included(PgNumeric::from(4)),
            high: Bound::Included(PgNumeric::from(5)),
        };
        assert!(intervals_disjoint(&a, &b));
        assert!(!intervals_disjoint(
            &a,
            &Interval { low: Bound::Included(PgNumeric::from(3)), high: Bound::Unbounded }
        ));
        assert!(!intervals_disjoint(&a, &a));
        // an excluded bound inside the other interval still meets it
        let b = Interval {
            low: Bound::Included(PgNumeric::from(0)),
            high: Bound::Included(PgNumeric::from(2)),
        };
        let c = Interval {
            low: Bound::Excluded(PgNumeric::from(1)),
            high: Bound::Excluded(PgNumeric::from(3)),
        };
        assert!(!intervals_disjoint(&b, &c));
        assert_eq!(b.meet(&c).low, Bound::Excluded(PgNumeric::from(1)));
        assert_eq!(b.meet(&c).high, Bound::Included(PgNumeric::from(2)));
    }

    #[test]
    fn number_bounds_reads_the_extremes() {
        let bounds = number_bounds(&Values::Listed(integers(&[2, 5]).into_iter().collect()));
        assert_eq!(bounds.low, Bound::Included(PgNumeric::from(2)));
        assert_eq!(bounds.high, Bound::Included(PgNumeric::from(5)));
        // a listed set without a numeric member admits no bound
        let text = Values::Listed(BTreeSet::from([PgValue::Text("a".to_string())]));
        let bounds = number_bounds(&text);
        assert_eq!(bounds.low, Bound::Unbounded);
        assert_eq!(bounds.high, Bound::Unbounded);
        // an unlisted set passes its interval through
        let interval =
            Interval { low: Bound::Excluded(PgNumeric::from(1)), high: Bound::Unbounded };
        let unlisted = Values::Unlisted(Unlisted { numbers: interval.clone(), ..Unlisted::any() });
        assert_eq!(number_bounds(&unlisted), interval);
    }

    #[test]
    fn values_share_matches_the_common_value() {
        let int4 = ScalarType::Integer(IntegerType::Integer);
        assert!(values_share(&listed(int4, &integers(&[1, 2])), &listed(int4, &integers(&[2, 3]))));
        assert!(!values_share(
            &listed(int4, &integers(&[1, 2])),
            &listed(int4, &integers(&[3, 4]))
        ));
        let empty = listed(int4, &[]);
        assert!(!values_share(&empty, &listed(int4, &integers(&[1]))));
        // a listed value is admitted by an unlisted set only inside its
        // interval, not at an excluded bound
        let interval = |low: i64, high: i64| {
            Interval {
                low: Bound::Included(PgNumeric::from(low)),
                high: Bound::Included(PgNumeric::from(high)),
            }
        };
        let unlisted = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted { numbers: interval(2, 3), ..Unlisted::any() }),
        );
        assert!(values_share(&listed(int4, &integers(&[1, 2])), &unlisted));
        let excluded = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                numbers: Interval {
                    low: Bound::Excluded(PgNumeric::from(2)),
                    high: Bound::Unbounded,
                },
                ..Unlisted::any()
            }),
        );
        assert!(!values_share(&listed(int4, &integers(&[1, 2])), &excluded));
        // two unlisted number sets share a value when their intervals meet
        let number = |low: i64, high: i64| {
            Evaluation::new(
                ScalarType::Numeric,
                false,
                false,
                Values::Unlisted(Unlisted { numbers: interval(low, high), ..Unlisted::any() }),
            )
        };
        assert!(values_share(&number(1, 3), &number(3, 5)));
        assert!(!values_share(&number(1, 2), &number(3, 5)));
        // two unlisted text sets share a value when both may hold some text
        let text = ScalarType::Text(TextRules::DEFAULT);
        let text_set = Evaluation::new(text, false, false, Values::Unlisted(Unlisted::any()));
        let empty_text = Evaluation::new(
            text,
            false,
            false,
            Values::Unlisted(Unlisted {
                length: Interval { low: Bound::Included(5), high: Bound::Included(3) },
                ..Unlisted::any()
            }),
        );
        assert!(values_share(&text_set, &text_set));
        assert!(!values_share(&text_set, &empty_text));
        // two unlisted date sets share a value when neither is empty
        let date =
            Evaluation::new(ScalarType::Date, false, false, Values::Unlisted(Unlisted::any()));
        assert!(values_share(&date, &date));
        let empty_date = Evaluation::impossible(ScalarType::Date);
        assert!(!values_share(&date, &empty_date));
    }

    fn call(sql: &str) -> Function {
        let Expr::Function(function) = parse(sql) else { panic!("expected a call") };
        function
    }

    /// Only an unqualified, unquoted name is the keyword construct.
    #[test]
    fn simple_name_reads_one_unquoted_segment() {
        assert_eq!(super::simple_name(&call("NULLIF(1, 1)").name).as_deref(), Some("nullif"));
        assert_eq!(super::simple_name(&call("rv.nullif(1, 1)").name), None);
        assert_eq!(super::simple_name(&call("\"nullif\"(1, 1)").name), None);
    }

    /// A named argument of either spelling reads its expression.
    #[test]
    fn call_arguments_read_every_named_spelling() {
        let arguments = super::call_arguments(&call("f(1, a => 2, *)"));
        assert_eq!(arguments.len(), 3);
        assert!(arguments[0].is_some() && arguments[1].is_some() && arguments[2].is_none());
    }

    /// A listed set holding no text bounds no length.
    #[test]
    fn a_set_without_text_bounds_no_length() {
        let numbers = Values::Listed(BTreeSet::from([PgValue::Integer(1)]));
        assert_eq!(super::length_bounds(&numbers), Interval::unbounded());
    }

    /// A literal reads through the `boolean` input and another type stays open.
    #[test]
    fn as_condition_reads_the_literals() {
        use crate::{domain::Evaluation, scalar::ScalarType};
        let literal =
            |text: &str| Evaluation::known(PgValue::Text(text.into()), ScalarType::UnknownLiteral);
        let truth = super::as_condition(literal("t"));
        assert!(truth.may_be_true() && !truth.may_be_false() && !truth.may_raise());
        assert!(super::as_condition(literal("a")).always_raises());
        let any = Values::Unlisted(Unlisted::any());
        let unlisted = Evaluation::new(ScalarType::UnknownLiteral, false, false, any);
        assert!(super::as_condition(unlisted).may_raise());
        let impossible =
            Evaluation::impossible(ScalarType::Integer(sql_values::IntegerType::Integer));
        assert!(super::as_condition(impossible).is_impossible());
    }
}
