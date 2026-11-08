//! Helpers the forward evaluation pass shares.

use alloc::{collections::BTreeSet, string::String, vec::Vec};
use core::ops::Bound;

use sql_values::{PgNumeric, PgValue};
use sqlparser::ast::{
    Expr, Function, FunctionArg, FunctionArgExpr, FunctionArguments, ObjectName, Value,
    ValueWithSpan,
};

use super::{
    context::{Context, Resolved},
    domain::{Evaluation, Interval, Outcomes, Values},
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

    use sql_values::{PgNumeric, PgValue};
    use sqlparser::ast::{Expr, Function};

    use super::{and_outcomes, intervals_disjoint, not_outcomes, number_bounds, or_outcomes};
    use crate::{
        domain::{Interval, Outcomes, Unlisted, Values},
        test_context::parse,
    };

    fn integers(values: &[i64]) -> Vec<PgValue> {
        values.iter().map(|value| PgValue::Integer(*value)).collect()
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
