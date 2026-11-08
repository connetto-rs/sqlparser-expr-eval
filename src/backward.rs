//! The backward pass: what the subjects must satisfy for the expression to
//! take an outcome.

use alloc::{
    boxed::Box,
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};
use core::{cell::Cell, ops::Bound};

use sql_values::postgres::{IntegerType, PgNumeric, PgValue};
use sqlparser::ast::{
    Array, BinaryOperator, CastKind, CharacterLength, DataType, Expr, Function, FunctionArguments,
    InTarget, Query, UnaryOperator,
};

use super::{
    context::{Context, NullInput},
    dialect::EvalDialect,
    domain::{Evaluation, Interval, Outcomes, Unlisted, Values},
    forward,
    guarantees::{Guarantees, Subject},
    scalar::ScalarType,
    support,
};

/// The dialect and its settings, the context, the guarantees narrowed so
/// far and the path budget shared with the forward pass.
struct State<'a, D: EvalDialect, C: Context> {
    dialect: &'a D,
    options: &'a D::Options,
    context: &'a mut C,
    guarantees: &'a mut Guarantees<C::Reference>,
    budget: &'a Cell<u32>,
    /// How deep the pass has walked, for the depth limit.
    depth: u32,
}

/// The guarantees the subjects must satisfy when `expression` takes one of
/// `outcomes`, under `guarantees`, in the dialect and its settings.
///
/// # Errors
///
/// Returns the context's error when a reference, a call or a subquery fails
/// to resolve.
pub fn implied<D: EvalDialect, C: Context>(
    expression: &Expr,
    outcomes: Outcomes,
    dialect: &D,
    options: &D::Options,
    context: &mut C,
    guarantees: &Guarantees<C::Reference>,
) -> Result<Guarantees<C::Reference>, C::Error> {
    implied_within(
        expression,
        outcomes,
        dialect,
        options,
        context,
        guarantees,
        &Cell::new(forward::PATH_BUDGET),
    )
}

/// [`implied`] drawing on the path budget of an enclosing pass.
pub(super) fn implied_within<D: EvalDialect, C: Context>(
    expression: &Expr,
    outcomes: Outcomes,
    dialect: &D,
    options: &D::Options,
    context: &mut C,
    guarantees: &Guarantees<C::Reference>,
    budget: &Cell<u32>,
) -> Result<Guarantees<C::Reference>, C::Error> {
    let mut narrowed = guarantees.clone();
    let mut state =
        State { dialect, options, context, guarantees: &mut narrowed, budget, depth: 0 };
    node(&mut state, expression, outcomes)?;
    Ok(narrowed)
}

/// Narrows the guarantees so that `expression` takes one of `outcomes`.
fn node<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let mut expression = expression;
    let mut outcomes = outcomes;
    // `NOT` negates the outcome set exactly, so a run of it is folded here,
    // interleaved with the parentheses; splitting per level would double the
    // branches at each.
    loop {
        expression = match expression {
            Expr::Nested(inner) => inner,
            Expr::UnaryOp { op: UnaryOperator::Not, expr: inner } => {
                outcomes = outcomes.negated();
                inner
            }
            _ => break,
        };
    }
    if outcomes.is_empty() {
        impossible(state);
        return Ok(());
    }
    if state.depth >= forward::DEPTH_LIMIT {
        // Past the depth limit the pass narrows nothing more, which is a
        // safe over-approximation.
        return Ok(());
    }
    // An error abandons the pass, so only the success paths restore the
    // depth.
    state.depth += 1;
    if !single_outcome(outcomes) {
        let mut branches = Vec::with_capacity(3);
        for single in singles(outcomes).into_iter().flatten() {
            branches.push(branch(state, |s| node(s, expression, single))?);
        }
        merge(state, branches);
        state.depth -= 1;
        return Ok(());
    }
    let result = single_node(state, expression, outcomes);
    state.depth -= 1;
    result
}

/// Whether `outcomes` holds a single outcome.
fn single_outcome(outcomes: Outcomes) -> bool {
    [outcomes.truth, outcomes.falsity, outcomes.null].iter().filter(|bit| **bit).count() == 1
}

/// The single-outcome members of `outcomes`.
fn singles(outcomes: Outcomes) -> [Option<Outcomes>; 3] {
    [
        outcomes.truth.then_some(Outcomes::TRUE),
        outcomes.falsity.then_some(Outcomes::FALSE),
        outcomes.null.then_some(Outcomes::NULL),
    ]
}

/// The guarantees after `apply` runs on a copy of the current ones.
fn branch<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    apply: impl FnOnce(&mut State<'_, D, C>) -> Result<(), C::Error>,
) -> Result<Guarantees<C::Reference>, C::Error> {
    let mut guarantees = state.guarantees.clone();
    let mut branch_state = State {
        dialect: state.dialect,
        options: state.options,
        context: state.context,
        guarantees: &mut guarantees,
        budget: state.budget,
        depth: state.depth,
    };
    apply(&mut branch_state)?;
    Ok(guarantees)
}

/// Replaces the guarantees with the disjunction of `branches`.
fn merge<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    branches: Vec<Guarantees<C::Reference>>,
) {
    let mut joined: Option<Guarantees<C::Reference>> = None;
    for branch in branches {
        if branch.is_impossible() {
            continue;
        }
        joined = Some(match joined {
            Some(joined) => joined.join(&branch),
            None => branch,
        });
    }
    *state.guarantees = joined.unwrap_or_else(Guarantees::impossible);
}

/// Marks the guarantees impossible.
fn impossible<D: EvalDialect, C: Context>(state: &mut State<'_, D, C>) {
    *state.guarantees = Guarantees::impossible();
}

/// The forward evaluation of `expression` under the current guarantees.
fn forward_of<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
) -> Result<Evaluation, C::Error> {
    forward::evaluate_within(
        expression,
        state.dialect,
        state.options,
        state.context,
        state.guarantees,
        state.budget,
    )
}

/// Restricts `expression`'s subject so it cannot be `NULL`.
fn restrict_non_null<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
) -> Result<(), C::Error> {
    let evaluation = forward_of(state, expression)?;
    restrict_subject(state, expression, evaluation.with_null(false))
}

/// Narrows the subject of `expression` to `evaluation`, or marks the
/// guarantees impossible when `evaluation` admits nothing. An expression
/// that may not key a guarantee narrows nothing else.
fn restrict_subject<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
    evaluation: Evaluation,
) -> Result<(), C::Error> {
    if evaluation.values().is_empty() && !evaluation.may_be_null() {
        impossible(state);
        return Ok(());
    }
    let expression = support::unnest(expression);
    // A value-preserving text cast holds exactly its operand's values, so a
    // restriction on the cast is a restriction on the operand's subject.
    let expression = if let Expr::Cast { kind, expr: operand, data_type, .. } = expression
        && value_preserving_text_cast(state, kind, data_type, operand)?
    {
        operand
    } else {
        expression
    };
    if let Some(subject) = support::subject(state.context, expression)? {
        state.guarantees.restrict(subject, evaluation);
    }
    Ok(())
}

/// Whether the cast of `operand` to `data_type` is a plain cast of a text
/// value to a text type that changes no character, which carries a value
/// restriction through to the operand. The operand's collation must decide
/// equality by identity, so the restriction passes only under the column's
/// identity collation.
fn value_preserving_text_cast<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    kind: &CastKind,
    data_type: &DataType,
    operand: &Expr,
) -> Result<bool, C::Error> {
    if !matches!(kind, CastKind::Cast | CastKind::DoubleColon) {
        return Ok(false);
    }
    if !ScalarType::of(data_type).is_text() || cast_text_limit(data_type).is_some() {
        return Ok(false);
    }
    let operand = forward_of(state, operand)?;
    Ok(matches!(
        operand.ty(),
        ScalarType::Text(rules) if rules.identity_equality
    ))
}

/// Restricts `expression`'s subject to hold no value.
fn restrict_none<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
) -> Result<(), C::Error> {
    let evaluation = forward_of(state, expression)?;
    restrict_subject(state, expression, evaluation.with_values(Values::none()))
}

/// Narrows the guarantees for one outcome of `expression`.
fn single_node<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let evaluation = forward_of(state, expression)?;
    // A nothing-known reading narrows nothing, save the structural `coalesce`
    // rule.
    let coalesce = matches!(
        expression,
        Expr::Function(call) if support::simple_name(&call.name).as_deref() == Some("coalesce")
    );
    if evaluation.is_uninformative() && !coalesce {
        return Ok(());
    }
    // An untyped literal reads through the `boolean` input and has no
    // subject to narrow.
    if evaluation.ty() == ScalarType::UnknownLiteral {
        if support::as_condition(evaluation).outcomes().intersection(outcomes).is_empty() {
            impossible(state);
        }
        return Ok(());
    }
    if support::could_be_boolean(evaluation.ty()) {
        let pruned = support::prune(&evaluation, outcomes);
        if pruned.outcomes().is_empty() {
            impossible(state);
            return Ok(());
        }
        restrict_subject(state, expression, pruned)?;
        return shape(state, expression, outcomes);
    }
    // A non-`boolean` subject reads every value as a condition.
    if outcomes.truth || outcomes.falsity {
        let pruned = support::prune(&evaluation, outcomes);
        restrict_subject(state, expression, pruned)?;
        return Ok(());
    }
    restrict_none(state, expression)
}

/// The input narrowing of one boolean outcome of `expression`.
fn shape<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    match expression {
        Expr::BinaryOp { left, op: BinaryOperator::And, right } => {
            and_or(state, &support::and_or_terms(left, right, true), true, outcomes)
        }
        Expr::BinaryOp { left, op: BinaryOperator::Or, right } => {
            and_or(state, &support::and_or_terms(left, right, false), false, outcomes)
        }
        Expr::BinaryOp { left, op, right } => {
            if *op == BinaryOperator::Eq
                && let Some(operand) = state.dialect.null_equals_operand(left, right, state.options)
            {
                return is_null_input(state, operand, outcomes);
            }
            let Some(op) = CmpOp::of(op) else {
                return Ok(());
            };
            comparison(state, left, right, op, Family::Single, outcomes)
        }
        Expr::AnyOp { left, compare_op, right, .. } => {
            any_all(state, left, right, compare_op, Family::Any, outcomes)
        }
        Expr::AllOp { left, compare_op, right, .. } => {
            any_all(state, left, right, compare_op, Family::All, outcomes)
        }
        Expr::Between { expr: value, negated, low, high } => {
            between(state, value, low, high, *negated, outcomes)
        }
        Expr::In { expr: value, target, negated, .. } => {
            in_target(state, value, target, *negated, outcomes)
        }
        Expr::Function(call) => call_input(state, call, outcomes),
        Expr::Cast { expr: operand, data_type, .. } => {
            cast_input(state, operand, data_type, outcomes)
        }
        Expr::IsNull(inner) | Expr::IsUnknown(inner) => is_null_input(state, inner, outcomes),
        Expr::IsNotNull(inner) | Expr::IsNotUnknown(inner) => {
            is_null_input(state, inner, outcomes.negated())
        }
        Expr::IsTrue(inner) => boolean_value(state, inner, true, false, outcomes),
        Expr::IsNotTrue(inner) => boolean_value(state, inner, true, true, outcomes),
        Expr::IsFalse(inner) => boolean_value(state, inner, false, false, outcomes),
        Expr::IsNotFalse(inner) => boolean_value(state, inner, false, true, outcomes),
        Expr::IsDistinctFrom(left, right) => distinct(state, left, right, false, outcomes),
        Expr::IsNotDistinctFrom(left, right) => distinct(state, left, right, true, outcomes),
        _ => Ok(()),
    }
}

/// The inputs of the flat `AND`/`OR` chain of `terms` taking one outcome.
fn and_or<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    terms: &[&Expr],
    and: bool,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    // `FALSE` absorbs into an `AND` and `TRUE` into an `OR`, so the chain
    // takes them when every term does.
    if (and && outcomes.truth) || (!and && outcomes.falsity) {
        let outcome = if and { Outcomes::TRUE } else { Outcomes::FALSE };
        for term in terms {
            node(state, term, outcome)?;
        }
        return Ok(());
    }
    // `FALSE` in an `AND` and `TRUE` in an `OR` settle on one term alone,
    // leaving the others open; `NULL` needs the others clear of the opposite.
    let (distributed, side) = if and {
        if outcomes.null {
            (Outcomes::NULL, Some(Outcomes::NOT_FALSE))
        } else {
            (Outcomes::FALSE, None)
        }
    } else if outcomes.null {
        (Outcomes::NULL, Some(Outcomes::NOT_TRUE))
    } else {
        (Outcomes::TRUE, None)
    };
    disjunctive(state, terms, distributed, side)
}

/// One term of a flat chain takes `distributed` while the others stay
/// within `side`, when `side` holds; `FALSE` in an `AND` and `TRUE` in an
/// `OR` need no such read, since the outcome settles on one term alone.
fn disjunctive<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    terms: &[&Expr],
    distributed: Outcomes,
    side: Option<Outcomes>,
) -> Result<(), C::Error> {
    let count = terms.len();
    let mut branches: Vec<Guarantees<C::Reference>> = Vec::with_capacity(count);
    for &term in terms {
        branches.push(branch(state, |s| node(s, term, distributed))?);
    }
    let joined = match side {
        None => branches.into_iter().reduce(|a, b| a.join(&b)).unwrap_or_default(),
        Some(side) => {
            let mut sides: Vec<Guarantees<C::Reference>> = Vec::with_capacity(count);
            for &term in terms {
                sides.push(branch(state, |s| node(s, term, side))?);
            }
            disjunctive_side(&sides, &branches)
        }
    };
    *state.guarantees = state.guarantees.meet(&joined);
    Ok(())
}

/// The per-subject join of the side-and-distributed branches, walked per
/// subject so no term's read is recomputed per term. The pass runs only when
/// the chain reads its side outcome, so at least one branch stands and the
/// join keeps its reads.
fn disjunctive_side<R: Ord + Clone>(
    sides: &[Guarantees<R>],
    distributed: &[Guarantees<R>],
) -> Guarantees<R> {
    let count = sides.len();
    let side_impossible: Vec<bool> = sides.iter().map(Guarantees::is_impossible).collect();
    let mut prefix_or = vec![false; count];
    let mut suffix_or = vec![false; count];
    let mut running = false;
    for i in 0..count {
        prefix_or[i] = running;
        running |= side_impossible[i];
    }
    running = false;
    for i in (0..count).rev() {
        suffix_or[i] = running;
        running |= side_impossible[i];
    }
    // Branch `i` is impossible when its own read or any other term's side
    // read is impossible.
    let branch_impossible: Vec<bool> = (0..count)
        .map(|i| distributed[i].is_impossible() || prefix_or[i] || suffix_or[i])
        .collect();
    let non_impossible = branch_impossible.iter().filter(|&&impossible| !impossible).count();
    let mut readers: BTreeMap<Subject<R>, (Vec<usize>, Vec<usize>)> = BTreeMap::new();
    for (i, guarantees) in sides.iter().enumerate() {
        for (reference, _) in guarantees.references() {
            readers.entry(Subject::Reference(reference.clone())).or_default().0.push(i);
        }
        for (expression, _) in guarantees.expressions() {
            readers.entry(Subject::Expression(expression.clone())).or_default().0.push(i);
        }
    }
    for (i, guarantees) in distributed.iter().enumerate() {
        for (reference, _) in guarantees.references() {
            readers.entry(Subject::Reference(reference.clone())).or_default().1.push(i);
        }
        for (expression, _) in guarantees.expressions() {
            readers.entry(Subject::Expression(expression.clone())).or_default().1.push(i);
        }
    }
    readers.iter().fold(
        Guarantees::new(),
        |mut joined, (subject, (side_readers, distributed_readers))| {
            // A side read is drawn from the same branch as its distributed
            // one, so the branches that touch the subject are the distributed
            // readers.
            let non_impossible_in =
                distributed_readers.iter().filter(|&&i| !branch_impossible[i]).count();
            if non_impossible_in > 0 && non_impossible <= non_impossible_in {
                // Every non-impossible branch reads the subject; the join runs
                // over their reads, each meeting the other terms' sides.
                let mut side_values = Vec::with_capacity(side_readers.len() + 1);
                let mut running: Option<Evaluation> = None;
                for &i in side_readers {
                    side_values.push(running.clone());
                    let value = sides[i].get(subject);
                    running = running
                        .as_ref()
                        .zip(value)
                        .map(|(previous, value)| previous.meet(value))
                        .or_else(|| running.clone())
                        .or_else(|| value.cloned());
                }
                side_values.push(running);
                let mut value: Option<Evaluation> = None;
                for i in (0..count).filter(|&i| !branch_impossible[i]) {
                    let own = distributed[i].get(subject);
                    let position = side_readers.iter().position(|&read| read == i);
                    // `side_values` is the running meet over the other terms'
                    // sides, so the value at the subject's position is already
                    // the meet with everything that follows.
                    let without_own = match position {
                        Some(position) => {
                            side_values[position]
                                .clone()
                                .or_else(|| side_values[position + 1].clone())
                        }
                        None => side_values[side_values.len() - 1].clone(),
                    };
                    // Every non-impossible branch reads the subject, in its own
                    // read or in the other terms' sides, so the own read is
                    // present and the join keeps it.
                    let term_value = own.map(|own| {
                        without_own.as_ref().map_or(own.clone(), |without| own.meet(without))
                    });
                    value = value
                        .as_ref()
                        .zip(term_value.as_ref())
                        .map(|(value, term)| value.join(term))
                        .or(value.take())
                        .or_else(|| term_value.clone());
                }
                restrict_some(&mut joined, subject, value);
            }
            if non_impossible > non_impossible_in {
                // A non-impossible branch that leaves the subject open settles
                // it at the meet of the sides that read it.
                let value = meet_values(side_readers.iter().filter_map(|&i| sides[i].get(subject)));
                restrict_some(&mut joined, subject, value);
            }
            joined
        },
    )
}

/// Narrows `subject` to `value` when the reading settles one.
fn restrict_some<R: Ord + Clone>(
    guarantees: &mut Guarantees<R>,
    subject: &Subject<R>,
    value: Option<Evaluation>,
) {
    if let Some(value) = value {
        guarantees.restrict(subject.clone(), value);
    }
}

/// The meet of the present values, `None` when none holds.
fn meet_values<'a>(values: impl IntoIterator<Item = &'a Evaluation>) -> Option<Evaluation> {
    values.into_iter().fold(None, |previous, value| {
        Some(match previous {
            Some(previous) => previous.meet(value),
            None => value.clone(),
        })
    })
}

/// The inputs of `value BETWEEN low AND high` taking one outcome.
fn between<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    value: &Expr,
    low: &Expr,
    high: &Expr,
    negated: bool,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let low_holds = Expr::BinaryOp {
        left: Box::new(low.clone()),
        op: BinaryOperator::LtEq,
        right: Box::new(value.clone()),
    };
    let high_holds = Expr::BinaryOp {
        left: Box::new(value.clone()),
        op: BinaryOperator::LtEq,
        right: Box::new(high.clone()),
    };
    let outcomes = if negated { outcomes.negated() } else { outcomes };
    and_or(state, &[&low_holds, &high_holds], true, outcomes)
}

/// The inputs of `value IN target` taking one outcome.
fn in_target<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    value: &Expr,
    target: &InTarget,
    negated: bool,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let outcomes = if negated { outcomes.negated() } else { outcomes };
    match target {
        InTarget::List(members) => in_list(state, value, members, outcomes),
        InTarget::Subquery(subquery) => in_subquery(state, value, subquery, outcomes),
        InTarget::Unnest(_) | InTarget::Table { .. } => Ok(()),
    }
}

/// The inputs of `value IN (member, ...)` taking one outcome.
fn in_list<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    value: &Expr,
    members: &[Expr],
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let mut member_evaluations: Vec<Evaluation> = Vec::new();
    for member in members {
        member_evaluations.push(forward_of(state, member)?);
    }
    let any_null = member_evaluations.iter().any(Evaluation::may_be_null);
    let mut all_listed = true;
    let mut union = BTreeSet::new();
    for evaluation in &member_evaluations {
        if let Values::Listed(listed) = evaluation.values() {
            union.extend(listed.iter().cloned());
        } else {
            all_listed = false;
        }
    }
    // A constant member is present on every row, but a member that is not
    // a constant holds one of its values on each row. Its values settle a
    // `TRUE` reading, while only the constants keep `x` apart otherwise.
    let constants: BTreeSet<PgValue> =
        member_evaluations.iter().filter_map(Evaluation::constant).cloned().collect();
    let exact = member_evaluations
        .iter()
        .all(|member| member.constant().is_some() || member.is_always_null());
    let values = if exact || outcomes.truth {
        if all_listed { Values::Listed(union) } else { Values::Unlisted(Unlisted::any()) }
    } else if constants.is_empty() {
        Values::Unlisted(Unlisted::any())
    } else {
        Values::Listed(constants)
    };
    let other = typed_members(values, any_null);
    comparison_with(state, value, &other, CmpOp::Eq, Family::Any, outcomes)
}

/// The inputs of `value IN (SELECT ...)` taking one outcome.
fn in_subquery<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    value: &Expr,
    subquery: &Query,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let Some(column) = subquery_column(state, subquery, Family::Any, outcomes)? else {
        return Ok(());
    };
    // A row set holds any of the column's values, so the condition reads
    // against one row.
    comparison_with(state, value, &column, CmpOp::Eq, Family::Single, outcomes)
}

/// The column a quantified subquery operand compares against, `None` where
/// the subquery narrows nothing. A possibly empty row set reads `FALSE` to
/// `ANY` and `TRUE` to `ALL` whatever the left side is, so those outcomes
/// carry no input value.
fn subquery_column<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    subquery: &Query,
    family: Family,
    outcomes: Outcomes,
) -> Result<Option<Evaluation>, C::Error> {
    let qe = state.context.query(subquery, state.guarantees)?;
    let Some(column) = qe.columns.as_ref().and_then(|columns| columns.first()) else {
        return Ok(None);
    };
    let empty_reading = if family == Family::All { outcomes.truth } else { outcomes.falsity };
    if !(qe.rows.one || qe.rows.many) || qe.rows.none && empty_reading {
        return Ok(None);
    }
    Ok(Some(column.clone().with_error(qe.error)))
}

/// The inputs of `left op ANY/ALL (right)` taking one outcome.
fn any_all<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    right: &Expr,
    op: &BinaryOperator,
    family: Family,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    // An operator that is not a value comparison, such as a `LIKE ANY`
    // pattern match, implies no value restriction.
    let Some(op) = CmpOp::of(op) else {
        return Ok(());
    };
    let other = match support::unnest(right) {
        Expr::Subquery(subquery) => {
            let Some(column) = subquery_column(state, subquery, family, outcomes)? else {
                return Ok(());
            };
            // A row set holds any of the column's values, so the condition
            // reads against one row.
            return comparison_with(state, left, &column, op, Family::Single, outcomes);
        }
        Expr::Array(Array { elem, .. }) => {
            let (values, null) = listed_union(state, elem)?;
            // An element neither known nor `NULL` admits every value, so a
            // non-`NULL` member may exist.
            let nonempty = match &values {
                Values::Listed(listed) => !listed.is_empty(),
                Values::Unlisted(_) => true,
            };
            if !nonempty {
                // The empty and `NULL`-only list outcomes are decided by the
                // forward pass.
                return Ok(());
            }
            typed_members(values, null)
        }
        Expr::Cast { expr: operand, .. } if matches!(support::unnest(operand), Expr::Array(_)) => {
            // A cast of an array, folded member by member by the forward
            // pass, such as PostgreSQL's `ARRAY[...]::text[]`.
            let evaluation = forward_of(state, right)?;
            let Values::Listed(listed) = evaluation.values() else {
                return Ok(());
            };
            let nonempty = !listed.is_empty();
            if !nonempty {
                // The cast-array forward is unknown, so this, not the prune,
                // decides the outcome.
                let takes = if evaluation.may_be_null() {
                    outcomes.null
                } else if family == Family::All {
                    outcomes.truth
                } else {
                    outcomes.falsity
                };
                if !takes {
                    impossible(state);
                }
                return Ok(());
            }
            typed_members(Values::Listed(listed.clone()), evaluation.may_be_null())
        }
        _ => return Ok(()),
    };
    comparison_with(state, left, &other, op, family, outcomes)
}

/// The union of the elements' values with their `NULL` possibility,
/// unknown when an element is neither a value nor always `NULL`.
fn listed_union<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    elements: &[Expr],
) -> Result<(Values, bool), C::Error> {
    let mut union = BTreeSet::new();
    let mut null = false;
    for element in elements {
        let evaluation = forward_of(state, element)?;
        match evaluation.constant() {
            Some(datum) => {
                union.insert(datum.clone());
            }
            None if evaluation.is_always_null() => null = true,
            None => return Ok((Values::Unlisted(Unlisted::any()), true)),
        }
    }
    Ok((Values::Listed(union), null))
}

/// The members of an `IN` list or an `ANY`/`ALL` array, typed by their
/// constants so each compares under its own type: integers as `bigint`,
/// which holds every one, integers among `numeric`s as `numeric`s, and text
/// untyped, as the literal a column reads under its own type.
fn typed_members(values: Values, null: bool) -> Evaluation {
    let Values::Listed(listed) = values else {
        return Evaluation::new(ScalarType::Unknown, null, false, values);
    };
    let kind = |value: &PgValue| {
        match value {
            PgValue::Integer(_) => ScalarType::Integer(IntegerType::Bigint),
            PgValue::Numeric(_) => ScalarType::Numeric,
            PgValue::Boolean(_) => ScalarType::Boolean,
            PgValue::Date(_) => ScalarType::Date,
            PgValue::Timestamp(_) => ScalarType::Timestamp,
            PgValue::Text(_) | PgValue::Array(_) => ScalarType::Unknown,
        }
    };
    let ty = listed.iter().map(kind).reduce(|ty, next| {
        match (ty, next) {
            _ if ty == next => ty,
            (
                ScalarType::Integer(_) | ScalarType::Numeric,
                ScalarType::Integer(_) | ScalarType::Numeric,
            ) => ScalarType::Numeric,
            _ => ScalarType::Unknown,
        }
    });
    let ty = ty.unwrap_or(ScalarType::Unknown);
    let listed = if ty == ScalarType::Numeric {
        listed
            .into_iter()
            .map(|value| {
                match value {
                    PgValue::Integer(integer) => PgValue::Numeric(PgNumeric::from(integer)),
                    other => other,
                }
            })
            .collect()
    } else {
        listed
    };
    Evaluation::new(ty, null, false, Values::Listed(listed))
}

/// A branch in which `value` is non-`NULL` and restricted by `restrict` to
/// `other`'s listed values, apart from them or among them.
fn member_branch<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    value: &Expr,
    other: &Evaluation,
    restrict: fn(BTreeSet<PgValue>) -> Restriction,
) -> Result<Guarantees<C::Reference>, C::Error> {
    branch(state, |s| {
        restrict_non_null(s, value)?;
        let evaluation = forward_of(s, value)?;
        let restriction = match other.values() {
            Values::Listed(listed) if !listed.is_empty() => restrict(listed.clone()),
            _ => Restriction::Skip,
        };
        apply(s, value, &evaluation, &restriction)
    })
}

/// The inputs of a call taking one outcome, by its callees' null behaviour.
fn call_input<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    call: &Function,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    if call.over.is_some() {
        return Ok(());
    }
    let arguments = support::call_arguments(call);
    if support::simple_name(&call.name).as_deref() == Some("coalesce") {
        // The result is `NULL` exactly when every argument is `NULL`.
        if outcomes.null {
            for argument in arguments.iter().flatten() {
                restrict_none(state, argument)?;
            }
        }
        return Ok(());
    }
    let callees = state.context.callees(call)?;
    // A strict callee's `NULL` argument forces the result `NULL` only when
    // every callee the call may reach is strict.
    let all_strict = !callees.is_empty()
        && callees.iter().all(|callee| matches!(callee.null_input, NullInput::Strict));
    // The arguments some reachable callee answers `NULL` for, each once.
    let mut positions = BTreeSet::new();
    for callee in &callees {
        if callee.null_input == NullInput::Strict && !all_strict {
            continue;
        }
        positions.extend((0..arguments.len()).filter(|&index| callee.null_input.nulls_at(index)));
    }
    let nulling = positions.into_iter().filter_map(|position| arguments[position].as_ref());
    if outcomes.truth || outcomes.falsity {
        for argument in nulling {
            restrict_non_null(state, argument)?;
        }
        return Ok(());
    }
    // A `NULL` result points at a `NULL` argument only for the built-ins
    // known to return `NULL` exactly on a `NULL` argument.
    let null_only_on_null_input =
        !callees.is_empty() && state.dialect.null_only_on_null_input(&callees, state.options);
    if !null_only_on_null_input {
        return Ok(());
    }
    let mut branches: Vec<Guarantees<C::Reference>> = Vec::new();
    for argument in nulling {
        branches.push(branch(state, |s| restrict_none(s, argument))?);
    }
    if !branches.is_empty() {
        merge(state, branches);
    }
    Ok(())
}

/// The operand of a cast taking one outcome.
fn cast_input<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    operand: &Expr,
    data_type: &DataType,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let to = state.context.data_type(data_type);
    let operand_evaluation = forward_of(state, operand)?;
    if operand_evaluation.ty() == to || (to.is_text() && cast_text_limit(data_type).is_none()) {
        return node(state, operand, outcomes);
    }
    if outcomes.truth || outcomes.falsity {
        return restrict_non_null(state, operand);
    }
    restrict_none(state, operand)
}

/// The character limit a `varchar` typmod states, when it names one.
fn cast_text_limit(data_type: &DataType) -> Option<usize> {
    let (DataType::Varchar(length)
    | DataType::CharacterVarying(length)
    | DataType::CharVarying(length)) = data_type
    else {
        return None;
    };
    match length {
        Some(CharacterLength::IntegerLength { length, .. }) => usize::try_from(*length).ok(),
        _ => None,
    }
}

/// The operand of an `IS [NOT] NULL` taking one outcome.
fn is_null_input<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    inner: &Expr,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    // The test reads `true` or `false` alone, so the single outcome
    // admitted after the forward pass admits one of its values.
    if outcomes.truth { restrict_none(state, inner) } else { restrict_non_null(state, inner) }
}

/// The inner of an `IS [NOT] TRUE/FALSE` taking one outcome.
fn boolean_value<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    inner: &Expr,
    is_true_test: bool,
    negated: bool,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    // The test reads `true` or `false` alone, so the single outcome
    // admitted after the forward pass admits one of its values.
    let value = if (outcomes.truth && !negated) || (outcomes.falsity && negated) {
        is_true_test
    } else {
        !is_true_test
    };
    let evaluation = forward_of(state, inner)?;
    // Only a `boolean` inner narrows to a `boolean` value.
    let narrowed = if evaluation.ty() == ScalarType::Boolean {
        evaluation.with_values(Values::Listed(BTreeSet::from([PgValue::Boolean(value)])))
    } else {
        evaluation
    };
    // `IS NOT TRUE` (and `IS NOT FALSE`) read `true` for a `NULL` inner, so
    // the `true` outcome of a negated test keeps `NULL` admitted.
    let narrowed = if outcomes.truth && !negated { narrowed.with_null(false) } else { narrowed };
    restrict_subject(state, inner, narrowed)
}

/// The inputs of `x IS [NOT] DISTINCT FROM y` when the predicate is true.
fn distinct<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    right: &Expr,
    negated: bool,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    if !outcomes.truth {
        return Ok(());
    }
    if negated {
        // Both `NULL`, or both non-`NULL` and equal.
        let both_null = branch(state, |s| {
            restrict_none(s, left)?;
            restrict_none(s, right)
        })?;
        let equal = branch(state, |s| {
            restrict_non_null(s, left)?;
            restrict_non_null(s, right)?;
            comparison(s, left, right, CmpOp::Eq, Family::Single, Outcomes::TRUE)
        })?;
        merge(state, Vec::from([both_null, equal]));
        return Ok(());
    }
    // Exactly one `NULL`, or both non-`NULL` and different.
    let left_null = branch(state, |s| {
        restrict_none(s, left)?;
        restrict_non_null(s, right)
    })?;
    let right_null = branch(state, |s| {
        restrict_none(s, right)?;
        restrict_non_null(s, left)
    })?;
    let different = branch(state, |s| {
        restrict_non_null(s, left)?;
        restrict_non_null(s, right)?;
        comparison(s, left, right, CmpOp::Eq, Family::Single, Outcomes::FALSE)
    })?;
    merge(state, Vec::from([left_null, right_null, different]));
    Ok(())
}

/// The inputs of `left op right` taking one outcome.
fn comparison<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    right: &Expr,
    op: CmpOp,
    family: Family,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    let (left, right) = (support::unnest(left), support::unnest(right));
    if outcomes.null {
        let branches = [side_null_branch(state, left)?, side_null_branch(state, right)?];
        merge(state, Vec::from(branches));
        return Ok(());
    }
    restrict_non_null(state, left)?;
    restrict_non_null(state, right)?;
    let left_evaluation = forward_of(state, left)?;
    let right_evaluation = forward_of(state, right)?;
    let left_restriction =
        restriction(family, op, outcomes.truth, &right_evaluation, &left_evaluation);
    apply(state, left, &left_evaluation, &left_restriction)?;
    let right_restriction =
        restriction(family, op.flipped(), outcomes.truth, &left_evaluation, &right_evaluation);
    apply(state, right, &right_evaluation, &right_restriction)?;
    // A length-like call's argument carries the constraint on its length.
    if let Expr::Function(call) = left
        && let Some(arg) = length_argument(state, call)?
        && let Some(length) = length_restriction(family, op, outcomes.truth, &right_evaluation)
    {
        apply_length(state, arg, &length)?;
    }
    if let Expr::Function(call) = right
        && let Some(arg) = length_argument(state, call)?
        && let Some(length) =
            length_restriction(family, op.flipped(), outcomes.truth, &left_evaluation)
    {
        apply_length(state, arg, &length)?;
    }
    Ok(())
}

/// The input narrowing of `left op right`, where `right` is evaluated
/// already and stands under `family`'s quantification.
fn comparison_with<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    right: &Evaluation,
    op: CmpOp,
    family: Family,
    outcomes: Outcomes,
) -> Result<(), C::Error> {
    if outcomes.null {
        let mut branches = Vec::from([side_null_branch(state, left)?]);
        // A `NULL` member reads `NULL` unless another member settles it.
        if right.may_be_null() {
            let branch = match (family, op) {
                (Family::Any, CmpOp::Eq) | (Family::All, CmpOp::NotEq) => {
                    member_branch(state, left, right, Restriction::Exclude)?
                }
                (Family::Any, CmpOp::NotEq) | (Family::All, CmpOp::Eq) => {
                    member_branch(state, left, right, Restriction::Keep)?
                }
                _ => branch(state, |s| restrict_non_null(s, left))?,
            };
            branches.push(branch);
        }
        merge(state, branches);
        return Ok(());
    }
    restrict_non_null(state, left)?;
    let left_evaluation = forward_of(state, left)?;
    let restriction = restriction(family, op, outcomes.truth, right, &left_evaluation);
    apply(state, left, &left_evaluation, &restriction)
}

/// A branch in which `side` is `NULL`, through its strict inputs when it
/// is a call or a comparison.
fn side_null_branch<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    side: &Expr,
) -> Result<Guarantees<C::Reference>, C::Error> {
    let side = support::unnest(side);
    let side_evaluation = forward_of(state, side)?;
    if !side_evaluation.may_be_null() {
        return Ok(Guarantees::impossible());
    }
    branch(state, |s| {
        match side {
            Expr::Function(call) => {
                if let Some(arg) = length_argument(s, call)? {
                    return restrict_none(s, arg);
                }
                let arguments = support::call_arguments(call);
                let callees = s.context.callees(call)?;
                // A strict function may still return `NULL` for non-`NULL`
                // input, so only the built-ins known to return `NULL` exactly
                // on a `NULL` argument tell which input was `NULL`.
                let all_strict = !callees.is_empty()
                    && callees.iter().all(|callee| callee.null_input == NullInput::Strict);
                let null_only_on_null_input =
                    all_strict && s.dialect.null_only_on_null_input(&callees, s.options);
                if !null_only_on_null_input {
                    return restrict_none(s, side);
                }
                let mut branches: Vec<Guarantees<C::Reference>> = Vec::new();
                for argument in arguments.iter().flatten() {
                    branches.push(branch(s, |s2| restrict_none(s2, argument))?);
                }
                if branches.is_empty() {
                    return restrict_none(s, side);
                }
                merge(s, branches);
                Ok(())
            }
            Expr::BinaryOp {
                left,
                op:
                    BinaryOperator::Lt
                    | BinaryOperator::LtEq
                    | BinaryOperator::Gt
                    | BinaryOperator::GtEq
                    | BinaryOperator::Eq
                    | BinaryOperator::NotEq,
                right,
            } => {
                let branches = [side_null_branch(s, left)?, side_null_branch(s, right)?];
                merge(s, Vec::from(branches));
                Ok(())
            }
            _ => restrict_none(s, side),
        }
    })
}

/// The operator a comparison restricts with, the way the restriction reads.
#[derive(Clone, Copy, PartialEq, Eq)]
enum CmpOp {
    /// `=`.
    Eq,
    /// `<>`.
    NotEq,
    /// `<` when `strict`, `<=` otherwise.
    Less { strict: bool },
    /// `>` when `strict`, `>=` otherwise.
    Greater { strict: bool },
}

impl CmpOp {
    /// The operator that `operator` reads as a value comparison, `None`
    /// where it compares no values, such as a `LIKE` pattern match.
    fn of(operator: &BinaryOperator) -> Option<Self> {
        Some(match operator {
            BinaryOperator::Eq => CmpOp::Eq,
            BinaryOperator::NotEq => CmpOp::NotEq,
            BinaryOperator::Lt => CmpOp::Less { strict: true },
            BinaryOperator::LtEq => CmpOp::Less { strict: false },
            BinaryOperator::Gt => CmpOp::Greater { strict: true },
            BinaryOperator::GtEq => CmpOp::Greater { strict: false },
            _ => return None,
        })
    }

    /// The operator the comparison reads as from the other side.
    fn flipped(self) -> Self {
        match self {
            CmpOp::Eq => CmpOp::Eq,
            CmpOp::NotEq => CmpOp::NotEq,
            CmpOp::Less { strict } => CmpOp::Greater { strict },
            CmpOp::Greater { strict } => CmpOp::Less { strict },
        }
    }
}

/// The quantification a comparison's right side stands under.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// A plain comparison of two single values.
    Single,
    /// `ANY` semantics: some right value.
    Any,
    /// `ALL` semantics: every right value.
    All,
}

/// One side of a value constraint.
#[derive(Clone)]
struct Side {
    value: PgValue,
    strict: bool,
}

/// How an outcome restricts the other side of a comparison.
enum Restriction {
    /// The values must lie in a numeric interval.
    Interval(Interval<PgNumeric>),
    /// The values must be among a set.
    Keep(BTreeSet<PgValue>),
    /// The values must be outside a set.
    Exclude(BTreeSet<PgValue>),
    /// The values must lie between datum bounds.
    Side { low: Option<Side>, high: Option<Side> },
    /// No value satisfies the outcome.
    Dead,
    /// The outcome imposes nothing knowable.
    Skip,
}

/// How `left op right` taking `is_true` restricts the left side, from the
/// right side's values.
fn restriction(
    family: Family,
    op: CmpOp,
    is_true: bool,
    right: &Evaluation,
    left: &Evaluation,
) -> Restriction {
    if !compatible(left.ty(), right.ty()) {
        return Restriction::Skip;
    }
    let untyped = |ty: ScalarType| matches!(ty, ScalarType::Unknown | ScalarType::UnknownLiteral);
    // A typed value side must not meet an untyped literal's raw datum.
    if left.ty() == ScalarType::UnknownLiteral && !untyped(right.ty()) {
        return Restriction::Skip;
    }
    // An untyped value side reads its values as the other side's type.
    let reinterpreted;
    let right = if untyped(right.ty()) && !untyped(left.ty()) && !left.ty().is_text() {
        let Some(values) = reinterpreted_values(right, left.ty()) else {
            return Restriction::Skip;
        };
        reinterpreted = Evaluation::new(
            left.ty(),
            right.may_be_null(),
            right.may_raise(),
            Values::Listed(values),
        );
        &reinterpreted
    } else {
        right
    };
    match op {
        CmpOp::Eq | CmpOp::NotEq => {
            // A non-identity collation can equate distinct strings, so a
            // membership guarantee needs identity equality on both sides.
            let identity = |ty: ScalarType| {
                match ty {
                    ScalarType::Text(rules) => rules.identity_equality,
                    _ => true,
                }
            };
            if !identity(left.ty()) || !identity(right.ty()) {
                return Restriction::Skip;
            }
            equality(family, op, is_true, right)
        }
        CmpOp::Less { strict } | CmpOp::Greater { strict } => {
            let greater = matches!(op, CmpOp::Greater { .. });
            if right.ty().is_number() {
                ordering_numeric(family, greater, strict, is_true, &right.number_range())
            } else if orderable(left.ty(), right.ty()) {
                if let Values::Listed(listed) = right.values()
                    && let (Some(min), Some(max)) =
                        (listed.iter().next(), listed.iter().next_back())
                {
                    ordering_datum(family, greater, strict, is_true, min, max)
                } else {
                    Restriction::Skip
                }
            } else {
                Restriction::Skip
            }
        }
    }
}

/// The values of `source` re-read as `target`, `None` when one stays undecided.
fn reinterpreted_values(source: &Evaluation, target: ScalarType) -> Option<BTreeSet<PgValue>> {
    let Values::Listed(listed) = source.values() else {
        return None;
    };
    let mut values = BTreeSet::new();
    for datum in listed {
        match crate::scalar::cast(datum, source.ty(), target) {
            crate::scalar::Cast::Folded(value) => {
                values.insert(value);
            }
            crate::scalar::Cast::Raises => {}
            crate::scalar::Cast::Unknown => {
                return None;
            }
        }
    }
    Some(values)
}

/// Whether the categories can compare as values.
fn compatible(left: ScalarType, right: ScalarType) -> bool {
    if left == right {
        return true;
    }
    matches!(left, ScalarType::Unknown | ScalarType::UnknownLiteral)
        || matches!(right, ScalarType::Unknown | ScalarType::UnknownLiteral)
        || (left.is_number() && right.is_number())
        || (left.is_text() && right.is_text())
}

/// Whether the categories compare by value order.
fn orderable(left: ScalarType, right: ScalarType) -> bool {
    match (left, right) {
        (ScalarType::Text(left), ScalarType::Text(right)) => left.byte_order && right.byte_order,
        (ScalarType::Date, ScalarType::Date) | (ScalarType::Timestamp, ScalarType::Timestamp) => {
            true
        }
        _ => false,
    }
}

/// The set restriction of an equality outcome.
fn equality(family: Family, op: CmpOp, is_true: bool, right: &Evaluation) -> Restriction {
    let set = match right.values() {
        Values::Listed(listed) if !listed.is_empty() => Some(listed.clone()),
        // A value-less other side reads `NULL` everywhere, so no input takes
        // the outcome.
        Values::Listed(_) => return Restriction::Dead,
        Values::Unlisted(_) => {
            match point_of(&right.number_range()) {
                Some(value) if right.ty().is_number() => {
                    Some(BTreeSet::from([PgValue::Numeric(value)]))
                }
                _ => None,
            }
        }
    };
    let Some(set) = set else {
        return Restriction::Skip;
    };
    // A one-member set reads the same as the single comparison under every
    // quantifier.
    if set.len() == 1 {
        match (family, op, is_true) {
            (_, CmpOp::Eq, true) | (_, CmpOp::NotEq, false) => Restriction::Keep(set),
            _ => Restriction::Exclude(set),
        }
    } else {
        // A value the other side holds is apart from some other member, so
        // a single `<>` or an `= ALL` negation it satisfies is not ruled
        // out by its membership.
        match (family, op, is_true) {
            (Family::All, CmpOp::Eq, true) | (Family::Any, CmpOp::NotEq, false) => {
                Restriction::Dead
            }
            (Family::Single | Family::Any, CmpOp::Eq, true)
            | (Family::Single | Family::All, CmpOp::NotEq, false) => Restriction::Keep(set),
            (Family::Any, CmpOp::Eq, false) | (Family::All, CmpOp::NotEq, true) => {
                Restriction::Exclude(set)
            }
            _ => Restriction::Skip,
        }
    }
}

/// The value an interval holds, when it holds exactly one.
fn point_of(interval: &Interval<PgNumeric>) -> Option<PgNumeric> {
    match (&interval.low, &interval.high) {
        (Bound::Included(low), Bound::Included(high)) if low == high => Some(low.clone()),
        _ => None,
    }
}

/// Whether the other's bound stays closed, open or flips.
#[derive(Clone, Copy, PartialEq, Eq)]
enum BoundKind {
    /// Always strict at the other's value.
    Strict,
    /// Inclusive when the other's bound is closed, strict when it is open.
    Follows,
    /// Inclusive when the other's bound is open, strict when it is closed.
    Inverts,
    /// Always inclusive at the other's value.
    AlwaysIn,
}

/// The bound a constraint takes from the other's bound.
fn mapped(bound: &Bound<PgNumeric>, kind: BoundKind) -> Option<Bound<PgNumeric>> {
    match (bound, kind) {
        (Bound::Unbounded, _) => None,
        (Bound::Included(value) | Bound::Excluded(value), BoundKind::Strict)
        | (Bound::Excluded(value), BoundKind::Follows)
        | (Bound::Included(value), BoundKind::Inverts) => Some(Bound::Excluded(value.clone())),
        (Bound::Included(value), BoundKind::Follows | BoundKind::AlwaysIn)
        | (Bound::Excluded(value), BoundKind::Inverts | BoundKind::AlwaysIn) => {
            Some(Bound::Included(value.clone()))
        }
    }
}

/// An end of an interval or of a listed set.
#[derive(Clone, Copy)]
enum End {
    Low,
    High,
}

/// Where an ordering outcome bounds the left side from the right's values.
struct OrderingBound {
    /// The end of the right side that bounds the left.
    from: End,
    /// The end of the left side it bounds.
    onto: End,
    /// Whether the bound excludes the right's value itself.
    strict: bool,
    /// Whether the left must stand past every right value, as for `ANY`
    /// taking `false` and `ALL` taking `true`, not past one of them.
    every: bool,
}

/// The bound an ordering outcome sets, `x > right` when `greater` and
/// `x < right` otherwise, strict when `strict`.
fn ordering_bound(family: Family, greater: bool, strict: bool, is_true: bool) -> OrderingBound {
    let every = matches!((family, is_true), (Family::Any, false) | (Family::All, true));
    // A true `x > v` bounds `x` from below, a false one from above.
    let onto_low = greater == is_true;
    // One right value bounds the left at its near end, every right value at
    // its far end.
    let from_low = onto_low != every;
    let end = |low: bool| if low { End::Low } else { End::High };
    // A false outcome holds the negated comparison, whose strictness flips.
    OrderingBound { from: end(from_low), onto: end(onto_low), strict: strict == is_true, every }
}

/// The interval a numeric ordering outcome restricts the left side to.
fn ordering_numeric(
    family: Family,
    greater: bool,
    strict: bool,
    is_true: bool,
    range: &Interval<PgNumeric>,
) -> Restriction {
    let bound = ordering_bound(family, greater, strict, is_true);
    let from = match bound.from {
        End::Low => &range.low,
        End::High => &range.high,
    };
    let kind = match (bound.every, bound.strict) {
        (false, true) => BoundKind::Strict,
        (false, false) => BoundKind::Follows,
        (true, true) => BoundKind::Inverts,
        (true, false) => BoundKind::AlwaysIn,
    };
    let mapped = mapped(from, kind).unwrap_or(Bound::Unbounded);
    Restriction::Interval(match bound.onto {
        End::Low => Interval { low: mapped, high: Bound::Unbounded },
        End::High => Interval { low: Bound::Unbounded, high: mapped },
    })
}

/// The datum bounds a listed right side sets on the left.
fn ordering_datum(
    family: Family,
    greater: bool,
    strict: bool,
    is_true: bool,
    min: &PgValue,
    max: &PgValue,
) -> Restriction {
    let bound = ordering_bound(family, greater, strict, is_true);
    let value = match bound.from {
        End::Low => min,
        End::High => max,
    };
    let side = Some(Side { value: value.clone(), strict: bound.strict });
    match bound.onto {
        End::Low => Restriction::Side { low: side, high: None },
        End::High => Restriction::Side { low: None, high: side },
    }
}

/// Restricts `expression`'s subject by `restriction`, against its values.
fn apply<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expression: &Expr,
    evaluation: &Evaluation,
    restriction: &Restriction,
) -> Result<(), C::Error> {
    let Some(values) = apply_values(evaluation, restriction) else {
        impossible(state);
        return Ok(());
    };
    restrict_subject(state, expression, evaluation.clone().with_values(values))
}

/// `evaluation`'s values narrowed by `restriction`, `None` when nothing
/// satisfies it.
fn apply_values(evaluation: &Evaluation, restriction: &Restriction) -> Option<Values> {
    match restriction {
        Restriction::Skip => Some(evaluation.values().clone()),
        Restriction::Dead => None,
        Restriction::Interval(range) => apply_interval(evaluation, range),
        Restriction::Keep(set) => Some(apply_set(evaluation, set, true)),
        Restriction::Exclude(set) => Some(apply_set(evaluation, set, false)),
        Restriction::Side { low, high } => {
            Some(apply_side(evaluation, low.as_ref(), high.as_ref()))
        }
    }
}

/// The values of `evaluation` inside `range`, `None` when none lie in it.
fn apply_interval(evaluation: &Evaluation, range: &Interval<PgNumeric>) -> Option<Values> {
    match evaluation.values() {
        Values::Listed(listed) => {
            Some(Values::Listed(support::filter_listed_by_interval(listed, range)))
        }
        Values::Unlisted(unlisted) => {
            let numbers = unlisted.numbers.meet(range);
            (!numbers.is_empty()).then(|| {
                Values::Unlisted(Unlisted {
                    excluded: unlisted.excluded.clone(),
                    numbers,
                    length: unlisted.length.clone(),
                })
            })
        }
    }
}

/// The values of `evaluation` kept to, or dropped from, `set`.
fn apply_set(evaluation: &Evaluation, set: &BTreeSet<PgValue>, keep: bool) -> Values {
    match evaluation.values() {
        Values::Listed(listed) => {
            let kept: BTreeSet<PgValue> =
                listed.iter().filter(|value| keep == set.contains(value)).cloned().collect();
            Values::Listed(kept)
        }
        Values::Unlisted(unlisted) => {
            if keep {
                let mut kept = set.clone();
                kept.retain(|value| !unlisted.excluded.contains(value));
                Values::Listed(kept)
            } else {
                let mut widened = unlisted.clone();
                widened.excluded.extend(set.iter().cloned());
                Values::Unlisted(widened)
            }
        }
    }
}

/// The listed values between datum bounds; unknown values stay untouched.
fn apply_side(evaluation: &Evaluation, low: Option<&Side>, high: Option<&Side>) -> Values {
    let Values::Listed(listed) = evaluation.values() else {
        return evaluation.values().clone();
    };
    let kept: BTreeSet<PgValue> = listed
        .iter()
        .filter(|value| {
            let value = *value;
            let above_low = match low {
                Some(Side { value: bound, strict: true }) => value > bound,
                Some(Side { value: bound, strict: false }) => value >= bound,
                None => true,
            };
            above_low
                && match high {
                    Some(Side { value: bound, strict: true }) => value < bound,
                    Some(Side { value: bound, strict: false }) => value <= bound,
                    None => true,
                }
        })
        .cloned()
        .collect();
    Values::Listed(kept)
}

/// The argument whose characters a call counts under the dialect, `None`
/// when the call does not count its argument.
fn length_argument<'e, D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    call: &'e Function,
) -> Result<Option<&'e Expr>, C::Error> {
    let callees = state.context.callees(call)?;
    let Some(position) = state.dialect.counted_length_argument(call, &callees, state.options)
    else {
        return Ok(None);
    };
    Ok(argument_at(call, position).map(support::unnest))
}

/// The argument expression at `position` of a call, `None` where an
/// unsupported form stands in.
fn argument_at(call: &Function, position: usize) -> Option<&Expr> {
    let FunctionArguments::List(list) = &call.args else {
        return None;
    };
    list.args.get(position).and_then(support::argument_expression)
}

/// The restriction an ordering outcome sets on a length-like call's result.
fn length_restriction(
    family: Family,
    op: CmpOp,
    is_true: bool,
    other: &Evaluation,
) -> Option<Interval<PgNumeric>> {
    let left = Evaluation::unknown(ScalarType::Numeric);
    match restriction(family, op, is_true, other, &left) {
        Restriction::Interval(range) => Some(range),
        Restriction::Keep(set) if set.len() == 1 => {
            let value = set.iter().next()?.as_numeric()?;
            Some(Interval::point(value))
        }
        _ => None,
    }
}

/// Restricts the argument of a length-like call to the lengths `range`
/// admits, making it non-`NULL`.
fn apply_length<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    arg: &Expr,
    range: &Interval<PgNumeric>,
) -> Result<(), C::Error> {
    let evaluation = forward_of(state, arg)?;
    if matches!(evaluation.ty(), ScalarType::Unknown | ScalarType::Other) {
        // A type the evaluator does not model, such as `char(n)`, may hold
        // text of any length.
        return Ok(());
    }
    if !evaluation.ty().is_text() && evaluation.ty() != ScalarType::UnknownLiteral {
        // A length only bounds text; anything else cannot satisfy the
        // outcome.
        impossible(state);
        return Ok(());
    }
    let Some(values) = apply_length_values(&evaluation, range) else {
        impossible(state);
        return Ok(());
    };
    restrict_subject(state, arg, evaluation.clone().with_values(values).with_null(false))
}

/// The values of `evaluation` whose lengths `range` admits.
fn apply_length_values(evaluation: &Evaluation, range: &Interval<PgNumeric>) -> Option<Values> {
    match evaluation.values() {
        Values::Listed(listed) => {
            let kept: BTreeSet<PgValue> = listed
                .iter()
                .filter_map(|value| {
                    let length = crate::scalar::text_length(value)?;
                    let number = support::count_numeric(length)?;
                    range.contains(&number).then(|| value.clone())
                })
                .collect();
            Some(Values::Listed(kept))
        }
        Values::Unlisted(unlisted) => {
            let narrowed = u64_range_to_numeric(&unlisted.length).meet(range);
            if narrowed.is_empty() {
                return None;
            }
            Some(Values::Unlisted(Unlisted {
                excluded: unlisted.excluded.clone(),
                numbers: unlisted.numbers.clone(),
                length: numeric_to_u64_range(&narrowed),
            }))
        }
    }
}

/// The numeric interval a character-count interval lies in.
fn u64_range_to_numeric(range: &Interval<u64>) -> Interval<PgNumeric> {
    Interval { low: u64_bound(&range.low), high: u64_bound(&range.high) }
}

/// The bound a character-count bound takes in numerics.
fn u64_bound(bound: &Bound<u64>) -> Bound<PgNumeric> {
    match bound {
        Bound::Unbounded => Bound::Unbounded,
        bound @ (Bound::Included(value) | Bound::Excluded(value)) => {
            match i64::try_from(*value) {
                Ok(value) => {
                    match bound {
                        Bound::Included(_) => Bound::Included(PgNumeric::from(value)),
                        _ => Bound::Excluded(PgNumeric::from(value)),
                    }
                }
                Err(_) => Bound::Unbounded,
            }
        }
    }
}

/// The character-count interval a numeric length interval lies in.
fn numeric_to_u64_range(range: &Interval<PgNumeric>) -> Interval<u64> {
    Interval { low: numeric_bound(&range.low), high: numeric_bound(&range.high) }
}

/// The bound a numeric length bound takes in character counts.
fn numeric_bound(bound: &Bound<PgNumeric>) -> Bound<u64> {
    match bound {
        Bound::Unbounded => Bound::Unbounded,
        bound @ (Bound::Included(value) | Bound::Excluded(value)) => {
            match crate::scalar::exact_i64(value) {
                Some(value) if value >= 0 => {
                    // The non-negative guard makes the conversion lossless.
                    let length = value.unsigned_abs();
                    match bound {
                        Bound::Included(_) => Bound::Included(length),
                        _ => Bound::Excluded(length),
                    }
                }
                _ => Bound::Unbounded,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use sql_values::postgres::IntegerType;
    use sqlparser::dialect::PostgreSqlDialect;

    use super::*;
    use crate::{
        Builtin, Guarantees, Outcomes, Rows, ScalarType, TextRules, Values,
        postgres::PostgreSqlOptions,
        test_context::{ColumnContext, parse, strict_callee},
    };

    /// A nullable, non-raising column evaluation.
    fn unknown_column(ty: ScalarType) -> Evaluation {
        Evaluation::new(ty, true, false, Values::Unlisted(Unlisted::any()))
    }

    /// A never-null, non-raising column evaluation.
    fn plain_column(ty: ScalarType) -> Evaluation {
        Evaluation::new(ty, false, false, Values::Unlisted(Unlisted::any()))
    }

    /// A boolean routine that runs on `NULL` input and returns `NULL` for a
    /// `NULL` at one of `positions`.
    fn called_on_null(positions: Vec<usize>) -> crate::Callee {
        crate::context::Callee {
            builtin: None,
            null_input: crate::context::NullInput::CalledOnNull(positions),
            volatility: crate::context::Volatility::Immutable,
            kind: crate::context::CalleeKind::Normal,
            returns: ScalarType::Boolean,
            returns_set: false,
            token: 1,
        }
    }

    /// PostgreSQL's `length(text)`.
    fn length_callee() -> crate::Callee {
        let mut callee = crate::test_context::strict_callee(ScalarType::Integer(
            sql_values::postgres::IntegerType::Integer,
        ));
        callee.builtin = Some(Builtin::Length);
        callee
    }

    fn implied_with(
        context: ColumnContext,
        expression: &Expr,
        outcomes: Outcomes,
    ) -> Guarantees<String> {
        let mut context = context;
        implied(
            expression,
            outcomes,
            &PostgreSqlDialect {},
            &PostgreSqlOptions::default(),
            &mut context,
            &Guarantees::new(),
        )
        .expect("the context resolves")
    }

    fn column<'a>(guarantees: &'a Guarantees<String>, name: &str) -> &'a Evaluation {
        let evaluation = guarantees.reference(&name.to_string());
        assert!(evaluation.is_some(), "the guarantee keeps {name}");
        evaluation.unwrap()
    }

    /// Whether the subject may be `NULL`, reading an unrestricted subject as
    /// nullable.
    fn nullable(guarantees: &Guarantees<String>, name: &str) -> bool {
        guarantees.reference(&name.to_string()).is_none_or(Evaluation::may_be_null)
    }

    #[test]
    fn in_list_keeps_the_members() {
        let expression = parse("x IN ('a', 'b')");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> =
            [PgValue::Text("a".to_string()), PgValue::Text("b".to_string())].into();
        assert_eq!(listed, &expected);
    }

    /// Typed members compare under their own type, so a member past the
    /// column's range or with a fraction is a value no row holds, never a
    /// raise or a rounded one.
    #[test]
    fn typed_members_compare_under_their_own_type() {
        let x = plain_column(ScalarType::Integer(IntegerType::Integer));
        let admits = |sql: &str, outcomes: Outcomes, value: i64| {
            let guarantees = implied_with(
                ColumnContext::new().with_column("x", x.clone()),
                &parse(sql),
                outcomes,
            );
            !guarantees.is_impossible()
                && guarantees
                    .reference(&"x".to_string())
                    .is_none_or(|x| x.may_be(&PgValue::Integer(value)))
        };
        assert!(admits("x IN (9223372036854775807)", Outcomes::FALSE, 1));
        assert!(admits("x NOT IN (2147483648)", Outcomes::TRUE, 1));
        assert!(admits("x IN (1.5)", Outcomes::FALSE, 2));
        assert!(!admits("x IN (1.5)", Outcomes::TRUE, 2));
        assert!(admits("x = ANY (ARRAY[9223372036854775807])", Outcomes::FALSE, 1));
        assert!(admits("x = ANY (ARRAY[1.5])", Outcomes::FALSE, 2));
    }

    /// Members of the other modelled types keep their own type too, and an
    /// integer among `numeric`s reads as a `numeric`.
    #[test]
    fn typed_members_of_each_type_narrow_their_column() {
        let implied = |name: &str, ty: ScalarType, sql: &str, outcomes: Outcomes| {
            implied_with(
                ColumnContext::new().with_column(name, plain_column(ty)),
                &parse(sql),
                outcomes,
            )
        };
        let n = implied("n", ScalarType::Numeric, "n IN (1, 1.5)", Outcomes::TRUE);
        let n = column(&n, "n");
        assert!(n.may_be(&PgValue::Numeric(PgNumeric::from(1))));
        assert!(!n.may_be(&PgValue::Numeric(PgNumeric::from(2))));
        let b = implied("b", ScalarType::Boolean, "b IN (true)", Outcomes::TRUE);
        let b = column(&b, "b");
        assert!(b.may_be(&PgValue::Boolean(true)) && !b.may_be(&PgValue::Boolean(false)));
        let d = implied("d", ScalarType::Date, "d IN (DATE '2024-03-01')", Outcomes::TRUE);
        let d = column(&d, "d");
        let day =
            |text: &str| PgValue::Date(sql_values::postgres::parse_date(text).expect("a date"));
        assert!(d.may_be(&day("2024-03-01")) && !d.may_be(&day("2025-01-01")));
        let ts = "ts IN (TIMESTAMP '2024-03-01 10:00')";
        assert!(!implied("ts", ScalarType::Timestamp, ts, Outcomes::TRUE).is_impossible());
        assert!(!implied("ts", ScalarType::Timestamp, ts, Outcomes::FALSE).is_impossible());
    }

    #[test]
    fn a_false_equality_keeps_the_apart_values_of_a_multi_value_side() {
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Text("read".into()),
                PgValue::Text("write".into()),
                PgValue::Text("admin".into()),
            ])),
        );
        let expression = parse("x = 'admin'");
        let guarantees =
            implied_with(ColumnContext::new().with_column("x", x), &expression, Outcomes::FALSE);
        assert!(!guarantees.is_impossible());
        let x = column(&guarantees, "x");
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> =
            [PgValue::Text("read".into()), PgValue::Text("write".into())].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_called_on_null_callee_keeps_the_argument_null() {
        let other = called_on_null(Vec::new());
        let expression = parse("f(x)");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean))
                .with_callee("f", other),
            &expression,
            Outcomes::TRUE,
        );
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn length_not_false_bounds_the_length() {
        let expression = parse("length(x) <= 10");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &expression,
            Outcomes::NOT_FALSE,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.length.low, Bound::Included(0));
        assert_eq!(unlisted.length.high, Bound::Included(10));
    }

    #[test]
    fn length_bounds_not_false_is_impossible_for_never_null() {
        let expression = parse("length(x) < 3 AND length(x) > 10");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", plain_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &expression,
            Outcomes::NOT_FALSE,
        );
        assert!(guarantees.is_impossible());

        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &expression,
            Outcomes::NOT_FALSE,
        );
        assert!(!guarantees.is_impossible());
        let x = column(&guarantees, "x");
        assert!(x.is_always_null());
    }

    #[test]
    fn disjuncts_partition_on_a() {
        let expression = parse("(a IS NULL AND b IS NULL) OR (a IS NOT NULL AND b IS NOT NULL)");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", Evaluation::null(ScalarType::Unknown))
                .with_column("b", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::NOT_FALSE,
        );
        let b = column(&guarantees, "b");
        assert!(b.is_always_null());
    }

    #[test]
    fn not_equal_excludes_the_value() {
        let expression = parse("x <> ''");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert!(unlisted.excluded.contains(&PgValue::Text(String::new())));
    }

    #[test]
    fn greater_bounds_from_below() {
        let expression = parse("x > 5");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Excluded(PgNumeric::from(5)));
    }

    #[test]
    fn greater_equal_false_bounds_from_above() {
        let expression = parse("x >= 5");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &expression,
            Outcomes::FALSE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.high, Bound::Excluded(PgNumeric::from(5)));
    }

    #[test]
    fn equality_keeps_the_value() {
        let expression = parse("x = 7");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> = [PgValue::Numeric(PgNumeric::from(7))].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn an_equality_to_a_fraction_is_impossible_for_an_integer() {
        let expression = parse("x = 3.5");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Integer(IntegerType::Integer))),
            &expression,
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn between_bounds_both_sides() {
        let expression = parse("x BETWEEN 1 AND 10");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Included(PgNumeric::from(1)));
        assert_eq!(unlisted.numbers.high, Bound::Included(PgNumeric::from(10)));
    }

    #[test]
    fn not_comparison_flips_the_outcome() {
        let expression = parse("NOT (x < 3)");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Included(PgNumeric::from(3)));
    }

    #[test]
    fn is_null_makes_the_column_always_null() {
        let expression = parse("x IS NULL");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(x.is_always_null());
    }

    #[test]
    fn is_not_null_makes_the_column_non_null() {
        let expression = parse("x IS NOT NULL");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
    }

    #[test]
    fn a_text_cast_passes_the_restriction_to_the_column() {
        let expression = parse("(v)::text = 'read'::text");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("v", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::TRUE,
        );
        let v = column(&guarantees, "v");
        assert!(!v.may_be_null());
        let Values::Listed(listed) = v.values() else { panic!("v is listed") };
        let expected: BTreeSet<PgValue> = [PgValue::Text("read".to_string())].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_cast_array_lists_its_folded_members() {
        let expression = parse("v = ANY ((ARRAY['read', 'write'])::varchar(2)[])");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("v", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::TRUE,
        );
        let v = column(&guarantees, "v");
        assert!(!v.may_be_null());
        let Values::Listed(listed) = v.values() else { panic!("v is listed") };
        let expected: BTreeSet<PgValue> =
            [PgValue::Text("re".to_string()), PgValue::Text("wr".to_string())].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_bounded_varchar_cast_keeps_the_restriction_off_the_column() {
        let expression = parse("(v)::varchar(2) = 're'");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("v", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::TRUE,
        );
        assert!(guarantees.reference(&"v".to_string()).is_none());
    }

    #[test]
    fn a_case_insensitive_column_cast_keeps_the_restriction_off_the_column() {
        let expression = parse("(v)::text = 'read'::text");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("v", unknown_column(ScalarType::Text(TextRules::UNKNOWN))),
            &expression,
            Outcomes::TRUE,
        );
        assert!(guarantees.reference(&"v".to_string()).is_none());
    }

    #[test]
    fn an_any_over_a_column_array_is_not_impossible() {
        let expression = parse("x = ANY (ARRAY[col])");
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("col", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(matches!(x.values(), Values::Unlisted(_)));
    }

    /// The guarantees `expression` takes `outcomes` under, over the column
    /// `x` and a nested query projecting `v` over several rows.
    fn quantified(
        x: Evaluation,
        v: Evaluation,
        sql: &str,
        outcomes: Outcomes,
    ) -> Guarantees<String> {
        let context = ColumnContext::new()
            .with_column("x", x)
            .with_query(Rows { none: false, one: false, many: true }, v);
        implied_with(context, &parse(sql), outcomes)
    }

    #[test]
    fn numeric_any_all_restrict_as_postgres_reads_them() {
        let x = plain_column(ScalarType::Numeric);
        let v = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(3), PgValue::Integer(5)])),
        );
        let open = || Bound::Unbounded;
        let inc = |n: i64| Bound::Included(PgNumeric::from(n));
        let exc = |n: i64| Bound::Excluded(PgNumeric::from(n));
        let expected: &[(&str, Outcomes, Bound<PgNumeric>, Bound<PgNumeric>)] = &[
            ("x < ANY (SELECT v FROM q)", Outcomes::TRUE, open(), exc(5)),
            ("x < ANY (SELECT v FROM q)", Outcomes::FALSE, inc(3), open()),
            ("x <= ANY (SELECT v FROM q)", Outcomes::TRUE, open(), inc(5)),
            ("x <= ANY (SELECT v FROM q)", Outcomes::FALSE, exc(3), open()),
            ("x > ANY (SELECT v FROM q)", Outcomes::TRUE, exc(3), open()),
            ("x > ANY (SELECT v FROM q)", Outcomes::FALSE, open(), inc(5)),
            ("x >= ANY (SELECT v FROM q)", Outcomes::TRUE, inc(3), open()),
            ("x >= ANY (SELECT v FROM q)", Outcomes::FALSE, open(), exc(5)),
            ("x < ALL (SELECT v FROM q)", Outcomes::TRUE, open(), exc(5)),
            ("x < ALL (SELECT v FROM q)", Outcomes::FALSE, inc(3), open()),
            ("x <= ALL (SELECT v FROM q)", Outcomes::TRUE, open(), inc(5)),
            ("x <= ALL (SELECT v FROM q)", Outcomes::FALSE, exc(3), open()),
            ("x > ALL (SELECT v FROM q)", Outcomes::TRUE, exc(3), open()),
            ("x > ALL (SELECT v FROM q)", Outcomes::FALSE, open(), inc(5)),
            ("x >= ALL (SELECT v FROM q)", Outcomes::TRUE, inc(3), open()),
            ("x >= ALL (SELECT v FROM q)", Outcomes::FALSE, open(), exc(5)),
            // An array holds every member on every row, so the readings
            // against every member bound `x` at the far member.
            ("x < ANY (ARRAY[3, 5])", Outcomes::FALSE, inc(5), open()),
            ("x <= ANY (ARRAY[3, 5])", Outcomes::FALSE, exc(5), open()),
            ("x > ANY (ARRAY[3, 5])", Outcomes::FALSE, open(), inc(3)),
            ("x >= ANY (ARRAY[3, 5])", Outcomes::FALSE, open(), exc(3)),
            ("x < ALL (ARRAY[3, 5])", Outcomes::TRUE, open(), exc(3)),
            ("x <= ALL (ARRAY[3, 5])", Outcomes::TRUE, open(), inc(3)),
            ("x > ALL (ARRAY[3, 5])", Outcomes::TRUE, exc(5), open()),
            ("x >= ALL (ARRAY[3, 5])", Outcomes::TRUE, inc(5), open()),
        ];
        for (sql, outcomes, low, high) in expected {
            let guarantees = quantified(x.clone(), v.clone(), sql, *outcomes);
            let x = column(&guarantees, "x");
            assert!(!x.may_be_null(), "{sql} {outcomes:?}");
            let Values::Unlisted(unlisted) = x.values() else { panic!("{sql} {outcomes:?}") };
            assert_eq!(unlisted.numbers.low, *low, "{sql} {outcomes:?}");
            assert_eq!(unlisted.numbers.high, *high, "{sql} {outcomes:?}");
        }
    }

    /// `IN` over an all-`NULL` projection reads `FALSE` on its empty rows,
    /// whatever `x` holds.
    #[test]
    fn an_in_subquery_over_an_always_null_column_reads_false_on_empty_rows() {
        let v =
            Evaluation::new(ScalarType::Integer(IntegerType::Integer), true, false, Values::none());
        let guarantees = quantified_over(
            Rows { none: true, one: false, many: true },
            unknown_column(ScalarType::Integer(IntegerType::Integer)),
            v,
            "x IN (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// `IN` over an all-`NULL` projection never reads `TRUE`, and `= ALL`
    /// over two distinct cast members never does either.
    #[test]
    fn a_member_set_no_value_can_satisfy_is_impossible_to_read_true() {
        let v =
            Evaluation::new(ScalarType::Integer(IntegerType::Integer), true, false, Values::none());
        let guarantees = quantified_over(
            Rows { none: false, one: true, many: true },
            unknown_column(ScalarType::Integer(IntegerType::Integer)),
            v,
            "x IN (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ALL (ARRAY[1, 2]::numeric[])"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    /// A member that is not a constant holds one of its values on each row,
    /// so `IN` reads its values as a possibility, not as members all at once.
    #[test]
    fn an_in_list_member_that_is_not_a_constant_reads_one_value() {
        let listed = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)])),
        );
        let context = || {
            ColumnContext::new().with_column("x", listed.clone()).with_column("y", listed.clone())
        };
        for (sql, outcomes) in [("x IN (y)", Outcomes::FALSE), ("x NOT IN (y)", Outcomes::TRUE)] {
            let guarantees = implied_with(context(), &parse(sql), outcomes);
            assert!(!guarantees.is_impossible(), "{sql}");
            assert!(column(&guarantees, "x").may_be(&PgValue::Integer(1)), "{sql}");
        }
        // A nullable member reads `NULL` for any non-`NULL` `x` on its
        // `NULL` rows.
        let context = || context().with_column("z", listed.clone().with_null(true));
        let guarantees = implied_with(context(), &parse("x IN (z)"), Outcomes::NULL);
        assert!(column(&guarantees, "x").may_be(&PgValue::Integer(1)));
        // Constant members are all present on every row.
        let guarantees = implied_with(context(), &parse("x IN (1, 3)"), Outcomes::FALSE);
        assert!(!column(&guarantees, "x").may_be(&PgValue::Integer(1)));
    }

    /// A value-less other side reads `NULL` everywhere, so no input takes an
    /// equality outcome.
    #[test]
    fn a_value_less_side_admits_no_equality_outcome() {
        let nothing = Evaluation::null(ScalarType::Integer(IntegerType::Integer));
        for is_true in [true, false] {
            let restriction = super::equality(Family::Single, CmpOp::Eq, is_true, &nothing);
            assert!(matches!(restriction, Restriction::Dead));
        }
    }

    /// An ordering against listed text leaves an unlisted side's values
    /// open.
    #[test]
    fn an_ordering_against_listed_text_keeps_an_unlisted_side_open() {
        let byte = |text: &str| PgValue::Text(text.to_string());
        let v = Evaluation::new(
            ScalarType::Text(TextRules::BYTEWISE),
            false,
            false,
            Values::Listed(BTreeSet::from([byte("a"), byte("e")])),
        );
        let guarantees = quantified(
            plain_column(ScalarType::Text(TextRules::BYTEWISE)),
            v,
            "x < ANY (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(matches!(column(&guarantees, "x").values(), Values::Unlisted(_)));
    }

    /// `<> ANY` reads `FALSE` on an empty row set whatever `x` holds.
    #[test]
    fn an_any_reading_false_over_possibly_empty_rows_admits_every_value() {
        let v = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)])),
        );
        let guarantees = quantified_over(
            Rows { none: true, one: false, many: true },
            plain_column(ScalarType::Integer(IntegerType::Integer)),
            v,
            "x <> ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// `< ANY` reads `FALSE` on an empty row set whatever `x` holds.
    #[test]
    fn a_less_any_reading_false_over_possibly_empty_rows_admits_every_value() {
        let v = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Included(PgNumeric::from(0)),
                    high: Bound::Unbounded,
                },
                length: Interval::unbounded(),
            }),
        );
        let guarantees = quantified_over(
            Rows { none: true, one: false, many: true },
            plain_column(ScalarType::Numeric),
            v,
            "x < ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// A listed date `> ALL` reading true keeps the values past the
    /// minimum, since a row set may hold the minimum alone.
    #[test]
    fn a_listed_date_all_reading_true_keeps_the_values_past_the_min() {
        let day =
            |text: &str| PgValue::Date(sql_values::postgres::parse_date(text).expect("a date"));
        let v = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([day("2024-01-01"), day("2024-06-01")])),
        );
        let x = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([day("2024-03-01"), day("2025-01-01")])),
        );
        let guarantees = quantified_over(
            Rows { none: false, one: false, many: true },
            x,
            v,
            "x > ALL (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        let Values::Listed(listed) = x.values() else { panic!("x stays listed") };
        let expected = BTreeSet::from([day("2024-03-01"), day("2025-01-01")]);
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_null_member_keeps_the_rows_postgres_reads_as_null() {
        let x = plain_column(ScalarType::Numeric);
        let admits = |sql: &str, value: i64| {
            let guarantees = implied_with(
                ColumnContext::new().with_column("x", x.clone()),
                &parse(sql),
                Outcomes::NULL,
            );
            column(&guarantees, "x").may_be(&PgValue::Numeric(PgNumeric::from(value)))
        };
        assert!(admits("x <> ALL (ARRAY[3, NULL])", 1));
        assert!(admits("x <> ANY (ARRAY[3, NULL])", 3));
        assert!(admits("x < ANY (ARRAY[0, NULL])", 5));
        assert!(admits("x > ALL (ARRAY[0, NULL])", 1));
        assert!(admits("x = ANY (ARRAY[3, NULL])", 1));
        assert!(admits("x = ALL (ARRAY[3, NULL])", 3));
    }

    /// Parentheses around the array of an `ANY` leave its members read.
    #[test]
    fn any_reads_a_parenthesised_array() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", plain_column(ScalarType::Numeric)),
            &parse("x = ANY ((ARRAY[3, 5]))"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(3))));
        assert!(!x.may_be(&PgValue::Numeric(PgNumeric::from(4))));
    }

    /// An untyped literal term reads through the `boolean` input.
    #[test]
    fn an_untyped_literal_term_reads_as_a_boolean() {
        let context = || ColumnContext::new().with_column("b", plain_column(ScalarType::Boolean));
        assert!(!implied_with(context(), &parse("'t' AND b"), Outcomes::TRUE).is_impossible());
        assert!(!implied_with(context(), &parse("'f' OR b"), Outcomes::FALSE).is_impossible());
        assert!(implied_with(context(), &parse("'f' AND b"), Outcomes::TRUE).is_impossible());
    }

    /// An untyped ordering bound reads as the column type.
    #[test]
    fn an_untyped_ordering_bound_reads_as_the_column_type() {
        let x = plain_column(ScalarType::Numeric);
        let admits = |sql: &str, outcomes: Outcomes, value: &str| {
            let guarantees = implied_with(
                ColumnContext::new().with_column("x", x.clone()),
                &parse(sql),
                outcomes,
            );
            let number = PgNumeric::parse(value).expect("a number");
            !guarantees.is_impossible()
                && column(&guarantees, "x").may_be(&PgValue::Numeric(number))
        };
        assert!(admits("x BETWEEN 1 AND '2'", Outcomes::TRUE, "1.5"));
        assert!(admits("x BETWEEN 1 AND '2'", Outcomes::TRUE, "2"));
        assert!(admits("x BETWEEN 1 AND '2'", Outcomes::FALSE, "2.25"));
        assert!(!admits("x <= '2'", Outcomes::TRUE, "2.25"));
    }

    #[test]
    fn an_untyped_literal_reads_as_the_column_type() {
        let x = plain_column(ScalarType::Numeric);
        // `x = '1'` reads the literal as `numeric`, so only one survives.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x = '1'"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        let evaluation = column(&guarantees, "x");
        assert!(evaluation.may_be(&PgValue::Numeric(PgNumeric::from(1))));
        assert!(!evaluation.may_be(&PgValue::Numeric(PgNumeric::from(2))));
        // `x IN (1, '2')` reads both members as `numeric`.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x IN (1, '2')"),
            Outcomes::TRUE,
        );
        let evaluation = column(&guarantees, "x");
        assert!(evaluation.may_be(&PgValue::Numeric(PgNumeric::from(1))));
        assert!(evaluation.may_be(&PgValue::Numeric(PgNumeric::from(2))));
        assert!(!evaluation.may_be(&PgValue::Numeric(PgNumeric::from(3))));
        // `b = 'true'` reads the literal as `boolean`, so `b` stays true.
        let b = plain_column(ScalarType::Boolean);
        let guarantees = implied_with(
            ColumnContext::new().with_column("b", b.clone()),
            &parse("b = 'true'"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        let evaluation = column(&guarantees, "b");
        assert!(evaluation.may_be(&PgValue::Boolean(true)));
        assert!(!evaluation.may_be(&PgValue::Boolean(false)));
        // `x = 'abc'` raises on every row, so it is never `TRUE`.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x = 'abc'"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
        // An unlisted untyped side reads no membership.
        let any = Evaluation::new(
            ScalarType::UnknownLiteral,
            false,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        assert!(super::reinterpreted_values(&any, ScalarType::Numeric).is_none());
        // A literal the type's input refuses drops out of the membership.
        let literals = Evaluation::new(
            ScalarType::UnknownLiteral,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Text("1".to_string()),
                PgValue::Text("abc".to_string()),
            ])),
        );
        let read = super::reinterpreted_values(&literals, ScalarType::Numeric);
        assert_eq!(read, Some(BTreeSet::from([PgValue::Numeric(PgNumeric::from(1))])));
    }

    #[test]
    fn not_equal_quantifiers_restrict_as_postgres_reads_them() {
        let x = plain_column(ScalarType::Numeric);
        let admits = |guarantees: &Guarantees<String>, value: i64| {
            column(guarantees, "x").may_be(&PgValue::Numeric(PgNumeric::from(value)))
        };
        // `x <> ANY (ARRAY[1, 2])` true holds for `x = 1`, so no value drops.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x <> ANY (ARRAY[1, 2])"),
            Outcomes::TRUE,
        );
        assert!(admits(&guarantees, 1));
        assert!(admits(&guarantees, 3));
        // `x <> ANY (ARRAY[1, 2])` false needs `x` to equal every member.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x <> ANY (ARRAY[1, 2])"),
            Outcomes::FALSE,
        );
        assert!(guarantees.is_impossible());
        // `x <> ALL (ARRAY[1, 2])` true keeps `x` apart from both members.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x <> ALL (ARRAY[1, 2])"),
            Outcomes::TRUE,
        );
        assert!(!admits(&guarantees, 1));
        assert!(!admits(&guarantees, 2));
        assert!(admits(&guarantees, 3));
        // `x <> ALL (ARRAY[1, 2])` false is one of the members.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x <> ALL (ARRAY[1, 2])"),
            Outcomes::FALSE,
        );
        assert!(admits(&guarantees, 1));
        assert!(admits(&guarantees, 2));
        assert!(!admits(&guarantees, 3));
        // A single member still decides.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x.clone()),
            &parse("x <> ANY (ARRAY[1])"),
            Outcomes::TRUE,
        );
        assert!(!admits(&guarantees, 1));
        assert!(admits(&guarantees, 2));
    }

    #[test]
    fn an_all_null_over_a_null_member_keeps_the_equal_value() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ALL (ARRAY[1, NULL])"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(1))));
        assert!(!x.may_be(&PgValue::Numeric(PgNumeric::from(2))));
    }

    #[test]
    fn text_any_all_restrict_as_postgres_reads_them() {
        // `x` holds two byte-ordered values, and the query projects `a` and
        // `e` around them.
        let byte = |text: &str| PgValue::Text(text.to_string());
        let x = Evaluation::new(
            ScalarType::Text(TextRules::BYTEWISE),
            false,
            false,
            Values::Listed(BTreeSet::from([byte("`"), byte("b")])),
        );
        let v = Evaluation::new(
            ScalarType::Text(TextRules::BYTEWISE),
            false,
            false,
            Values::Listed(BTreeSet::from([byte("a"), byte("e")])),
        );
        let keeps: BTreeSet<PgValue> = [byte("`"), byte("b")].into();
        let only_b: BTreeSet<PgValue> = [byte("b")].into();
        let expected: &[(&str, Outcomes, &BTreeSet<PgValue>)] = &[
            ("x < ANY (SELECT v FROM q)", Outcomes::TRUE, &keeps),
            ("x < ANY (SELECT v FROM q)", Outcomes::FALSE, &only_b),
            ("x <= ANY (SELECT v FROM q)", Outcomes::TRUE, &keeps),
            ("x <= ANY (SELECT v FROM q)", Outcomes::FALSE, &only_b),
            ("x > ANY (SELECT v FROM q)", Outcomes::TRUE, &only_b),
            ("x > ANY (SELECT v FROM q)", Outcomes::FALSE, &keeps),
            ("x >= ANY (SELECT v FROM q)", Outcomes::TRUE, &only_b),
            ("x >= ANY (SELECT v FROM q)", Outcomes::FALSE, &keeps),
            ("x < ALL (SELECT v FROM q)", Outcomes::TRUE, &keeps),
            ("x < ALL (SELECT v FROM q)", Outcomes::FALSE, &only_b),
            ("x <= ALL (SELECT v FROM q)", Outcomes::TRUE, &keeps),
            ("x <= ALL (SELECT v FROM q)", Outcomes::FALSE, &only_b),
            ("x > ALL (SELECT v FROM q)", Outcomes::TRUE, &only_b),
            ("x > ALL (SELECT v FROM q)", Outcomes::FALSE, &keeps),
            ("x >= ALL (SELECT v FROM q)", Outcomes::TRUE, &only_b),
            ("x >= ALL (SELECT v FROM q)", Outcomes::FALSE, &keeps),
        ];
        for (sql, outcomes, listed) in expected {
            let guarantees = quantified(x.clone(), v.clone(), sql, *outcomes);
            assert!(!guarantees.is_impossible(), "{sql} {outcomes:?}");
            let x = column(&guarantees, "x");
            let Values::Listed(values) = x.values() else { panic!("{sql} {outcomes:?}") };
            assert_eq!(values, *listed, "{sql} {outcomes:?}");
        }
    }

    /// A dialect that resolves a `NULL` result of a call to a `NULL`
    /// argument.
    struct NullOnNullInput;

    impl EvalDialect for NullOnNullInput {
        type Options = ();

        fn null_only_on_null_input(
            &self,
            _callees: &[crate::Callee],
            _options: &Self::Options,
        ) -> bool {
            true
        }
    }

    /// A dialect that counts the characters of any call's first argument.
    struct CountsAnyArgument;

    impl EvalDialect for CountsAnyArgument {
        type Options = ();

        fn counted_length_argument(
            &self,
            _call: &Function,
            _callees: &[crate::Callee],
            _options: &Self::Options,
        ) -> Option<usize> {
            Some(0)
        }
    }

    /// The guarantees `expression` takes `outcomes` under, with the generic
    /// dialect.
    fn implied_generic(
        context: ColumnContext,
        expression: &Expr,
        outcomes: Outcomes,
    ) -> Guarantees<String> {
        let mut context = context;
        implied(
            expression,
            outcomes,
            &sqlparser::dialect::GenericDialect {},
            &(),
            &mut context,
            &Guarantees::new(),
        )
        .expect("the context resolves")
    }

    /// The guarantees `expression` takes `outcomes` under, with a dialect
    /// that resolves a `NULL` call result to a `NULL` argument.
    fn implied_null_on_null(
        context: ColumnContext,
        expression: &Expr,
        outcomes: Outcomes,
    ) -> Guarantees<String> {
        let mut context = context;
        implied(expression, outcomes, &NullOnNullInput {}, &(), &mut context, &Guarantees::new())
            .expect("the context resolves")
    }

    /// The guarantees `expression` takes `outcomes` under, with a dialect
    /// that counts the first argument's characters in every call.
    fn implied_counting(
        context: ColumnContext,
        expression: &Expr,
        outcomes: Outcomes,
    ) -> Guarantees<String> {
        let mut context = context;
        implied(expression, outcomes, &CountsAnyArgument {}, &(), &mut context, &Guarantees::new())
            .expect("the context resolves")
    }

    /// The guarantees `expression` takes `outcomes` under with
    /// `transform_null_equals` on.
    fn implied_null_equals(
        context: ColumnContext,
        expression: &Expr,
        outcomes: Outcomes,
    ) -> Guarantees<String> {
        let mut context = context;
        implied(
            expression,
            outcomes,
            &PostgreSqlDialect {},
            &PostgreSqlOptions { transform_null_equals: true },
            &mut context,
            &Guarantees::new(),
        )
        .expect("the context resolves")
    }

    /// The guarantees `sql` takes `outcomes` under over `x` and a query
    /// projecting `v` over `rows`.
    fn quantified_over(
        rows: Rows,
        x: Evaluation,
        v: Evaluation,
        sql: &str,
        outcomes: Outcomes,
    ) -> Guarantees<String> {
        let context = ColumnContext::new().with_column("x", x).with_query(rows, v);
        implied_with(context, &parse(sql), outcomes)
    }

    /// A subquery's rows may hold any of the values its column may hold, so
    /// a condition read against every row reads against one row backward.
    #[test]
    fn a_subquery_condition_reads_any_row_backward() {
        let int4 = ScalarType::Integer(sql_values::postgres::IntegerType::Integer);
        let listed = Evaluation::new(
            int4,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)])),
        );
        let rows = Rows { none: false, one: true, many: true };
        for (sql, outcomes) in [
            ("x = ANY (SELECT y FROM t)", Outcomes::FALSE),
            ("x IN (SELECT y FROM t)", Outcomes::FALSE),
            ("x <> ALL (SELECT y FROM t)", Outcomes::TRUE),
            ("x < ALL (SELECT y FROM t)", Outcomes::TRUE),
        ] {
            let guarantees = quantified_over(rows, listed.clone(), listed.clone(), sql, outcomes);
            assert!(!guarantees.is_impossible(), "{sql}");
            let x = guarantees.reference(&"x".to_string());
            assert!(x.is_none_or(|x| x.may_be(&PgValue::Integer(1))), "{sql}");
        }
        // An empty row set reads `FALSE` to `ANY` whatever `x` holds.
        let one = Evaluation::known(PgValue::Integer(1), int4);
        let rows = Rows { none: true, one: true, many: false };
        let guarantees = quantified_over(
            rows,
            listed.clone(),
            one,
            "x = ANY (SELECT y FROM t)",
            Outcomes::FALSE,
        );
        let x = guarantees.reference(&"x".to_string());
        assert!(x.is_none_or(|x| x.may_be(&PgValue::Integer(1))));
    }

    #[test]
    fn no_outcome_marks_the_guarantees_impossible() {
        let expression = parse("x IS NULL");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::NONE,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_multi_outcome_admitting_no_single_is_impossible() {
        // `x` is certainly `true`, so `NOT TRUE` admits nothing.
        let x = Evaluation::new(
            ScalarType::Boolean,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Boolean(true)])),
        );
        let guarantees =
            implied_with(ColumnContext::new().with_column("x", x), &parse("x"), Outcomes::NOT_TRUE);
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_non_boolean_subject_narrows_only_its_null() {
        let expression = parse("x");
        // A `true` or `false` outcome narrows only the `NULL` possibility.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(!nullable(&guarantees, "x"));
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
        assert!(!nullable(&guarantees, "x"));
        // `x IS TRUE` keeps every value a non-`boolean` `x` may hold.
        let x = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Numeric(PgNumeric::from(1)),
                PgValue::Numeric(PgNumeric::from(2)),
            ])),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x),
            &parse("x IS TRUE"),
            Outcomes::TRUE,
        );
        assert!(column(&guarantees, "x").may_be(&PgValue::Numeric(PgNumeric::from(2))));
        // A `NULL` outcome narrows a nullable subject to `NULL` alone.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
        // A never-`NULL` subject admits no outcome at all.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", plain_column(ScalarType::Text(TextRules::DEFAULT))),
            &expression,
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_nested_parenthesis_reads_the_inner_expression() {
        // Outer parentheses read the inner expression.
        let expression = parse("((x IS NULL))");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::TRUE,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    #[test]
    fn an_unrelated_unary_op_keeps_the_outcome_on_the_expression() {
        // PostgreSQL raises on `+x` for a boolean `x`; the forward pass
        // keeps the type with unlisted values, and the subject stays on the
        // expression it names.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Boolean)),
            &parse("+x"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", plain_column(ScalarType::Boolean)),
            &parse("+x"),
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    /// PostgreSQL has no `+` for booleans, so the pair raises and the
    /// outcome is ruled out.
    #[test]
    fn an_operator_less_binary_op_rules_out_the_outcome() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Boolean))
                .with_column("y", unknown_column(ScalarType::Boolean)),
            &parse("x + y"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_true_conjunction_makes_both_sides_true() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", unknown_column(ScalarType::Boolean))
                .with_column("b", unknown_column(ScalarType::Boolean)),
            &parse("a AND b"),
            Outcomes::TRUE,
        );
        for name in ["a", "b"] {
            let side = column(&guarantees, name);
            assert!(!side.may_be_null(), "{name} is non-NULL");
            assert!(side.may_be_true(), "{name} may be true");
            assert!(!side.may_be_false(), "{name} is not false");
        }
    }

    #[test]
    fn a_false_conjunction_keeps_no_single_side() {
        // Either side may be the false one, and `FALSE` settles the chain
        // on its own, so each side is left open.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", unknown_column(ScalarType::Boolean))
                .with_column("b", unknown_column(ScalarType::Boolean)),
            &parse("a AND b"),
            Outcomes::FALSE,
        );
        assert!(guarantees.reference(&"a".to_string()).is_none());
        assert!(guarantees.reference(&"b".to_string()).is_none());
    }

    #[test]
    fn a_null_conjunction_keeps_a_null_side_and_a_true_side() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", unknown_column(ScalarType::Boolean))
                .with_column("b", unknown_column(ScalarType::Boolean)),
            &parse("a AND b"),
            Outcomes::NULL,
        );
        // The `NULL` outcome needs one side `NULL` and the other not false.
        let a = column(&guarantees, "a");
        assert!(a.may_be_null());
        assert!(a.may_be_true());
        assert!(!a.may_be_false());
        let b = column(&guarantees, "b");
        assert!(b.may_be_null());
        assert!(b.may_be_true());
        assert!(!b.may_be_false());
    }

    #[test]
    fn a_true_disjunction_keeps_no_single_side() {
        // Either side may be the true one, so neither is constrained on its
        // own.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", unknown_column(ScalarType::Boolean))
                .with_column("b", unknown_column(ScalarType::Boolean)),
            &parse("a OR b"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"a".to_string()).is_none());
        assert!(guarantees.reference(&"b".to_string()).is_none());
    }

    #[test]
    fn a_false_disjunction_makes_both_sides_false() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", unknown_column(ScalarType::Boolean))
                .with_column("b", unknown_column(ScalarType::Boolean)),
            &parse("a OR b"),
            Outcomes::FALSE,
        );
        for name in ["a", "b"] {
            let side = column(&guarantees, name);
            assert!(!side.may_be_null(), "{name} is non-NULL");
            assert!(side.may_be_false(), "{name} is false");
            assert!(!side.may_be_true(), "{name} is not true");
        }
    }

    #[test]
    fn a_null_disjunction_keeps_false_or_null_sides() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", unknown_column(ScalarType::Boolean))
                .with_column("b", unknown_column(ScalarType::Boolean)),
            &parse("a OR b"),
            Outcomes::NULL,
        );
        // `NULL OR b` is `NULL` only when `b` is not true, and the other
        // branch around.
        for name in ["a", "b"] {
            let side = column(&guarantees, name);
            assert!(side.may_be_null(), "{name} may be NULL");
            assert!(side.may_be_false(), "{name} may be false");
            assert!(!side.may_be_true(), "{name} is not true");
        }
    }

    #[test]
    fn a_true_chain_past_the_depth_limit_narrows_only_what_it_reaches() {
        // A right-nested chain walks one depth per term, so a spine past the
        // depth limit stops narrowing and leaves the rest open
        let identifier = |name: &str| Expr::Identifier(sqlparser::ast::Ident::new(name));
        let mut expression = identifier("a32");
        for i in (0..32).rev() {
            expression = Expr::BinaryOp {
                left: Box::new(identifier(&format!("a{i}"))),
                op: BinaryOperator::And,
                right: Box::new(expression),
            };
        }
        let mut context = ColumnContext::new();
        for i in 0..33 {
            context = context.with_column(&format!("a{i}"), unknown_column(ScalarType::Boolean));
        }
        let guarantees = implied_with(context, &expression, Outcomes::TRUE);
        assert!(!guarantees.is_impossible());
        // The terms reached stay restricted by the true outcome
        for name in ["a29", "a30"] {
            let term = column(&guarantees, name);
            assert!(term.may_be_true(), "{name} may be true");
            assert!(!term.may_be_false(), "{name} is not false");
            assert!(!term.may_be_null(), "{name} is not NULL");
        }
        // The terms past the limit are left open
        assert!(guarantees.reference(&"a31".to_string()).is_none());
        assert!(guarantees.reference(&"a32".to_string()).is_none());
    }

    #[test]
    fn a_boolean_overlap_keeps_the_outcome_on_the_pair() {
        let context = ColumnContext::new()
            .with_column("a", plain_column(ScalarType::Unknown))
            .with_column("b", plain_column(ScalarType::Unknown));
        let guarantees = implied_with(context, &parse("a && b"), Outcomes::TRUE);
        // PostgreSQL has no input narrowing for `&&`, so only the pair itself
        // carries the outcome
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"a".to_string()).is_none());
        assert!(guarantees.reference(&"b".to_string()).is_none());
    }

    /// A condition the backward pass has no narrowing for leaves its inputs
    /// open.
    #[test]
    fn an_unnarrowed_condition_leaves_its_inputs_open() {
        let context = ColumnContext::new()
            .with_column("a", plain_column(ScalarType::Unknown))
            .with_column("b", plain_column(ScalarType::Unknown));
        let guarantees = implied_with(context, &parse("a + b"), Outcomes::TRUE);
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"a".to_string()).is_none());
    }

    #[test]
    fn a_null_conjunction_past_an_admitted_side_is_impossible() {
        let b = Evaluation::new(
            ScalarType::Boolean,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Boolean(false)])),
        );
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", unknown_column(ScalarType::Boolean))
                .with_column("b", b),
            &parse("a AND b"),
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_null_conjunction_of_two_false_terms_and_a_nullable_one_is_impossible() {
        // Both false terms cannot be the null one, and the nullable term
        // being null needs the false terms clear of `false`, so no branch
        // stands
        let false_column = || {
            Evaluation::new(
                ScalarType::Boolean,
                false,
                false,
                Values::Listed(BTreeSet::from([PgValue::Boolean(false)])),
            )
        };
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("a", false_column())
                .with_column("b", false_column())
                .with_column("c", unknown_column(ScalarType::Boolean)),
            &parse("a AND b AND c"),
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_null_conjunction_keeps_its_fixed_call_and_cast_terms() {
        // The fixed call `f(1)` reads on its side only, and the always-null
        // cast reads in its branch only
        let context = ColumnContext::new()
            .with_column("s", unknown_column(ScalarType::Boolean))
            .with_callee("f", strict_callee(ScalarType::Boolean));
        let guarantees = implied_with(context, &parse("NULL::text AND f(1) AND s"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        assert!(nullable(&guarantees, "s"));
    }

    #[test]
    fn a_null_any_over_a_null_array_keeps_the_value_open() {
        let context = ColumnContext::new()
            .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)));
        let guarantees = implied_with(context, &parse("x = ANY (NULL::text[])"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        assert!(nullable(&guarantees, "x"));
    }

    #[test]
    fn a_null_coalesced_conjunction_points_at_its_arguments() {
        let context = ColumnContext::new()
            .with_column("s", unknown_column(ScalarType::Boolean))
            .with_column("t", unknown_column(ScalarType::Boolean));
        let guarantees =
            implied_with(context, &parse("coalesce(s, t) AND coalesce(s, t)"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        let s = column(&guarantees, "s");
        assert!(s.is_always_null());
        let t = column(&guarantees, "t");
        assert!(t.is_always_null());
    }

    #[test]
    fn a_null_conjunction_with_a_call_keeps_the_side_meets() {
        let context = ColumnContext::new()
            .with_column("s", unknown_column(ScalarType::Boolean))
            .with_column("t", unknown_column(ScalarType::Boolean))
            .with_callee("f", strict_callee(ScalarType::Boolean));
        let guarantees =
            implied_with(context, &parse("coalesce(s, t) AND s AND f(s)"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        let s = column(&guarantees, "s");
        assert!(s.may_be_true());
        assert!(s.may_be_null());
        assert!(!s.may_be_false());
    }

    #[test]
    fn a_null_conjunction_reads_a_column_on_one_side_and_in_a_branch() {
        // `s > 0` reads `s` in its own branch and on its side, and the strict
        // call reads `s` on its side alone
        let s = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        let context = ColumnContext::new()
            .with_column("s", s)
            .with_callee("f", strict_callee(ScalarType::Boolean));
        let guarantees = implied_with(context, &parse("s > 0 AND f(s)"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        let s = column(&guarantees, "s");
        // The branch read keeps `s` nullable, the side read keeps it valued
        assert!(s.may_be_null());
        assert!(!s.is_always_null());
    }

    #[test]
    fn a_null_conjunction_reads_a_column_in_a_branch_and_on_one_side() {
        // The strict call reads `s` on its side alone, and `s > 0` reads `s`
        // in its own branch and on its side
        let s = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        let context = ColumnContext::new()
            .with_column("s", s)
            .with_callee("f", strict_callee(ScalarType::Boolean));
        let guarantees = implied_with(context, &parse("f(s) AND s > 0"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        let s = column(&guarantees, "s");
        // The branch read keeps `s` nullable, the side read keeps it valued
        assert!(s.may_be_null());
        assert!(!s.is_always_null());
    }

    #[test]
    fn a_null_conjunction_meets_a_column_read_on_two_sides() {
        // Both comparisons read `s` on their sides, and the third term leaves
        // a side open, so the open sides settle `s` at their meet
        let s = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        let context = ColumnContext::new()
            .with_column("s", s)
            .with_column("c", unknown_column(ScalarType::Boolean));
        let guarantees = implied_with(context, &parse("s > 0 AND s < 10 AND c"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        let s = column(&guarantees, "s");
        // The sides keep `s` within the meet of `(0, infinity)` and
        // `(-infinity, 10)`, and the branch reads keep it nullable
        assert!(s.may_be_null());
        assert!(!s.is_always_null());
        let c = column(&guarantees, "c");
        assert!(c.may_be_null());
        assert!(c.may_be_true());
    }

    #[test]
    fn is_null_tests_take_no_null_outcome() {
        let context = || ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown));
        let guarantees = implied_with(context(), &parse("x IS NULL"), Outcomes::NULL);
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(context(), &parse("x IS NOT NULL"), Outcomes::NULL);
        assert!(guarantees.is_impossible());
        // `IS UNKNOWN` reads as `IS NULL`.
        let guarantees = implied_with(context(), &parse("x IS UNKNOWN"), Outcomes::TRUE);
        assert!(column(&guarantees, "x").is_always_null());
        let guarantees = implied_with(context(), &parse("x IS NOT UNKNOWN"), Outcomes::TRUE);
        assert!(!column(&guarantees, "x").may_be_null());
    }

    #[test]
    fn a_null_equality_reads_as_an_is_null_under_transform_null_equals() {
        // `x = NULL` is `NULL` everywhere in default PostgreSQL, so `TRUE`
        // admits nothing.
        let expression = parse("x = NULL");
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
        // Under `transform_null_equals` the form reads as `x IS NULL`.
        let guarantees = implied_null_equals(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::TRUE,
        );
        assert!(column(&guarantees, "x").is_always_null());
        let guarantees = implied_null_equals(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Unknown)),
            &expression,
            Outcomes::FALSE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
    }

    #[test]
    fn is_true_tests_narrow_the_boolean() {
        let context = || ColumnContext::new().with_column("x", unknown_column(ScalarType::Boolean));
        let guarantees = implied_with(context(), &parse("x IS TRUE"), Outcomes::TRUE);
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be_true());
        assert!(!x.may_be_false());
        // `IS TRUE` false admits `false` and `NULL`.
        let guarantees = implied_with(context(), &parse("x IS TRUE"), Outcomes::FALSE);
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(x.may_be_false());
        assert!(!x.may_be_true());
        let guarantees = implied_with(context(), &parse("x IS NOT TRUE"), Outcomes::TRUE);
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(x.may_be_false());
        assert!(!x.may_be_true());
        let guarantees = implied_with(context(), &parse("x IS FALSE"), Outcomes::TRUE);
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be_false());
        assert!(!x.may_be_true());
        // `IS NOT FALSE` true admits `true` and `NULL`.
        let guarantees = implied_with(context(), &parse("x IS NOT FALSE"), Outcomes::TRUE);
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(x.may_be_true());
        assert!(!x.may_be_false());
        // The `IS` tests take no `NULL` outcome.
        let guarantees = implied_with(context(), &parse("x IS TRUE"), Outcomes::NULL);
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_distinct_from_true_keeps_a_null_or_a_different_side() {
        let value = |text: &str| PgValue::Text(text.to_string());
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let y = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a"), value("b")])),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x).with_column("y", y),
            &parse("x IS DISTINCT FROM y"),
            Outcomes::TRUE,
        );
        // Neither side is `NULL` to begin with, so the distinct side reads
        // as the apart values; the pairs PostgreSQL admits are those where
        // the sides differ.
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&value("a")));
        assert!(!x.may_be(&value("b")));
        let y = column(&guarantees, "y");
        assert!(!y.may_be_null());
        assert!(!y.may_be(&value("a")));
        assert!(y.may_be(&value("b")));
    }

    #[test]
    fn a_not_distinct_from_true_keeps_a_null_or_an_equal_side() {
        let value = |text: &str| PgValue::Text(text.to_string());
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let y = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a"), value("b")])),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x).with_column("y", y),
            &parse("x IS NOT DISTINCT FROM y"),
            Outcomes::TRUE,
        );
        // The equal pair admits neither `NULL` side; the pair PostgreSQL
        // alone admits reads the side apart from its other values.
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&value("a")));
        assert!(!x.may_be(&value("b")));
        let y = column(&guarantees, "y");
        assert!(!y.may_be_null());
        assert!(y.may_be(&value("a")));
        assert!(!y.may_be(&value("b")));
    }

    #[test]
    fn a_not_distinct_from_false_keeps_no_single_side() {
        // The false outcome needs a `NULL` on one side only, or distinct
        // values, so neither side is constrained on its own.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x IS NOT DISTINCT FROM y"),
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        assert!(guarantees.reference(&"y".to_string()).is_none());
    }

    #[test]
    fn a_null_equality_with_a_never_null_side_points_at_the_other() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", plain_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x = y"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "y").is_always_null());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_null_comparison_with_two_never_null_sides_is_impossible() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", plain_column(ScalarType::Numeric))
                .with_column("y", plain_column(ScalarType::Numeric)),
            &parse("x < y"),
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_null_equality_with_a_never_null_constant_side_points_at_the_other() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("'a' = x"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    #[test]
    fn a_two_column_equality_restricts_both_sides() {
        let value = |text: &str| PgValue::Text(text.to_string());
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let y = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a"), value("b")])),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x).with_column("y", y.clone()),
            &parse("x = y"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&value("a")));
        let other = column(&guarantees, "y");
        assert!(!other.may_be_null());
        assert!(other.may_be(&value("a")));
        assert!(!other.may_be(&value("b")));

        // A false equality over a multi-value side rules nothing out, and a
        // false `<>` keeps the other side's values.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", y.clone()),
            &parse("x = y"),
            Outcomes::FALSE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", y),
            &parse("x <> y"),
            Outcomes::FALSE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&value("a")));
        assert!(x.may_be(&value("b")));
    }

    #[test]
    fn an_incompatible_comparison_keeps_no_value_under_the_generic_dialect() {
        // PostgreSQL errors on `text = date`; the generic dialect reads the
        // comparison as an unknown boolean, which still rules out `NULL`.
        let guarantees = implied_generic(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Date)),
            &parse("x = y"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(!column(&guarantees, "x").may_be_null());
        assert!(!column(&guarantees, "y").may_be_null());
        let x = column(&guarantees, "x");
        assert!(matches!(x.values(), Values::Unlisted(_)));
    }

    #[test]
    fn a_non_identity_collation_keeps_no_membership() {
        // Equality under a non-identity collation may equate distinct
        // strings, so no membership guarantee stands.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::UNKNOWN))),
            &parse("x = 'a'"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(matches!(x.values(), Values::Unlisted(_)));
    }

    #[test]
    fn an_unorderable_pair_keeps_no_bound() {
        // A byte-ordered side against a plain collation reads no order.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::BYTEWISE))),
            &parse("x < y"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(!column(&guarantees, "x").may_be_null());
        assert!(!column(&guarantees, "y").may_be_null());
        let x = column(&guarantees, "x");
        assert!(matches!(x.values(), Values::Unlisted(_)));
    }

    #[test]
    fn a_single_value_equality_outcome_bounds_the_other_side() {
        let context = || ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric));
        let guarantees = implied_with(context(), &parse("x <= 5"), Outcomes::TRUE);
        let x = column(&guarantees, "x");
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.high, Bound::Included(PgNumeric::from(5)));

        let guarantees = implied_with(context(), &parse("x < 5"), Outcomes::FALSE);
        let x = column(&guarantees, "x");
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Included(PgNumeric::from(5)));

        // `x > 5` false keeps the five and the below.
        let guarantees = implied_with(context(), &parse("x > 5"), Outcomes::FALSE);
        let x = column(&guarantees, "x");
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.high, Bound::Included(PgNumeric::from(5)));

        let guarantees = implied_with(context(), &parse("x >= 5"), Outcomes::TRUE);
        let x = column(&guarantees, "x");
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.numbers.low, Bound::Included(PgNumeric::from(5)));
    }

    #[test]
    fn a_single_value_equality_negation_keeps_or_excludes_it() {
        let context = || ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric));
        // `x = 7` false keeps every value apart from seven.
        let guarantees = implied_with(context(), &parse("x = 7"), Outcomes::FALSE);
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(!x.may_be(&PgValue::Numeric(PgNumeric::from(7))));
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(8))));
        // `x <> 7` false keeps the seven alone.
        let guarantees = implied_with(context(), &parse("x <> 7"), Outcomes::FALSE);
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> = [PgValue::Numeric(PgNumeric::from(7))].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_numeric_point_reads_as_a_listed_value() {
        // `y` holds one numeric value the guarantees do not list.
        let y = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::point(PgNumeric::from(7)),
                length: Interval::unbounded(),
            }),
        );
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Numeric))
                .with_column("y", y),
            &parse("x = y"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> = [PgValue::Numeric(PgNumeric::from(7))].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_listed_date_comparison_restricts_by_value() {
        let day = |text: &str| {
            PgValue::Date(sql_values::postgres::parse_date(text).expect("the date parses"))
        };
        let x = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([day("2026-01-01"), day("2026-01-15")])),
        );
        let y = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([day("2026-01-10")])),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x).with_column("y", y),
            &parse("x > y"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&day("2026-01-15")));
        assert!(!x.may_be(&day("2026-01-01")));
        let y = column(&guarantees, "y");
        assert!(!y.may_be_null());
        assert!(y.may_be(&day("2026-01-10")));
    }

    #[test]
    fn an_in_list_with_a_column_member_keeps_the_union() {
        let value = |text: &str| PgValue::Text(text.to_string());
        let y = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a"), value("b")])),
        );
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", y.clone()),
            &parse("x IN (y, 'c')"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> = [value("a"), value("b"), value("c")].into();
        assert_eq!(listed, &expected);

        // `NOT IN` true keeps every value apart from the constant members,
        // and `a` too, which `y` may not hold on its row.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", y.clone()),
            &parse("x NOT IN (y, 'c')"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&value("a")));
        assert!(!x.may_be(&value("c")));
        assert!(x.may_be(&value("d")));

        // With no `NULL` member, `x IN (y, 'c')` is `NULL` only for a
        // `NULL` `x`; every other `x` compares to a value.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", y),
            &parse("x IN (y, 'c')"),
            Outcomes::NULL,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(!x.may_be(&value("d")));
        assert!(!x.may_be(&value("a")));
    }

    #[test]
    fn an_in_list_with_a_null_member_reads_as_postgres_reads_it() {
        let numeric = |value: i64| PgValue::Numeric(PgNumeric::from(value));
        let context = || ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric));
        // `x IN (1, NULL)` true holds only for `x = 1`, never false, and
        // `NULL` for every other `x`.
        let guarantees = implied_with(context(), &parse("x IN (1, NULL)"), Outcomes::TRUE);
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&numeric(1)));
        assert!(!x.may_be(&numeric(2)));
        let guarantees = implied_with(context(), &parse("x IN (1, NULL)"), Outcomes::FALSE);
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(context(), &parse("x IN (1, NULL)"), Outcomes::NULL);
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(!x.may_be(&numeric(1)));
        assert!(x.may_be(&numeric(2)));
        // `NOT IN` with a `NULL` member is never true.
        let guarantees = implied_with(context(), &parse("x NOT IN (1, NULL)"), Outcomes::TRUE);
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn an_in_list_with_an_unlisted_member_keeps_no_values() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x IN (y)"),
            Outcomes::TRUE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
        let x = column(&guarantees, "x");
        assert!(matches!(x.values(), Values::Unlisted(_)));
    }

    #[test]
    fn an_in_unnest_keeps_no_value_restriction() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x IN UNNEST(ARRAY[1, 2])"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn an_in_of_a_never_null_value_is_impossible_under_the_generic_dialect() {
        // PostgreSQL prunes the form to `NULL` first; the generic dialect
        // keeps the boolean unknown, and the `NULL` value admits no member.
        let x = Evaluation::new(ScalarType::Numeric, true, false, Values::none());
        let guarantees = implied_generic(
            ColumnContext::new().with_column("x", x),
            &parse("x IN (1)"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn an_in_subquery_over_no_rows_admits_only_the_false_outcome() {
        let no_rows = Rows { none: true, one: false, many: false };
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x IN (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
        // With no row the predicate is `FALSE` everywhere, constraining
        // nothing.
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x IN (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x IN (SELECT v FROM q)",
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn an_in_subquery_keeps_the_projected_values() {
        let one = Rows { none: false, one: true, many: false };
        let value = |text: &str| PgValue::Text(text.to_string());
        let v = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a"), value("b")])),
        );
        let guarantees = quantified_over(
            one,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            v,
            "x IN (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&value("a")));
        assert!(!x.may_be(&value("c")));
    }

    #[test]
    fn an_in_subquery_null_outcome_admits_any_side() {
        let many = Rows { none: false, one: false, many: true };
        let value = |text: &str| PgValue::Text(text.to_string());
        let v = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            v,
            "x IN (SELECT v FROM q)",
            Outcomes::NULL,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(x.may_be(&value("b")));
        // A row set holding only the `NULL` row reads `NULL` for `a` too.
        assert!(x.may_be(&value("a")));
    }

    #[test]
    fn an_in_subquery_with_no_admitted_side_is_impossible() {
        let many = Rows { none: false, one: false, many: true };
        let value = |text: &str| PgValue::Text(text.to_string());
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let v = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let guarantees = quantified_over(many, x, v, "x IN (SELECT v FROM q)", Outcomes::NULL);
        assert!(guarantees.is_impossible());
    }

    /// The `NULL` outcome of an `IN` subquery whose projection is certain of
    /// being non-`NULL` keeps only the null side.
    #[test]
    fn an_in_subquery_null_outcome_with_a_certainly_non_null_column_keeps_a_null_side() {
        let many = Rows { none: false, one: false, many: true };
        let value = |text: &str| PgValue::Text(text.to_string());
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let v = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a")])),
        );
        let guarantees = quantified_over(many, x, v, "x IN (SELECT v FROM q)", Outcomes::NULL);
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(!x.may_be(&value("a")));
    }

    /// The `NULL` outcome of a quantified comparison against a certainly
    /// non-`NULL` column keeps only the null side.
    #[test]
    fn a_quantified_null_outcome_with_a_certainly_non_null_column_keeps_a_null_side() {
        let x = Evaluation::new(ScalarType::Numeric, true, false, Values::none());
        let v = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(3)])),
        );
        let guarantees = quantified(x, v, "x < ANY (SELECT v FROM q)", Outcomes::NULL);
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
    }

    #[test]
    fn an_in_subquery_without_query_information_keeps_nothing() {
        // The context answers nothing about the query, so no restriction
        // stands.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x IN (SELECT v FROM q)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn an_empty_array_any_reads_as_false_and_an_empty_array_all_as_true() {
        let context = || ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric));
        // PostgreSQL reads `ANY` over an empty array as `FALSE` everywhere and
        // `ALL` over an empty array as `TRUE` everywhere.
        let guarantees = implied_with(context(), &parse("x = ANY (ARRAY[])"), Outcomes::TRUE);
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(context(), &parse("x = ANY (ARRAY[])"), Outcomes::FALSE);
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        let guarantees = implied_with(context(), &parse("x = ANY (ARRAY[])"), Outcomes::NULL);
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(context(), &parse("x = ALL (ARRAY[])"), Outcomes::TRUE);
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        let guarantees = implied_with(context(), &parse("x = ALL (ARRAY[])"), Outcomes::FALSE);
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(context(), &parse("x = ALL (ARRAY[])"), Outcomes::NULL);
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_null_only_array_keeps_only_the_null_outcome() {
        let context = || ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric));
        // `ANY` over `NULL` alone is `NULL` everywhere.
        let guarantees = implied_with(context(), &parse("x = ANY (ARRAY[NULL])"), Outcomes::TRUE);
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(context(), &parse("x = ANY (ARRAY[NULL])"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        // `ALL` over `NULL` alone is `NULL` where `x` is `NULL` and `NULL`
        // otherwise: never a value.
        let guarantees = implied_with(context(), &parse("x = ALL (ARRAY[NULL])"), Outcomes::TRUE);
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(context(), &parse("x = ALL (ARRAY[NULL])"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn a_cast_array_keeps_the_members_the_cast_folds() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ANY ((ARRAY[1])::int[])"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> = [PgValue::Numeric(PgNumeric::from(1))].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_valueless_cast_array_keeps_only_its_null_outcome() {
        // `(ARRAY[NULL])::text[]` holds no non-`NULL` member.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x = ANY ((ARRAY[NULL])::text[])"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x = ANY ((ARRAY[NULL])::text[])"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn an_any_over_a_plain_column_keeps_no_value_restriction() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x = ANY (y)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x = ALL (y)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_pattern_any_reads_as_unsupported() {
        // PostgreSQL: the pattern `'a'` matches only `'a'`; the crate
        // models `LIKE` comparison as unsupported, so every outcome reads
        // as impossible.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x ~~ ANY (ARRAY['a'])"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    /// `~~` is not a value comparison, so the `NULL` outcome of `~~ ANY`
    /// narrows nothing.
    #[test]
    fn a_pattern_any_null_outcome_narrows_nothing() {
        // `NULL ~~ ANY (ARRAY['a', NULL])` reads `NULL` in PostgreSQL.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x ~~ ANY (ARRAY['a', NULL])"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn an_any_over_no_rows_reads_as_false_and_an_all_as_true() {
        // With no row, `ANY` is `FALSE` everywhere and `ALL` is `TRUE`.
        let no_rows = Rows { none: true, one: false, many: false };
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x = ANY (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x = ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x = ANY (SELECT v FROM q)",
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x = ALL (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x = ALL (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(guarantees.is_impossible());
        let guarantees = quantified_over(
            no_rows,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x = ALL (SELECT v FROM q)",
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_null_any_over_a_null_member_keeps_a_null_or_an_apart_side() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ANY (ARRAY[1, NULL])"),
            Outcomes::NULL,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(2))));
        assert!(!x.may_be(&PgValue::Numeric(PgNumeric::from(1))));
    }

    #[test]
    fn a_quantified_outcome_over_an_unbounded_column_admits_every_value() {
        let many = Rows { none: false, one: false, many: true };
        let v =
            Evaluation::new(ScalarType::Numeric, false, false, Values::Unlisted(Unlisted::any()));
        // A row set may hold one value of an unbounded `v`, and every `x`
        // finds one on the side each reading needs.
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Numeric),
            v.clone(),
            "x < ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Numeric),
            v.clone(),
            "x > ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Numeric),
            v.clone(),
            "x < ALL (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Numeric),
            v,
            "x > ALL (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn a_listed_quantified_equality_restricts_the_side() {
        let many = Rows { none: false, one: false, many: true };
        let value = |text: &str| PgValue::Text(text.to_string());
        let v = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a"), value("b")])),
        );
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            v.clone(),
            "x = ANY (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be(&value("a")));
        assert!(!x.may_be(&value("c")));
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            v.clone(),
            "x = ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        let x = column(&guarantees, "x");
        // A row set holding only `b` reads `a` as false.
        assert!(x.may_be(&value("a")));
        assert!(x.may_be(&value("c")));
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            v,
            "x <> ALL (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be(&value("a")));
        assert!(!x.may_be(&value("c")));
    }

    #[test]
    fn a_quantified_null_outcome_admits_any_non_null_side() {
        // `x = ANY (SELECT v)` is `NULL` when `x` is `NULL`, or when a row
        // set holds only `NULL` rows, which any `x` meets.
        let many = Rows { none: false, one: false, many: true };
        let guarantees = quantified_over(
            many,
            unknown_column(ScalarType::Numeric),
            Evaluation::new(
                ScalarType::Numeric,
                true,
                false,
                Values::Listed(BTreeSet::from([PgValue::Numeric(PgNumeric::from(1))])),
            ),
            "x = ANY (SELECT v FROM q)",
            Outcomes::NULL,
        );
        let x = column(&guarantees, "x");
        assert!(x.may_be_null());
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(2))));
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(1))));
    }

    #[test]
    fn a_quantified_outcome_with_a_never_admitted_side_is_impossible_under_the_generic_dialect() {
        // PostgreSQL prunes the form to `NULL` first; the generic dialect
        // keeps it an unknown boolean, and the certainly-`NULL` side admits
        // no member.
        let many = Rows { none: false, one: false, many: true };
        let x = Evaluation::new(ScalarType::Numeric, true, false, Values::none());
        let context = ColumnContext::new().with_column("x", x).with_query(
            many,
            Evaluation::new(ScalarType::Numeric, false, false, Values::Unlisted(Unlisted::any())),
        );
        let guarantees =
            implied_generic(context, &parse("x = ANY (SELECT v FROM q)"), Outcomes::TRUE);
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_not_between_true_keeps_the_apart_values() {
        let values: BTreeSet<PgValue> = [
            PgValue::Numeric(PgNumeric::from(0)),
            PgValue::Numeric(PgNumeric::from(5)),
            PgValue::Numeric(PgNumeric::from(20)),
        ]
        .into();
        let x = Evaluation::new(ScalarType::Numeric, false, false, Values::Listed(values));
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x),
            &parse("x NOT BETWEEN 1 AND 10"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(0))));
        assert!(!x.may_be(&PgValue::Numeric(PgNumeric::from(5))));
        assert!(x.may_be(&PgValue::Numeric(PgNumeric::from(20))));
    }

    #[test]
    fn a_null_between_with_constant_bounds_keeps_only_null() {
        // With both bounds non-`NULL`, the predicate is `NULL` only when
        // `x` is.
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x BETWEEN 1 AND 10"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    #[test]
    fn a_cast_to_a_non_boolean_type_keeps_the_outcome_on_the_operand() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("CAST(x AS boolean)"),
            Outcomes::TRUE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("CAST(x AS boolean)"),
            Outcomes::FALSE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("CAST(x AS boolean)"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    #[test]
    fn a_boolean_cast_passes_the_outcome_to_the_column() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Boolean)),
            &parse("CAST(x AS boolean)"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(x.may_be_true());
        assert!(!x.may_be_false());
    }

    #[test]
    fn an_unbounded_varchar_cast_passes_the_restriction_to_the_column() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("v", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("CAST(v AS VARCHAR(MAX)) = 'a'"),
            Outcomes::TRUE,
        );
        let v = column(&guarantees, "v");
        assert!(!v.may_be_null());
        let Values::Listed(listed) = v.values() else { panic!("v is listed") };
        let expected: BTreeSet<PgValue> = [PgValue::Text("a".to_string())].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_window_call_keeps_no_input_restriction() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(x) OVER ()"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_coalesce_null_points_at_every_argument() {
        // PostgreSQL: `coalesce` is `NULL` exactly when every argument is.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Boolean))
                .with_column("y", unknown_column(ScalarType::Boolean)),
            &parse("coalesce(x, y)"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
        assert!(column(&guarantees, "y").is_always_null());
        // A true `coalesce` may come from either argument, so neither is
        // constrained.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Boolean))
                .with_column("y", unknown_column(ScalarType::Boolean)),
            &parse("coalesce(x, y)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
        assert!(guarantees.reference(&"y".to_string()).is_none());
        // The name reads case-insensitively.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Boolean))
                .with_column("y", unknown_column(ScalarType::Boolean)),
            &parse("COALESCE(x, y)"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
        assert!(column(&guarantees, "y").is_always_null());
    }

    #[test]
    fn a_strict_call_true_keeps_the_argument_non_null() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(x)"),
            Outcomes::TRUE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(x)"),
            Outcomes::FALSE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
    }

    #[test]
    fn a_call_making_several_routines_keeps_the_argument_when_each_routine_nulls_it() {
        let called_on_null = called_on_null(vec![0]);
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean))
                .with_callee("f", called_on_null),
            &parse("f(x)"),
            Outcomes::TRUE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
    }

    #[test]
    fn a_call_null_under_a_null_on_null_input_dialect_points_at_the_argument() {
        let guarantees = implied_null_on_null(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(x)"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    /// A `NULL` comparison with a strict call that answers `NULL` only on a
    /// `NULL` argument points that argument at `NULL` when the other side
    /// is never `NULL`.
    #[test]
    fn a_comparison_null_under_a_null_on_null_input_dialect_points_at_the_call_argument() {
        // PostgreSQL reads `NULL < 'b'` as `NULL`.
        let guarantees = implied_null_on_null(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", plain_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee(
                    "f",
                    crate::test_context::strict_callee(ScalarType::Text(TextRules::DEFAULT)),
                ),
            &parse("f(x) < y"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    #[test]
    fn a_called_on_null_callee_null_points_at_the_argument_under_a_null_on_null_input_dialect() {
        let called_on_null = called_on_null(vec![0]);
        let guarantees = implied_null_on_null(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("g", called_on_null),
            &parse("g(x)"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    #[test]
    fn a_called_on_null_callee_null_under_postgres_points_at_nothing() {
        // PostgreSQL's `NULL` result may come from non-`NULL` input, so no
        // argument is constrained.
        let called_on_null = called_on_null(vec![0]);
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("g", called_on_null),
            &parse("g(x)"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_call_null_over_no_argument_keeps_nothing() {
        // The routine may answer `NULL` however it likes, so no argument is
        // pointed at.
        let no_position = called_on_null(Vec::new());
        let guarantees = implied_null_on_null(
            ColumnContext::new().with_callee("g", no_position),
            &parse("g()"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.references().next().is_none());
    }

    #[test]
    fn a_length_call_null_keeps_no_argument_restriction() {
        // PostgreSQL reads `length(x)` `NULL` only for a `NULL` `x`; a
        // non-boolean call carries no expression subject, so the backward
        // pass states the conservative nothing on the argument.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &parse("length(x)"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_named_argument_keeps_its_subject_non_null() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(a => x)"),
            Outcomes::TRUE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
    }

    #[test]
    fn a_wildcard_argument_keeps_no_subject() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(*)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.references().next().is_none());
    }

    #[test]
    fn a_out_of_range_called_on_null_position_keeps_no_subject() {
        // The routine nulls its second argument, which the call holds no.
        let called_on_null = called_on_null(vec![1]);
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("g", called_on_null),
            &parse("g(x)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_null_call_comparison_points_at_the_call_and_the_other_side() {
        // `f(1)` carries no column, so its `NULL` reading stands on the
        // call alone.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(1) IS NULL"),
            Outcomes::TRUE,
        );
        let call = guarantees.expression(&parse("f(1)")).expect("the call is guaranteed");
        assert!(call.is_always_null());
        // A `NULL` equality with a strict call side reads the call as
        // `NULL`, or the other side as `NULL`.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee(
                    "f",
                    crate::test_context::strict_callee(ScalarType::Text(TextRules::DEFAULT)),
                ),
            &parse("f(x) = x"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_length_call_null_equality_points_at_the_argument() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &parse("length(x) = 5"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    #[test]
    fn an_exists_keeps_no_column_restriction() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("v", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_query(
                    Rows { none: false, one: false, many: true },
                    Evaluation::new(
                        ScalarType::Text(TextRules::DEFAULT),
                        false,
                        false,
                        Values::Unlisted(Unlisted::any()),
                    ),
                ),
            &parse("EXISTS (SELECT v FROM q)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn a_length_equality_keeps_the_values_of_that_length() {
        let value = |text: &str| PgValue::Text(text.to_string());
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(BTreeSet::from([value("a"), value("bb")])),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x).with_callee("length", length_callee()),
            &parse("length(x) = 2"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Listed(listed) = x.values() else { panic!("x is listed") };
        let expected: BTreeSet<PgValue> = [value("bb")].into();
        assert_eq!(listed, &expected);
    }

    #[test]
    fn a_length_inequality_keeps_no_argument_restriction() {
        // PostgreSQL admits every `x` whose length is not `5`, `NULL`
        // among them; the length range model keeps no single interval for
        // a `<>` outcome, so the argument stays unconstrained.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &parse("length(x) <> 5"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn a_length_bound_meets_the_known_length_range() {
        let long = Unlisted {
            excluded: BTreeSet::new(),
            numbers: Interval::unbounded(),
            length: Interval { low: Bound::Excluded(5), high: Bound::Unbounded },
        };
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Unlisted(long),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x).with_callee("length", length_callee()),
            &parse("length(x) < 10"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.length.low, Bound::Excluded(5));
        assert_eq!(unlisted.length.high, Bound::Excluded(10));
    }

    #[test]
    fn a_length_bound_widens_a_range_past_the_numeric_range() {
        // A character count past `i64` reads no numeric bound.
        let huge = Unlisted {
            excluded: BTreeSet::new(),
            numbers: Interval::unbounded(),
            length: Interval { low: Bound::Included(u64::MAX), high: Bound::Unbounded },
        };
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Unlisted(huge),
        );
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", x).with_callee("length", length_callee()),
            &parse("length(x) <= 10"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.length.high, Bound::Included(10));
        // A character length is never negative, so the range reads from
        // `0`.
        assert_eq!(unlisted.length.low, Bound::Included(0));
    }

    #[test]
    fn a_negative_length_bound_meets_to_nothing() {
        // A character length is never negative.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &parse("length(x) <= -3"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    #[test]
    fn a_fractional_length_bound_keeps_no_high_bound() {
        // PostgreSQL: a character length below `5.5` is at most `5`; the
        // crate reads a non-integral bound as no character count, keeping
        // the sound open end, while a length is never negative.
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("length", length_callee()),
            &parse("length(x) < 5.5"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.length.low, Bound::Included(0));
        assert_eq!(unlisted.length.high, Bound::Unbounded);
    }

    #[test]
    fn a_counting_dialect_reads_the_first_argument_as_the_length() {
        let guarantees = implied_counting(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee(
                    "f",
                    crate::test_context::strict_callee(ScalarType::Integer(
                        sql_values::postgres::IntegerType::Integer,
                    )),
                ),
            &parse("f(x) < 10"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.length.high, Bound::Excluded(10));
    }

    #[test]
    fn a_counting_dialect_over_an_unlisted_argument_keeps_nothing_when_it_conflicts() {
        let x = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::unbounded(),
                length: Interval { low: Bound::Included(1), high: Bound::Included(2) },
            }),
        );
        let guarantees = implied_counting(
            ColumnContext::new().with_column("x", x).with_callee(
                "f",
                crate::test_context::strict_callee(ScalarType::Integer(
                    sql_values::postgres::IntegerType::Integer,
                )),
            ),
            &parse("f(x) < 1"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    /// A non-value comparison under `ANY` narrows no input.
    #[test]
    fn a_pattern_match_any_narrows_nothing() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x LIKE ANY (ARRAY['a%'])"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    /// An unregistered subquery yields no output column, so `ANY` narrows no
    /// input.
    #[test]
    fn an_unregistered_subquery_any_narrows_nothing() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ANY (SELECT v FROM q)"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    /// `ALL` over a possibly-empty subquery reading true narrows no input
    /// value.
    #[test]
    fn a_possibly_empty_all_true_narrows_nothing() {
        let rows = Rows { none: true, one: true, many: false };
        let guarantees = quantified_over(
            rows,
            unknown_column(ScalarType::Numeric),
            unknown_column(ScalarType::Numeric),
            "x = ALL (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    /// A cast of an array that folds to no listed value narrows no input.
    #[test]
    fn a_non_listed_cast_array_narrows_nothing() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Numeric))
                .with_column("y", unknown_column(ScalarType::Numeric)),
            &parse("x = ANY ((ARRAY[y])::numeric[])"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    /// An empty cast array reads `FALSE` to `ANY` and `TRUE` to `ALL`, for
    /// every input.
    #[test]
    fn an_empty_cast_array_reads_the_empty_set_outcomes() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ANY ((ARRAY[])::numeric[])"),
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ALL ((ARRAY[])::numeric[])"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ANY ((ARRAY[])::numeric[])"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    /// A `NULL` `IN` with a non-listed member admits a `NULL` subject.
    #[test]
    fn an_in_with_a_non_listed_member_keeps_the_null_subject() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x IN (y, 'a')"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").may_be_null());
    }

    /// A `NULL` `= ALL` over a nullable column admits a `NULL` subject.
    #[test]
    fn an_all_null_over_a_nullable_column_keeps_the_null_subject() {
        let guarantees = quantified(
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            unknown_column(ScalarType::Text(TextRules::DEFAULT)),
            "x = ALL (SELECT v FROM q)",
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").may_be_null());
    }

    /// Two routines nulling the same argument still point at it once.
    #[test]
    fn two_routines_nulling_one_argument_keep_it_non_null() {
        let called_on_null = called_on_null(vec![0]);
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", called_on_null.clone())
                .with_callee("f", called_on_null),
            &parse("f(x)"),
            Outcomes::TRUE,
        );
        assert!(!column(&guarantees, "x").may_be_null());
    }

    /// Two routines nulling the same argument point a `NULL` result at it once.
    #[test]
    fn two_routines_nulling_one_argument_keep_the_null_subject() {
        let called_on_null = called_on_null(vec![0]);
        let guarantees = implied_null_on_null(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", called_on_null.clone())
                .with_callee("f", called_on_null),
            &parse("f(x)"),
            Outcomes::NULL,
        );
        assert!(column(&guarantees, "x").is_always_null());
    }

    /// A routine nulling an argument the call holds no points a `NULL` result
    /// at nothing.
    #[test]
    fn an_out_of_range_null_position_keeps_no_subject() {
        let called_on_null = called_on_null(vec![5]);
        let guarantees = implied_null_on_null(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", called_on_null),
            &parse("f(x)"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(guarantees.reference(&"x".to_string()).is_none());
    }

    /// A `NULL` result of a wildcard call points at no argument.
    #[test]
    fn a_wildcard_call_null_keeps_no_subject() {
        let guarantees = implied_null_on_null(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("f", crate::test_context::strict_callee(ScalarType::Boolean)),
            &parse("f(*) = x"),
            Outcomes::NULL,
        );
        assert!(!guarantees.is_impossible());
        assert!(nullable(&guarantees, "x"));
    }

    /// A `NULL` subject to an argument-less call narrows nothing.
    #[test]
    fn an_argument_less_call_null_keeps_no_subject() {
        let guarantees =
            implied_null_on_null(ColumnContext::new(), &parse("current_user = x"), Outcomes::NULL);
        assert!(!guarantees.is_impossible());
    }

    /// A value-preserving text cast of a `TRY_CAST` carries no value
    /// restriction.
    #[test]
    fn a_try_cast_carries_no_value_restriction() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("TRY_CAST(x AS text) = 'a'"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// A `varchar` typmod with a limit keeps the cast from passing its values
    /// through.
    #[test]
    fn a_limited_string_cast_carries_no_value_restriction() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x::string(3) = 'a'"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// `IS NULL` never reads `NULL`, so a `NULL` outcome is impossible.
    #[test]
    fn is_null_admits_no_null_outcome() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x IS NULL"),
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    /// `IS TRUE` never reads `NULL`, so a `NULL` outcome is impossible.
    #[test]
    fn is_true_admits_no_null_outcome() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Boolean)),
            &parse("x IS TRUE"),
            Outcomes::NULL,
        );
        assert!(guarantees.is_impossible());
    }

    /// A length bound on the right side of a comparison bounds the right's
    /// argument.
    #[test]
    fn a_length_on_the_right_bounds_its_argument() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_callee("char_length", length_callee()),
            &parse("3 < char_length(x)"),
            Outcomes::TRUE,
        );
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        let Values::Unlisted(unlisted) = x.values() else { panic!("x stays unlisted") };
        assert_eq!(unlisted.length.low, Bound::Excluded(3));
    }

    /// A `NULL` subject to a nested comparison keeps every operand nullable.
    #[test]
    fn a_null_nested_comparison_keeps_every_operand_nullable() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Numeric))
                .with_column("y", unknown_column(ScalarType::Numeric))
                .with_column("z", unknown_column(ScalarType::Numeric)),
            &parse("x = (y > z)"),
            Outcomes::NULL,
        );
        assert!(nullable(&guarantees, "x"));
        assert!(nullable(&guarantees, "y"));
        assert!(nullable(&guarantees, "z"));
    }

    /// An `IN` over an always-`NULL` member is impossible to read true.
    #[test]
    fn an_in_over_an_always_null_member_is_impossible_to_read_true() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", Evaluation::null(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x IN (y)"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    /// `= ALL` over two distinct values is impossible to read true.
    #[test]
    fn an_all_over_two_values_is_impossible_to_read_true() {
        let guarantees = implied_with(
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Numeric)),
            &parse("x = ALL (ARRAY[1, 2])"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    /// `> ANY` reading false over an unbounded value needs `x` at most one
    /// row's value, which every `x` finds.
    #[test]
    fn a_greater_any_reading_false_over_an_unbounded_value_admits_every_value() {
        let guarantees = quantified(
            unknown_column(ScalarType::Numeric),
            unknown_column(ScalarType::Numeric),
            "x > ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// A text ordering under a non-byte collation admits no value restriction.
    #[test]
    fn a_non_byte_text_ordering_is_unrestricted() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::DEFAULT)))
                .with_column("y", unknown_column(ScalarType::Text(TextRules::DEFAULT))),
            &parse("x < y"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(!column(&guarantees, "x").may_be_null());
    }

    /// A date ordering over an unlisted date admits no value restriction.
    #[test]
    fn a_date_ordering_over_an_unlisted_date_is_unrestricted() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Date))
                .with_column("y", unknown_column(ScalarType::Date)),
            &parse("x < y"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
        assert!(!column(&guarantees, "x").may_be_null());
    }

    /// A listed date `ANY` reading false bounds the value from above.
    #[test]
    fn a_listed_date_any_reading_false_bounds_above() {
        let v = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Date(sql_values::postgres::parse_date("2020-01-01").expect("a date")),
                PgValue::Date(sql_values::postgres::parse_date("2020-01-03").expect("a date")),
            ])),
        );
        let guarantees = quantified_over(
            Rows::ANY,
            unknown_column(ScalarType::Date),
            v,
            "x < ANY (SELECT v FROM q)",
            Outcomes::FALSE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// A listed date `ALL` reading true bounds the value from above.
    #[test]
    fn a_listed_date_all_reading_true_bounds_above() {
        let v = Evaluation::new(
            ScalarType::Date,
            false,
            false,
            Values::Listed(BTreeSet::from([
                PgValue::Date(sql_values::postgres::parse_date("2020-01-01").expect("a date")),
                PgValue::Date(sql_values::postgres::parse_date("2020-01-03").expect("a date")),
            ])),
        );
        let guarantees = quantified_over(
            Rows::ANY,
            unknown_column(ScalarType::Date),
            v,
            "x > ALL (SELECT v FROM q)",
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// A numeric value in a closed range that no admitted value meets is
    /// impossible.
    #[test]
    fn a_range_that_no_value_meets_is_impossible() {
        let x = Evaluation::new(
            ScalarType::Numeric,
            false,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval {
                    low: Bound::Included(PgNumeric::from(10)),
                    high: Bound::Included(PgNumeric::from(20)),
                },
                length: Interval::unbounded(),
            }),
        );
        let guarantees =
            implied_with(ColumnContext::new().with_column("x", x), &parse("x < 5"), Outcomes::TRUE);
        assert!(guarantees.is_impossible());
    }

    /// A text ordering over an unlisted value keeps the value untouched.
    #[test]
    fn a_text_ordering_over_an_unlisted_value_keeps_it() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Text(TextRules::BYTEWISE))),
            &parse("x < 'b'"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// An argument-less call read as a length bounds no argument.
    #[test]
    fn an_argument_less_call_read_as_a_length_bounds_nothing() {
        let guarantees =
            implied_counting(ColumnContext::new(), &parse("current_user < 10"), Outcomes::TRUE);
        assert!(!guarantees.is_impossible());
    }

    /// A wildcard call read as a length bounds no argument.
    #[test]
    fn a_wildcard_call_read_as_a_length_bounds_nothing() {
        let guarantees = implied_counting(
            ColumnContext::new().with_callee(
                "f",
                crate::test_context::strict_callee(ScalarType::Integer(
                    sql_values::postgres::IntegerType::Integer,
                )),
            ),
            &parse("f(*) < 10"),
            Outcomes::TRUE,
        );
        assert!(!guarantees.is_impossible());
    }

    /// A length over a non-text value is impossible to satisfy.
    #[test]
    fn a_length_over_a_non_text_value_is_impossible() {
        let guarantees = implied_with(
            ColumnContext::new()
                .with_column(
                    "x",
                    unknown_column(ScalarType::Integer(sql_values::postgres::IntegerType::Integer)),
                )
                .with_callee("char_length", length_callee()),
            &parse("char_length(x) = 3"),
            Outcomes::TRUE,
        );
        assert!(guarantees.is_impossible());
    }

    /// A length over a type the evaluator does not model, such as a
    /// `char(n)` cast, bounds nothing.
    #[test]
    fn a_length_over_an_unmodelled_value_bounds_nothing() {
        for (ty, sql) in [
            (ScalarType::Date, "char_length(x::char(2)) < 5"),
            (ScalarType::Other, "char_length(x) < 5"),
            (ScalarType::Unknown, "char_length(x) < 5"),
        ] {
            let guarantees = implied_with(
                ColumnContext::new()
                    .with_column("x", unknown_column(ty))
                    .with_callee("char_length", length_callee()),
                &parse(sql),
                Outcomes::TRUE,
            );
            assert!(!guarantees.is_impossible(), "{sql}");
        }
    }

    /// A context that names a text column `x` and reads a `text` cast as
    /// unknown, like the forward pass does.
    #[derive(Default)]
    struct RemappedText;

    impl Context for RemappedText {
        type Reference = String;
        type Error = core::convert::Infallible;

        fn reference(
            &mut self,
            expression: &Expr,
        ) -> Result<Option<crate::context::Resolved<String>>, Self::Error> {
            let Expr::Identifier(sqlparser::ast::Ident { value, .. }) = expression else {
                return Ok(None);
            };
            Ok((value.as_str() == "x").then(|| {
                crate::context::Resolved {
                    reference: "x".to_string(),
                    evaluation: unknown_column(ScalarType::Text(TextRules::DEFAULT)),
                }
            }))
        }

        fn callees(
            &mut self,
            _call: &sqlparser::ast::Function,
        ) -> Result<alloc::vec::Vec<crate::Callee>, Self::Error> {
            Ok(alloc::vec::Vec::new())
        }

        fn data_type(&self, data_type: &sqlparser::ast::DataType) -> ScalarType {
            if matches!(data_type, sqlparser::ast::DataType::Text) {
                ScalarType::Unknown
            } else {
                ScalarType::of(data_type)
            }
        }
    }

    /// A cast's backward narrowing reads its target type from the context's
    /// `data_type`, the way the forward pass does.
    #[test]
    fn a_cast_reads_its_target_type_from_the_context() {
        let mut context = RemappedText;
        let guarantees = implied(
            &parse("x::text"),
            Outcomes::TRUE,
            &PostgreSqlDialect {},
            &PostgreSqlOptions::default(),
            &mut context,
            &Guarantees::new(),
        )
        .expect("the context resolves");
        assert!(!guarantees.is_impossible());
        let x = column(&guarantees, "x");
        assert!(!x.may_be_null());
        assert!(context.reference(&number("1")).unwrap().is_none());
        assert!(
            context
                .callees(&sqlparser::ast::Function {
                    name: sqlparser::ast::ObjectName(vec![
                        sqlparser::ast::ObjectNamePart::Identifier(sqlparser::ast::Ident::new(
                            "length"
                        ),)
                    ]),
                    uses_odbc_syntax: false,
                    parameters: sqlparser::ast::FunctionArguments::None,
                    args: sqlparser::ast::FunctionArguments::None,
                    within_group: vec![],
                    filter: None,
                    null_treatment: None,
                    over: None,
                })
                .unwrap()
                .is_empty()
        );
        assert_eq!(
            context.data_type(&sqlparser::ast::DataType::Integer(None)),
            ScalarType::Integer(sql_values::postgres::IntegerType::Integer)
        );
    }

    use sqlparser::{
        ast::{BinaryOperator, Expr, Ident, UnaryOperator, Value, ValueWithSpan},
        tokenizer::Span,
    };

    /// The boolean column `name` as a read expression.
    fn column_expr(name: &str) -> Expr {
        Expr::Identifier(Ident::new(name.to_string()))
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

    /// Runs `implied` under `NOT_FALSE` on a 2 MiB thread and returns the
    /// guarantees. The deep expression is leaked: a left-nested tree's
    /// recursive destructor would itself overflow the small stack, so it must
    /// not run there.
    fn implied_on_small_stack(expression: Expr, mut context: ColumnContext) -> Guarantees<String> {
        std::thread::Builder::new()
            .stack_size(2 * 1024 * 1024)
            .spawn(move || {
                let result = implied(
                    &expression,
                    Outcomes::NOT_FALSE,
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
    fn a_5_672_term_and_chain_narrows_on_a_small_stack() {
        let terms = DEPTH;
        let mut expression = column_expr("c0");
        for i in 1..terms {
            expression = Expr::BinaryOp {
                left: Box::new(expression),
                op: BinaryOperator::And,
                right: Box::new(column_expr(&format!("c{i}"))),
            };
        }
        let mut context = ColumnContext::new();
        for i in 0..terms {
            context = context.with_column(&format!("c{i}"), unknown_column(ScalarType::Boolean));
        }
        let guarantees = implied_on_small_stack(expression, context);
        assert!(!guarantees.is_impossible());
        // A conjunction that is not false leaves no term free to be false.
        let c0 = guarantees.reference(&"c0".to_string()).expect("c0 is narrowed");
        assert!(!c0.may_be_false());
        assert!(c0.may_be_true());
    }

    #[test]
    fn a_5_672_term_or_chain_narrows_on_a_small_stack() {
        let terms = DEPTH;
        let mut expression = column_expr("c0");
        for i in 1..terms {
            expression = Expr::BinaryOp {
                left: Box::new(expression),
                op: BinaryOperator::Or,
                right: Box::new(column_expr(&format!("c{i}"))),
            };
        }
        let mut context = ColumnContext::new();
        for i in 0..terms {
            context = context.with_column(&format!("c{i}"), unknown_column(ScalarType::Boolean));
        }
        let guarantees = implied_on_small_stack(expression, context);
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn a_5_672_nested_nots_narrow_on_a_small_stack() {
        let mut expression = column_expr("x");
        for _ in 0..DEPTH {
            expression = Expr::UnaryOp { op: UnaryOperator::Not, expr: Box::new(expression) };
        }
        let guarantees = implied_on_small_stack(
            expression,
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Boolean)),
        );
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn a_5_672_nested_parentheses_narrow_on_a_small_stack() {
        let mut expression = column_expr("x");
        for _ in 0..DEPTH {
            expression = Expr::Nested(Box::new(expression));
        }
        let guarantees = implied_on_small_stack(
            expression,
            ColumnContext::new().with_column("x", unknown_column(ScalarType::Boolean)),
        );
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn a_5_672_nested_additions_narrow_on_a_small_stack() {
        let one = number("1");
        let mut expression = column_expr("x");
        for _ in 0..DEPTH {
            expression = Expr::BinaryOp {
                left: Box::new(expression),
                op: BinaryOperator::Plus,
                right: Box::new(one.clone()),
            };
        }
        let guarantees = implied_on_small_stack(
            expression,
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Integer(IntegerType::Integer))),
        );
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn a_5_672_nested_negations_narrow_on_a_small_stack() {
        let mut expression = column_expr("x");
        for _ in 0..DEPTH {
            expression = Expr::UnaryOp { op: UnaryOperator::Minus, expr: Box::new(expression) };
        }
        let guarantees = implied_on_small_stack(
            expression,
            ColumnContext::new()
                .with_column("x", unknown_column(ScalarType::Integer(IntegerType::Integer))),
        );
        assert!(!guarantees.is_impossible());
    }

    /// A qualified or quoted `coalesce` names a routine, which may return
    /// `NULL` on non-`NULL` input, so its `NULL` outcome narrows nothing.
    #[test]
    fn a_qualified_coalesce_narrows_nothing() {
        for sql in ["rv.coalesce(x, y)", "\"coalesce\"(x, y)"] {
            let guarantees = implied_with(
                ColumnContext::new()
                    .with_column("x", unknown_column(ScalarType::Boolean))
                    .with_column("y", unknown_column(ScalarType::Boolean)),
                &parse(sql),
                Outcomes::NULL,
            );
            assert!(guarantees.reference(&"x".to_string()).is_none(), "{sql}");
            assert!(guarantees.reference(&"y".to_string()).is_none(), "{sql}");
        }
    }
}
