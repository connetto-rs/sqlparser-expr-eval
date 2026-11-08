//! Calls, `CASE`, `IN`, `LIKE`, subqueries and `IS` tests, forward.

use alloc::{collections::BTreeSet, string::String, vec::Vec};
use core::{fmt::Write, ops::Bound};

use sql_values::postgres::{IntegerType, PgNumeric, PgValue};
use sqlparser::ast::{Array, BinaryOperator, CaseWhen, Expr, Function, InTarget, Query};

use super::{
    super::{
        context::{Builtin, CalleeKind, Context, NullInput, Rows},
        dialect::EvalDialect,
        domain::{Evaluation, Interval, Outcomes, Unlisted, Values},
        scalar::{ScalarType, TextRules},
        support,
    },
    State,
    fold::{Compared, RowKinds, comparison_evals},
    forward,
};

/// The forward evaluation of a function call.
pub(super) fn call<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    function: &Function,
) -> Result<Evaluation, C::Error> {
    let arguments = support::call_arguments(function);
    let mut argument_evaluations: Vec<Evaluation> = Vec::new();
    let mut error = false;
    for argument in &arguments {
        let Some(argument) = argument else {
            argument_evaluations.push(Evaluation::impossible(ScalarType::Unknown));
            continue;
        };
        let evaluation = forward(state, argument)?;
        error |= evaluation.may_raise();
        argument_evaluations.push(evaluation);
    }
    if !arguments.is_empty()
        && function.over.is_none()
        && let Some(name) = support::simple_name(&function.name)
        && let Some(evaluation) =
            state.dialect.named_call(&name, &argument_evaluations, state.options)
    {
        return Ok(evaluation);
    }
    let callees = state.context.callees(function)?;
    // A call no known function answers may name none at all.
    if callees.is_empty() {
        return Ok(Evaluation::unknown(ScalarType::Unknown).with_error(true));
    }
    let returns = callees.first().map_or(ScalarType::Unknown, |callee| callee.returns);
    if function.over.is_some()
        || callees
            .iter()
            .any(|callee| matches!(callee.kind, CalleeKind::Aggregate | CalleeKind::Window))
    {
        let count = callees
            .iter()
            .all(|callee| callee.builtin.is_some_and(|builtin| builtin == Builtin::Count));
        return Ok(Evaluation::new(returns, !count, error, Values::Unlisted(Unlisted::any())));
    }
    let mut null = false;
    // The result is forced `NULL` only when every callee the call may reach
    // answers `NULL` for the `NULL` input.
    let mut forced_null = true;
    for callee in &callees {
        let nulling = |test| nulling_argument(&callee.null_input, &argument_evaluations, test);
        forced_null &= nulling(Evaluation::is_always_null);
        null |= nulling(Evaluation::may_be_null);
        match callee.builtin {
            None => {
                // An unknown routine may return `NULL` on non-`NULL` input.
                null = true;
            }
            Some(Builtin::ConcatWs) => {
                null |= argument_evaluations.first().is_some_and(Evaluation::may_be_null);
            }
            Some(_) => {}
        }
    }
    if forced_null {
        return Ok(Evaluation::null(returns).with_error(error));
    }
    let mut evaluation = Evaluation::new(returns, null, error, Values::Unlisted(Unlisted::any()));
    // The values fold only when every callee is the same builtin.
    let first_builtin = callees.first().and_then(|callee| callee.builtin);
    if function.over.is_none()
        && first_builtin
            .is_some_and(|first| callees.iter().all(|callee| callee.builtin == Some(first)))
        && let Some(builtin) = first_builtin
        && let Some(folded) = fold_builtin(builtin, &argument_evaluations, returns)
    {
        let raises = folded.may_raise() || error;
        evaluation = folded.with_error(raises);
    }
    // The call may reach any callee, so the answer joins each callee's
    // refined reading, unrefined for a callee the context reads no body of.
    let mut refinements: Vec<Evaluation> = Vec::with_capacity(callees.len());
    for callee in &callees {
        let mut refined = evaluation.clone();
        if let Some(body) = state.context.evaluate_body(callee, &argument_evaluations)? {
            // The arguments raise before the body runs, and a `NULL` argument
            // the callee is strict in answers `NULL` without running it.
            let null = body.may_be_null()
                || nulling_argument(
                    &callee.null_input,
                    &argument_evaluations,
                    Evaluation::may_be_null,
                );
            refined = refined.meet(&body).with_null(null).with_error(body.may_raise() || error);
        }
        refinements.push(refined);
    }
    Ok(refinements
        .into_iter()
        .reduce(|joined, refined| joined.join(&refined))
        .unwrap_or(evaluation))
}

/// Whether an argument the callee answers `NULL` for passes `test`.
fn nulling_argument(
    null_input: &NullInput,
    arguments: &[Evaluation],
    test: fn(&Evaluation) -> bool,
) -> bool {
    arguments
        .iter()
        .enumerate()
        .any(|(position, argument)| null_input.nulls_at(position) && test(argument))
}

/// The folds of the builtins the core recognises, `None` where the values
/// stay unknown.
fn fold_builtin(
    builtin: Builtin,
    arguments: &[Evaluation],
    returns: ScalarType,
) -> Option<Evaluation> {
    match builtin {
        Builtin::Count => {
            Some(Evaluation::new(
                returns,
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
            ))
        }
        Builtin::Length => fold_length(arguments, returns, false),
        Builtin::OctetLength => {
            // A byte count bounds the character count from below, but the
            // encoding's factor is unknown.
            fold_length(arguments, returns, true)
        }
        Builtin::Lower | Builtin::Upper => fold_case(builtin, arguments),
        Builtin::Btrim => fold_btrim(arguments),
        Builtin::Abs => fold_abs(arguments, returns),
        Builtin::Concat => fold_concat(arguments),
        Builtin::ConcatWs => fold_concat_ws(arguments, returns),
    }
}

/// Whether a `length` argument holds text or a not-yet-typed literal.
fn text_argument(evaluation: &Evaluation) -> bool {
    evaluation.ty().is_text() || evaluation.ty() == ScalarType::UnknownLiteral
}

/// The `length` and `octet_length` folds, counting bytes instead of
/// characters when `bytes`.
fn fold_length(arguments: &[Evaluation], returns: ScalarType, bytes: bool) -> Option<Evaluation> {
    let argument = arguments.first()?;
    if !text_argument(argument) {
        return None;
    }
    let count = single_value(argument).and_then(|value| {
        if bytes {
            value.as_text().and_then(|text| u64::try_from(text.len()).ok())
        } else {
            crate::scalar::text_length(&value)
        }
    });
    match count {
        Some(count) => {
            Some(Evaluation::known(
                PgValue::Integer(integer_of(count)),
                ScalarType::Integer(IntegerType::Integer),
            ))
        }
        None => length_interval(argument, returns, bytes),
    }
}

/// The `lower` and `upper` fold on a byte-order text argument.
fn fold_case(builtin: Builtin, arguments: &[Evaluation]) -> Option<Evaluation> {
    let ScalarType::Text(rules) = argument_type(arguments)? else {
        return None;
    };
    if !rules.byte_order {
        return None;
    }
    let text = single_text(&arguments[0])?;
    let mapped: String = text
        .chars()
        .map(|ch| {
            match builtin {
                Builtin::Lower => ch.to_ascii_lowercase(),
                _ => ch.to_ascii_uppercase(),
            }
        })
        .collect();
    Some(Evaluation::known(PgValue::Text(mapped), ScalarType::Text(TextRules::BYTEWISE)))
}

/// The `btrim` fold of a constant text argument.
fn fold_btrim(arguments: &[Evaluation]) -> Option<Evaluation> {
    let text = single_text(&arguments[0])?;
    let characters = match arguments.get(1) {
        Some(set) => single_text(set)?,
        None => String::from(" "),
    };
    let set: Vec<char> = characters.chars().collect();
    let trimmed = {
        let mut chars: Vec<char> = text.chars().collect();
        while let Some(last) = chars.last()
            && set.contains(last)
        {
            chars.pop();
        }
        let front = chars.iter().position(|ch| !set.contains(ch)).unwrap_or(0);
        chars[front..].iter().copied().collect()
    };
    Some(Evaluation::known(PgValue::Text(trimmed), ScalarType::Text(TextRules::DEFAULT)))
}

/// The `abs` fold of a single listed value.
fn fold_abs(arguments: &[Evaluation], returns: ScalarType) -> Option<Evaluation> {
    let argument = arguments.first()?;
    // An untyped literal is read through a number type's input, which may
    // refuse it.
    if argument.ty() == ScalarType::UnknownLiteral {
        return Some(Evaluation::new(
            returns,
            argument.may_be_null(),
            true,
            Values::Unlisted(Unlisted::any()),
        ));
    }
    let Some(datum) = single_value(argument) else {
        // The absolute of the type's minimum is past its range.
        let ScalarType::Integer(integer) = argument.ty() else { return None };
        let minimum = PgNumeric::from(*integer.range().start());
        let raises = argument.number_range().contains(&minimum);
        return raises.then(|| {
            Evaluation::new(
                returns,
                argument.may_be_null(),
                true,
                Values::Unlisted(Unlisted::any()),
            )
        });
    };
    match datum {
        PgValue::Integer(value) => {
            // The absolute of the type's or `i64` minimum is past its range.
            if value == i64::MIN
                || matches!(
                    argument.ty(),
                    ScalarType::Integer(integer) if value == *integer.range().start()
                )
            {
                return Some(Evaluation::raising(returns));
            }
            Some(Evaluation::known(PgValue::Integer(value.abs()), returns))
        }
        PgValue::Numeric(numeric) => {
            Some(Evaluation::known(PgValue::Numeric(numeric.abs()), returns))
        }
        _ => None,
    }
}

/// The `concat` fold of constant arguments.
fn fold_concat(arguments: &[Evaluation]) -> Option<Evaluation> {
    let mut out = String::new();
    for argument in arguments {
        if argument.is_always_null() {
            continue;
        }
        let value = single_value(argument)?;
        write!(out, "{}", value.output()?).ok()?;
    }
    Some(Evaluation::known(PgValue::Text(out), ScalarType::Text(TextRules::DEFAULT)))
}

/// The `concat_ws` fold of constant arguments.
fn fold_concat_ws(arguments: &[Evaluation], returns: ScalarType) -> Option<Evaluation> {
    let separator = arguments.first()?;
    if separator.is_always_null() {
        return Some(Evaluation::null(returns));
    }
    let separator_value = single_value(separator)?;
    let separator = separator_value.output()?;
    let mut out = String::new();
    let mut first = true;
    for argument in arguments.iter().skip(1) {
        if argument.is_always_null() {
            continue;
        }
        let value = single_value(argument)?;
        let part = value.output()?;
        if first {
            write!(out, "{part}").ok()?;
        } else {
            write!(out, "{separator}{part}").ok()?;
        }
        first = false;
    }
    Some(Evaluation::known(PgValue::Text(out), ScalarType::Text(TextRules::DEFAULT)))
}

/// The one value an evaluation always takes.
fn single_value(evaluation: &Evaluation) -> Option<PgValue> {
    evaluation.constant().cloned()
}

/// The type that holds the values of both `ty` and `branch`, and whether an
/// engine may refuse to join them.
fn joined_type(ty: ScalarType, branch: ScalarType) -> (ScalarType, bool) {
    let untyped = |t: ScalarType| matches!(t, ScalarType::Unknown | ScalarType::UnknownLiteral);
    let number = |t: ScalarType| t.is_number() || t == ScalarType::Float;
    if ty == branch || (untyped(branch) && ty != ScalarType::Unknown) {
        (ty, false)
    } else if untyped(ty) {
        (branch, false)
    } else if number(ty) && number(branch) {
        (super::fold::arithmetic_type(ty, branch), false)
    } else {
        (ScalarType::Other, true)
    }
}

/// The evaluation read as the type a join resolved it to.
fn coerce_to(evaluation: Evaluation, ty: ScalarType) -> Evaluation {
    let from = evaluation.ty();
    if from == ty || matches!(ty, ScalarType::Unknown | ScalarType::UnknownLiteral) {
        return evaluation;
    }
    match (from, ty) {
        (ScalarType::UnknownLiteral, _) => super::fold::reinterpret_unknown(evaluation, ty),
        // A `numeric` may overflow a `float`, and unmodelled values stay open.
        (_, ScalarType::Float | ScalarType::Other) => {
            Evaluation::new(
                ty,
                evaluation.may_be_null(),
                evaluation.may_raise() || from == ScalarType::Numeric,
                Values::Unlisted(Unlisted::any()),
            )
        }
        _ => evaluation.with_type(ty),
    }
}

/// The one listed text value an evaluation holds.
fn single_text(evaluation: &Evaluation) -> Option<String> {
    match single_value(evaluation) {
        Some(PgValue::Text(text)) => Some(text),
        _ => None,
    }
}

/// The scalar type of the sole argument.
fn argument_type(arguments: &[Evaluation]) -> Option<ScalarType> {
    let [argument] = arguments else {
        return None;
    };
    Some(argument.ty())
}

/// The integer value of a character count, `i64::MAX` past the range.
fn integer_of(count: u64) -> i64 {
    i64::try_from(count).unwrap_or(i64::MAX)
}

/// The integer interval a text evaluation's character length lies in.
fn length_interval(
    argument: &Evaluation,
    returns: ScalarType,
    open_at_top: bool,
) -> Option<Evaluation> {
    let range = argument.length_range();
    let low = match range.low {
        Bound::Unbounded => Bound::Unbounded,
        Bound::Included(value) => Bound::Included(support::count_numeric(value)?),
        Bound::Excluded(value) => Bound::Excluded(support::count_numeric(value)?),
    };
    let high = if open_at_top {
        Bound::Unbounded
    } else {
        match range.high {
            Bound::Unbounded => Bound::Unbounded,
            Bound::Included(value) => Bound::Included(support::count_numeric(value)?),
            Bound::Excluded(value) => {
                // The character count is integral, so `length < value` is
                // `length <= value - 1`.
                let value = value.checked_sub(1)?;
                Bound::Included(support::count_numeric(value)?)
            }
        }
    };
    Some(Evaluation::new(
        returns,
        argument.may_be_null(),
        false,
        Values::Unlisted(Unlisted {
            excluded: BTreeSet::new(),
            numbers: Interval { low, high },
            length: Interval::unbounded(),
        }),
    ))
}

/// The forward evaluation of a `CASE`.
pub(super) fn case<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    operand: Option<&Expr>,
    conditions: &[CaseWhen],
    else_result: Option<&Expr>,
) -> Result<Evaluation, C::Error> {
    // PostgreSQL reads an untyped operand as `text` before any comparison.
    let operand_evaluation = match operand {
        Some(operand) => {
            let evaluation = forward(state, operand)?;
            let untyped = evaluation.ty() == ScalarType::UnknownLiteral
                || (evaluation.ty() == ScalarType::Unknown && evaluation.is_always_null());
            Some(if untyped {
                coerce_to(evaluation, ScalarType::Text(TextRules::DEFAULT))
            } else {
                evaluation
            })
        }
        None => None,
    };
    let mut null = false;
    let mut error = false;
    let mut ty = ScalarType::Unknown;
    let mut results = Vec::new();
    let mut end_reachable = true;
    for when in conditions {
        // Every condition is type-checked, even past a decisive branch.
        let condition = if let Some(operand) = &operand_evaluation {
            let value = forward(state, &when.condition)?;
            let condition = comparison_evals(state, operand.clone(), &BinaryOperator::Eq, value);
            // The comparison carries every operand and type raise.
            error |= condition.error;
            condition.outcomes()
        } else {
            let evaluation = support::as_condition(forward(state, &when.condition)?);
            error |= evaluation.may_raise();
            evaluation.outcomes()
        };
        // Every branch is typed, even one no row reaches.
        let result = forward(state, &when.result)?;
        let (joined, refused) = joined_type(ty, result.ty());
        ty = joined;
        error |= refused;
        let reachable = end_reachable && condition.truth;
        results.push((result, reachable));
        if !reachable {
            continue;
        }
        if condition.truth && !condition.falsity && !condition.null {
            end_reachable = false;
        }
    }
    if let Some(else_result) = else_result {
        let result = forward(state, else_result)?;
        let (joined, refused) = joined_type(ty, result.ty());
        ty = joined;
        error |= refused;
        results.push((result, end_reachable));
    } else if end_reachable {
        null = true;
    }
    // Untyped literals and `NULL`s alone resolve to `text`, as PostgreSQL
    // resolves them.
    let untyped_null = |(result, _): &(Evaluation, bool)| {
        result.ty() == ScalarType::Unknown && result.is_always_null()
    };
    if ty == ScalarType::UnknownLiteral
        || (ty == ScalarType::Unknown && results.iter().all(untyped_null))
    {
        ty = ScalarType::Text(TextRules::DEFAULT);
    }
    let mut values = Values::none();
    for (result, reachable) in results {
        let result = coerce_to(result, ty);
        // A branch no row reaches may still refuse its types when planned.
        error |= result.may_raise();
        if reachable {
            values = values.join(result.values());
            null |= result.may_be_null();
        }
    }
    Ok(Evaluation::new(ty, null, error, values))
}

/// The forward evaluation of `x IN target`.
pub(super) fn in_target<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
    target: &InTarget,
    negated: bool,
) -> Result<Evaluation, C::Error> {
    let value = forward(state, expr)?;
    if value.is_impossible() {
        return Ok(Evaluation::impossible(ScalarType::Boolean));
    }
    let (o, error) = match target {
        InTarget::List(list) => {
            let mut o = Outcomes { truth: false, falsity: true, null: false };
            let mut error = value.may_raise();
            let mut ty = value.ty();
            let mut members = Vec::new();
            for member in list {
                let member = forward(state, member)?;
                let (joined, refused) = joined_type(ty, member.ty());
                ty = joined;
                error |= refused;
                members.push(member);
            }
            // The list's untyped literals are read as the type it joins to.
            let value =
                if value.ty() == ScalarType::UnknownLiteral { coerce_to(value, ty) } else { value };
            error |= value.may_raise();
            for member in members {
                let member = if member.ty() == ScalarType::UnknownLiteral {
                    coerce_to(member, ty)
                } else {
                    member
                };
                error |= member.may_raise();
                let cmp = comparison_evals(state, value.clone(), &BinaryOperator::Eq, member);
                error |= cmp.error;
                o = support::or_outcomes(o, cmp.outcomes());
            }
            (o, error)
        }
        InTarget::Subquery(subquery) => {
            let SubqueryOperand { compared: cmp, rows, .. } =
                match subquery_comparison(state, value, &BinaryOperator::Eq, subquery)? {
                    Ok(operand) => operand,
                    Err(unknown) => return Ok(unknown),
                };
            let o = if rows.one || rows.many {
                // On the rows the subquery returns the comparison holds, and
                // a possibly-empty subquery is `FALSE` on its empty rows.
                let mut o = cmp.outcomes();
                o.falsity = o.falsity || rows.none;
                o
            } else {
                // A certainly-empty subquery: `x IN (…)` is `FALSE`.
                Outcomes { truth: false, falsity: true, null: false }
            };
            (o, cmp.error)
        }
        InTarget::Unnest(_) | InTarget::Table { .. } => (Outcomes::ANY, value.may_raise()),
    };
    let o = if negated { support::not_outcomes(o) } else { o };
    Ok(Evaluation::condition(o, error))
}

/// The forward evaluation of `LIKE`, `ILIKE` and `SIMILAR TO`; `NOT` keeps
/// the condition unknown.
pub(super) fn like<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
    pattern: &Expr,
    escape: Option<&Expr>,
    similar: bool,
) -> Result<Evaluation, C::Error> {
    let value = forward(state, expr)?;
    let pattern = forward(state, pattern)?;
    let escape = match escape {
        Some(escape) => Some(forward(state, escape)?),
        None => None,
    };
    let text = |ty: ScalarType| {
        matches!(ty, ScalarType::Text(_) | ScalarType::Unknown | ScalarType::UnknownLiteral)
    };
    let mut error =
        value.may_raise() || pattern.may_raise() || !text(value.ty()) || !text(pattern.ty());
    if value.is_impossible()
        || pattern.is_impossible()
        || escape.as_ref().is_some_and(Evaluation::is_impossible)
    {
        return Ok(Evaluation::impossible(ScalarType::Boolean));
    }
    let mut null = value.may_be_null() || pattern.may_be_null();
    // The escape characters the pattern may be read with, `None` among them
    // for an empty escape, which turns escaping off.
    let mut escapes = Vec::new();
    match &escape {
        None => escapes.push(Some('\\')),
        Some(escape) => {
            error |= escape.may_raise() || !text(escape.ty());
            null |= escape.may_be_null();
            match escape.values() {
                Values::Listed(listed) => {
                    for datum in listed {
                        let PgValue::Text(text) = datum else { continue };
                        let mut chars = text.chars();
                        match (chars.next(), chars.next()) {
                            (first, None) => escapes.push(first),
                            // PostgreSQL takes an empty escape or one
                            // character.
                            _ => error = true,
                        }
                    }
                }
                Values::Unlisted(_) => error = true,
            }
        }
    }
    match pattern.values() {
        Values::Listed(listed) => {
            for datum in listed {
                let PgValue::Text(text) = datum else { continue };
                error |= escapes.iter().any(|escape| pattern_may_raise(text, *escape, similar));
            }
        }
        // A pattern the evaluator does not know may end with its escape.
        Values::Unlisted(_) => error = true,
    }
    // The evaluator does not match `LIKE` patterns, so the value stays
    // unknown.
    let o = Outcomes { truth: true, falsity: true, null };
    Ok(Evaluation::condition(o, error))
}

/// Whether matching `pattern` may raise: a pattern ending with an unpaired
/// `escape`, which PostgreSQL refuses once the match reaches it, or a
/// `SIMILAR TO` pattern with a regular expression operator, which may not
/// make a valid regular expression.
fn pattern_may_raise(pattern: &str, escape: Option<char>, similar: bool) -> bool {
    if similar && pattern.contains(['(', ')', '[', ']', '{', '}', '*', '+', '?', '|']) {
        return true;
    }
    let mut chars = pattern.chars();
    while let Some(next) = chars.next() {
        if Some(next) == escape && chars.next().is_none() {
            return true;
        }
    }
    false
}

/// The forward evaluation of `x op ANY/ALL (list | subquery)`.
pub(super) fn any_all<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    op: &BinaryOperator,
    right: &Expr,
    is_any: bool,
) -> Result<Evaluation, C::Error> {
    let value = forward(state, left)?;
    if value.is_impossible() {
        return Ok(Evaluation::impossible(ScalarType::Boolean));
    }
    let unknown = |error: bool| -> Result<Evaluation, C::Error> {
        Ok(Evaluation::new(ScalarType::Boolean, true, error, Values::Unlisted(Unlisted::any())))
    };
    let (value_null, value_has_values) = (value.may_be_null(), !value.values().is_empty());
    let (cmp, members, error) = match support::unnest(right) {
        Expr::Subquery(subquery) => {
            let SubqueryOperand { compared: cmp, column, rows } =
                match subquery_comparison(state, value, op, subquery)? {
                    Ok(operand) => operand,
                    Err(unknown) => return Ok(unknown),
                };
            let error = cmp.error;
            let rows_present = rows.one || rows.many;
            let empty = if rows.none {
                if rows_present { Presence::Sometimes } else { Presence::Always }
            } else {
                Presence::Never
            };
            // A row set may leave out the `NULL` rows a column may hold, or
            // hold nothing else.
            let null = if column.is_always_null() {
                Presence::Always
            } else if column.may_be_null() {
                Presence::Sometimes
            } else {
                Presence::Never
            };
            let members = Members {
                nonempty: !column.values().is_empty(),
                null,
                empty,
                null_only: rows_present && column.may_be_null(),
            };
            (cmp, members, error)
        }
        other => {
            // An engine may refuse a row constructor where it takes an array.
            let (elements, row) = match other {
                Expr::Array(Array { elem, .. }) => (elem.clone(), false),
                Expr::Tuple(elements) => (elements.clone(), true),
                _ => {
                    // An array is `Other`, or `Unknown` when untyped, and
                    // text is read as an array literal that may be malformed.
                    let operand = forward(state, other)?;
                    let array = matches!(operand.ty(), ScalarType::Other | ScalarType::Unknown);
                    return unknown(value.may_raise() || operand.may_raise() || !array);
                }
            };
            let mut values = BTreeSet::new();
            let mut null = false;
            let mut error = value.may_raise() || row;
            let mut ty = value.ty();
            let mut evaluations = Vec::new();
            for element in &elements {
                let evaluation = forward(state, element)?;
                let (joined, refused) = joined_type(ty, evaluation.ty());
                ty = joined;
                error |= refused;
                evaluations.push(evaluation);
            }
            // An engine may type an array of untyped literals as `text[]`.
            let untyped =
                |e: &Evaluation| matches!(e.ty(), ScalarType::UnknownLiteral | ScalarType::Unknown);
            error |= !value.ty().is_text()
                && !untyped(&value)
                && !evaluations.is_empty()
                && evaluations.iter().all(untyped);
            let value =
                if value.ty() == ScalarType::UnknownLiteral { coerce_to(value, ty) } else { value };
            error |= value.may_raise();
            let evaluations: Vec<Evaluation> =
                evaluations.into_iter().map(|evaluation| coerce_to(evaluation, ty)).collect();
            error |= evaluations.iter().any(Evaluation::may_raise);
            for evaluation in evaluations {
                match evaluation.constant() {
                    Some(datum) => {
                        values.insert(datum.clone());
                    }
                    None if evaluation.is_always_null() => null = true,
                    None => {
                        return unknown(error);
                    }
                }
            }
            let set = Evaluation::new(ty, null, false, Values::Listed(values.clone()));
            let cmp = comparison_evals(state, value, op, set);
            let error = cmp.error || error;
            let empty = if values.is_empty() && !null { Presence::Always } else { Presence::Never };
            // An element that is always `NULL` is in every member set.
            let null = if null { Presence::Always } else { Presence::Never };
            let members = Members { nonempty: !values.is_empty(), null, empty, null_only: false };
            (cmp, members, error)
        }
    };
    Ok(Evaluation::condition(
        quantified_outcomes(cmp.rows, &members, value_null, value_has_values, is_any),
        error,
    ))
}

/// The outcomes of a left value `op ANY` the members, or `op ALL` them when
/// `is_any` is unset, from the kinds of answer the value may give against
/// the non-`NULL` members.
fn quantified_outcomes(
    rows: RowKinds,
    members: &Members,
    value_null: bool,
    value_has_values: bool,
    is_any: bool,
) -> Outcomes {
    // With no member at all, `ANY` is `FALSE` and `ALL` is `TRUE`.
    if matches!(members.empty, Presence::Always) {
        let outcomes = if is_any {
            Outcomes { truth: false, falsity: true, null: false }
        } else {
            Outcomes { truth: true, falsity: false, null: false }
        };
        return outcomes;
    }
    // A `NULL` member nulls its row's comparisons, so one in every member
    // set kills both a certain `true` and a certain `false`. An empty result
    // reads `FALSE` to `ANY` and `TRUE` to `ALL`, whatever the left side is.
    let sometimes_empty = matches!(members.empty, Presence::Sometimes);
    let pair = value_has_values && members.nonempty;
    let null_free = !matches!(members.null, Presence::Always);
    let (truth, falsity) = if is_any {
        // `ANY` over an empty set is `FALSE`, so a possibly-empty operand
        // is false on its empty rows.
        (pair && rows.may_be_true(), sometimes_empty || pair && null_free && rows.all_false)
    } else {
        (sometimes_empty || pair && null_free && rows.all_true, pair && rows.may_be_false())
    };
    // `ANY` is `NULL` when some value compares false against every value
    // and a member is `NULL`; `ALL` is `NULL` when some value compares
    // true against every value and a member is `NULL`, both against the
    // non-`NULL` members.
    let nulls = if is_any {
        rows.all_false || !members.nonempty || members.null_only
    } else {
        rows.all_true || !members.nonempty || members.null_only
    };
    let null_member = !matches!(members.null, Presence::Never);
    let null = (value_null && (members.nonempty || null_member))
        || (null_member && value_has_values && nulls);
    Outcomes { truth, falsity, null }
}

/// The members an `ANY`/`ALL` operand may hold.
struct Members {
    /// A non-`NULL` member is possible.
    nonempty: bool,
    /// Whether a `NULL` member is among the members.
    null: Presence,
    /// Whether the members are absent.
    empty: Presence,
    /// A member set may hold `NULL` members alone.
    null_only: bool,
}

/// How often a fact about an `ANY`/`ALL` operand's members holds across the
/// rows the condition is read on.
enum Presence {
    /// It holds on no row.
    Never,
    /// It may hold on a row and may not.
    Sometimes,
    /// It holds on every row.
    Always,
}

/// A subquery operand of `IN` or `ANY`/`ALL` compared with the left value.
struct SubqueryOperand {
    /// The comparison, read over any subset of the column's values, as the
    /// subquery's rows may hold any of them.
    compared: Compared,
    /// The values the subquery's first column may hold.
    column: Evaluation,
    /// The row counts the subquery may return.
    rows: Rows,
}

/// `value` compared by `op` with a subquery's first column. `Err` holds the
/// unknown condition when the subquery projects no known column.
fn subquery_comparison<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    value: Evaluation,
    op: &BinaryOperator,
    subquery: &Query,
) -> Result<Result<SubqueryOperand, Evaluation>, C::Error> {
    let qe = state.context.query(subquery, state.guarantees)?;
    let Some(column) = qe.columns.as_ref().and_then(|columns| columns.first()) else {
        let error = value.may_raise() || qe.error;
        let unknown = Values::Unlisted(Unlisted::any());
        return Ok(Err(Evaluation::new(ScalarType::Boolean, true, error, unknown)));
    };
    let mut compared = comparison_evals(state, value, op, column.clone());
    compared.rows = compared.rows.over_subsets();
    compared.error |= qe.error;
    Ok(Ok(SubqueryOperand { compared, column: column.clone(), rows: qe.rows }))
}

/// The forward evaluation of `EXISTS subquery`.
pub(super) fn exists<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    subquery: &Query,
    negated: bool,
) -> Result<Evaluation, C::Error> {
    let qe = state.context.query(subquery, state.guarantees)?;
    let o = Outcomes { truth: qe.rows.one || qe.rows.many, falsity: qe.rows.none, null: false };
    let o = if negated { support::not_outcomes(o) } else { o };
    Ok(Evaluation::condition(o, qe.error))
}

/// The forward evaluation of a scalar subquery.
pub(super) fn scalar_subquery<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    query: &Query,
) -> Result<Evaluation, C::Error> {
    let qe = state.context.query(query, state.guarantees)?;
    let Some(column) = qe.columns.as_ref().and_then(|columns| columns.first()) else {
        return Ok(Evaluation::unknown(ScalarType::Unknown).with_error(qe.error || qe.rows.many));
    };
    // PostgreSQL raises when a scalar subquery yields more than one row.
    let error = qe.error || qe.rows.many;
    let evaluation = if qe.rows.many {
        Evaluation::new(
            column.ty(),
            column.may_be_null() || qe.rows.none,
            error,
            Values::Unlisted(Unlisted::any()),
        )
    } else {
        Evaluation::new(
            column.ty(),
            column.may_be_null() || qe.rows.none,
            error,
            column.values().clone(),
        )
    };
    Ok(evaluation)
}

/// The forward evaluation of `x IS NULL`.
pub(super) fn is_null<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
    negated: bool,
) -> Result<Evaluation, C::Error> {
    let inner = forward(state, expr)?;
    Ok(null_test(&inner, negated))
}

/// The forward evaluation of `x IS UNKNOWN`, which reads as `x IS NULL`.
pub(super) fn is_unknown<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
    negated: bool,
) -> Result<Evaluation, C::Error> {
    let inner = support::as_condition(forward(state, expr)?);
    Ok(null_test(&inner, negated))
}

/// The outcomes of `inner IS [NOT] NULL`.
fn null_test(inner: &Evaluation, negated: bool) -> Evaluation {
    let o =
        Outcomes { truth: inner.may_be_null(), falsity: !inner.values().is_empty(), null: false };
    let o = if negated { support::not_outcomes(o) } else { o };
    Evaluation::condition(o, inner.may_raise())
}

/// The forward evaluation of `x IS [NOT] TRUE/FALSE`.
pub(super) fn is_true_false<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    expr: &Expr,
    is_true: bool,
    negated: bool,
) -> Result<Evaluation, C::Error> {
    let inner = support::as_condition(forward(state, expr)?);
    // A `NULL` operand reads false from `IS TRUE` and `IS FALSE`, so the
    // `NULL` possibility sits on the outcome the test rejects.
    let (truth, falsity) = if is_true {
        (inner.may_be_true(), inner.may_be_false() || inner.may_be_null())
    } else {
        (inner.may_be_false(), inner.may_be_true() || inner.may_be_null())
    };
    let o = if negated {
        Outcomes { truth: falsity, falsity: truth, null: false }
    } else {
        Outcomes { truth, falsity, null: false }
    };
    Ok(Evaluation::condition(o, inner.may_raise()))
}

/// The forward evaluation of `x IS [NOT] DISTINCT FROM y`, never `NULL`.
pub(super) fn is_distinct<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    left: &Expr,
    right: &Expr,
    negated: bool,
) -> Result<Evaluation, C::Error> {
    let left = forward(state, left)?;
    let right = forward(state, right)?;
    if left.is_impossible() || right.is_impossible() {
        return Ok(Evaluation::impossible(ScalarType::Boolean));
    }
    let (left_null, left_values) = (left.may_be_null(), !left.values().is_empty());
    let (right_null, right_values) = (right.may_be_null(), !right.values().is_empty());
    let cmp = comparison_evals(state, left, &BinaryOperator::Eq, right);
    // `IS NOT DISTINCT FROM` holds on equal rows and on the rows where
    // both sides are `NULL`; the rows with exactly one `NULL` side read
    // `IS DISTINCT FROM`.
    let one_null = (left_null && right_values) || (right_null && left_values);
    let both_null = left_null && right_null;
    let (equal, unequal) = (cmp.rows.may_be_true(), cmp.rows.may_be_false());
    let (truth, falsity) = if negated {
        (equal || both_null, unequal || one_null)
    } else {
        (unequal || one_null, equal || both_null)
    };
    Ok(Evaluation::condition(Outcomes { truth, falsity, null: false }, cmp.error))
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use core::ops::Bound;

    use sql_values::postgres::{IntegerType, PgNumeric, PgValue};
    use sqlparser::ast::{Expr, Function, FunctionArguments, ObjectName, ObjectNamePart};

    use crate::{
        context::{Builtin, Callee, CalleeKind, NullInput, Rows, Volatility},
        domain::{Evaluation, Interval, Unlisted, Values},
        scalar::{ScalarType, TextRules},
        test_context::{ColumnContext, builtin_callee, parse, read},
    };

    fn int(value: i64) -> Evaluation {
        Evaluation::known(PgValue::Integer(value), ScalarType::Integer(IntegerType::Integer))
    }

    fn num(value: &str) -> Evaluation {
        Evaluation::known(
            PgValue::Numeric(PgNumeric::parse(value).expect("a numeric literal")),
            ScalarType::Numeric,
        )
    }

    fn text(value: &str) -> Evaluation {
        Evaluation::known(PgValue::Text(value.to_string()), ScalarType::Text(TextRules::DEFAULT))
    }

    fn byte_text(value: &str) -> Evaluation {
        Evaluation::known(PgValue::Text(value.to_string()), ScalarType::Text(TextRules::BYTEWISE))
    }

    fn text_len(low: Bound<u64>, high: Bound<u64>) -> Evaluation {
        Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Unlisted(Unlisted {
                excluded: BTreeSet::new(),
                numbers: Interval::unbounded(),
                length: Interval { low, high },
            }),
        )
    }

    fn mixed_untyped() -> Evaluation {
        Evaluation::new(
            ScalarType::UnknownLiteral,
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Text("ab".to_string())])),
        )
    }

    fn int_col() -> Evaluation {
        Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        )
    }

    fn callee(
        builtin: Option<Builtin>,
        null_input: NullInput,
        kind: CalleeKind,
        returns: ScalarType,
    ) -> Callee {
        Callee {
            builtin,
            null_input,
            volatility: Volatility::Immutable,
            kind,
            returns,
            returns_set: false,
            token: 0,
        }
    }

    fn int4() -> ScalarType {
        ScalarType::Integer(IntegerType::Integer)
    }

    fn int8() -> ScalarType {
        ScalarType::Integer(IntegerType::Bigint)
    }

    fn default_text() -> ScalarType {
        ScalarType::Text(TextRules::DEFAULT)
    }

    /// `count` counts from zero and is never `NULL`.
    #[test]
    fn count_counts_from_zero_and_is_never_null() {
        let folded = super::fold_builtin(Builtin::Count, &[], int8()).expect("the builtin folds");
        assert!(!folded.may_be_null());
        assert!(!folded.may_raise());
        assert!(folded.may_be(&PgValue::Integer(0)));
    }

    /// `length` counts characters and `octet_length` counts bytes.
    #[test]
    fn length_counts_characters_and_octet_length_bytes() {
        let characters =
            super::fold_length(&[byte_text("héllo")], int4(), false).expect("the builtin folds");
        assert_eq!(characters.constant(), Some(&PgValue::Integer(5)));
        let bytes =
            super::fold_length(&[byte_text("héllo")], int4(), true).expect("the builtin folds");
        assert_eq!(bytes.constant(), Some(&PgValue::Integer(6)));
    }

    /// `length` of a non-text argument stays unknown.
    #[test]
    fn length_of_a_non_text_argument_is_unknown() {
        assert!(super::fold_length(&[int(5)], int4(), false).is_none());
    }

    /// `lower` and `upper` fold a byte-order constant.
    #[test]
    fn lower_and_upper_fold_a_byte_order_constant() {
        let lower =
            super::fold_case(Builtin::Lower, &[byte_text("aBc")]).expect("the builtin folds");
        assert_eq!(lower.constant(), Some(&PgValue::Text("abc".to_string())));
        let upper =
            super::fold_case(Builtin::Upper, &[byte_text("aBc")]).expect("the builtin folds");
        assert_eq!(upper.constant(), Some(&PgValue::Text("ABC".to_string())));
    }

    /// `lower` of a non-text, default-collation, or non-constant argument is
    /// `None`.
    #[test]
    fn lower_of_a_non_foldable_argument_is_none() {
        assert!(super::fold_case(Builtin::Lower, &[int(5)]).is_none());
        assert!(super::fold_case(Builtin::Lower, &[text("aBc")]).is_none());
        assert!(
            super::fold_case(Builtin::Lower, &[text_len(Bound::Unbounded, Bound::Unbounded)])
                .is_none()
        );
    }

    /// `btrim` trims the default or a named set from both ends.
    #[test]
    fn btrim_trims_the_default_or_named_set() {
        let default = super::fold_btrim(&[text("  hi  ")]).expect("the builtin folds");
        assert_eq!(default.constant(), Some(&PgValue::Text("hi".to_string())));
        let named = super::fold_btrim(&[text("xxhixx"), text("x")]).expect("the builtin folds");
        assert_eq!(named.constant(), Some(&PgValue::Text("hi".to_string())));
    }

    /// `btrim` of a non-constant argument is `None`.
    #[test]
    fn btrim_of_a_non_constant_argument_is_none() {
        assert!(super::fold_btrim(&[text_len(Bound::Unbounded, Bound::Unbounded)]).is_none());
        assert!(
            super::fold_btrim(&[text("hi"), text_len(Bound::Unbounded, Bound::Unbounded)])
                .is_none()
        );
    }

    /// `abs` of a listed value, taking the absolute value.
    #[test]
    fn abs_of_a_listed_value_is_its_absolute_value() {
        let negative = super::fold_abs(&[int(-5)], int8()).expect("the builtin folds");
        assert_eq!(negative.constant(), Some(&PgValue::Integer(5)));
        let positive = super::fold_abs(&[int(5)], int8()).expect("the builtin folds");
        assert_eq!(positive.constant(), Some(&PgValue::Integer(5)));
        let numeric =
            super::fold_abs(&[num("-1.5")], ScalarType::Numeric).expect("the builtin folds");
        assert_eq!(
            numeric.constant(),
            Some(&PgValue::Numeric(PgNumeric::parse("1.5").expect("a number")))
        );
    }

    /// `abs` of a type's minimum integer raises.
    #[test]
    fn abs_of_the_minimum_raises() {
        let minimum =
            Evaluation::known(PgValue::Integer(i64::MIN), ScalarType::Integer(IntegerType::Bigint));
        let folded = super::fold_abs(&[minimum], int8()).expect("the builtin folds");
        assert!(folded.always_raises());
        // The `int4` minimum raises as well
        let minimum = Evaluation::known(
            PgValue::Integer(i64::from(i32::MIN)),
            ScalarType::Integer(IntegerType::Integer),
        );
        let folded = super::fold_abs(&[minimum], int4()).expect("the builtin folds");
        assert!(folded.always_raises());
        // So does the `smallint` minimum
        let minimum = Evaluation::known(
            PgValue::Integer(i64::from(i16::MIN)),
            ScalarType::Integer(IntegerType::Smallint),
        );
        let folded = super::fold_abs(&[minimum], int4()).expect("the builtin folds");
        assert!(folded.always_raises());
    }

    /// `abs` of a non-numeric constant is `None`.
    #[test]
    fn abs_of_a_non_numeric_value_is_none() {
        assert!(super::fold_abs(&[text("x")], ScalarType::Numeric).is_none());
    }

    /// `concat` joins the constant arguments.
    #[test]
    fn concat_joins_the_constant_arguments() {
        let joined = super::fold_concat(&[text("a"), text("b")]).expect("the builtin folds");
        assert_eq!(joined.constant(), Some(&PgValue::Text("ab".to_string())));
    }

    /// `concat_ws` joins the constant parts with the separator.
    #[test]
    fn concat_ws_joins_the_constant_parts() {
        let joined = super::fold_concat_ws(&[text(","), text("a"), text("b")], default_text())
            .expect("the builtin folds");
        assert_eq!(joined.constant(), Some(&PgValue::Text("a,b".to_string())));
        // An empty first part still takes the separator after it.
        let joined = super::fold_concat_ws(&[text(","), text(""), text("a")], default_text())
            .expect("the builtin folds");
        assert_eq!(joined.constant(), Some(&PgValue::Text(",a".to_string())));
    }

    /// `concat_ws` with a `NULL` separator is `NULL`, skipping `NULL` parts.
    #[test]
    fn concat_ws_with_a_null_separator_or_part() {
        let result =
            super::fold_concat_ws(&[Evaluation::null(default_text()), text("a")], default_text())
                .expect("the builtin folds");
        assert!(result.is_always_null());
        let result = super::fold_concat_ws(
            &[text(","), text("a"), Evaluation::null(default_text())],
            default_text(),
        )
        .expect("the builtin folds");
        assert_eq!(result.constant(), Some(&PgValue::Text("a".to_string())));
    }

    /// `concat_ws` with a non-constant separator is `None`.
    #[test]
    fn concat_ws_with_a_non_constant_separator_is_none() {
        assert!(
            super::fold_concat_ws(
                &[text_len(Bound::Unbounded, Bound::Unbounded), text("a")],
                default_text(),
            )
            .is_none()
        );
    }

    /// `argument_type` needs exactly one argument.
    #[test]
    fn argument_type_needs_exactly_one_argument() {
        assert!(super::argument_type(&[]).is_none());
        assert!(super::argument_type(&[int(1), int(2)]).is_none());
        assert_eq!(super::argument_type(&[text("x")]), Some(default_text()));
    }

    /// `length` of a column bounds the count by the column's length interval.
    #[test]
    fn length_of_a_column_bounds_the_count() {
        let bounded = super::length_interval(
            &text_len(Bound::Included(2), Bound::Included(5)),
            int4(),
            false,
        )
        .expect("the interval folds");
        assert!(bounded.may_be_null());
        assert!(bounded.may_be(&PgValue::Integer(3)));
        assert!(!bounded.may_be(&PgValue::Integer(6)));
    }

    /// `length` handles unbounded, excluded and open-top length bounds.
    #[test]
    fn length_handles_the_various_length_bounds() {
        assert!(super::length_interval(&mixed_untyped(), int4(), false).is_some());
        // A set holding no text bounds no length.
        let numbers = Evaluation::known(PgValue::Integer(1), ScalarType::UnknownLiteral);
        assert!(super::length_interval(&numbers, int4(), false).is_some());
        assert!(
            super::length_interval(&text_len(Bound::Excluded(5), Bound::Unbounded), int4(), false)
                .is_some()
        );
        assert!(
            super::length_interval(&text_len(Bound::Included(0), Bound::Unbounded), int4(), true)
                .is_some()
        );
        assert!(
            super::length_interval(
                &text_len(Bound::Included(0), Bound::Excluded(10)),
                int4(),
                false
            )
            .is_some()
        );
    }

    /// `call_arguments` gives `None` for wildcard arguments.
    #[test]
    fn call_arguments_gives_none_for_wildcards() {
        let Expr::Function(function) = &parse("count(*)") else { panic!("expected a function") };
        let arguments = super::support::call_arguments(function);
        assert!(arguments.len() == 1 && arguments[0].is_none());
    }

    /// `call_arguments` is empty when the call has no argument list.
    #[test]
    fn call_arguments_is_empty_without_an_argument_list() {
        let function = Function {
            name: ObjectName(vec![ObjectNamePart::Identifier(sqlparser::ast::Ident::new("f"))]),
            uses_odbc_syntax: false,
            parameters: FunctionArguments::None,
            args: FunctionArguments::None,
            within_group: Vec::new(),
            filter: None,
            null_treatment: None,
            over: None,
        };
        assert_eq!(crate::support::call_arguments(&function), Vec::new());
    }

    /// `count(*)` is an aggregate whose wildcard argument is never `NULL`.
    #[test]
    fn count_star_is_an_aggregate_and_never_null() {
        let context = ColumnContext::new().with_callee(
            "count",
            callee(Some(Builtin::Count), NullInput::Strict, CalleeKind::Aggregate, int8()),
        );
        let evaluation = read(context, "count(*)");
        assert!(!evaluation.may_be_null());
        assert_eq!(evaluation.ty(), int8());
    }

    /// `count` of a column counts from zero.
    #[test]
    fn count_of_a_column_counts_from_zero() {
        let context = ColumnContext::new()
            .with_column("id", int_col())
            .with_callee("count", builtin_callee(Builtin::Count, int8()));
        let evaluation = read(context, "count(id)");
        assert!(!evaluation.may_be_null());
        assert!(evaluation.may_be(&PgValue::Integer(0)));
    }

    /// A `NULL` at a called-on-null position forces the call to `NULL`.
    #[test]
    fn called_on_null_forces_null_for_a_null_position() {
        let context = ColumnContext::new().with_callee(
            "foo",
            callee(None, NullInput::CalledOnNull(vec![0]), CalleeKind::Normal, int4()),
        );
        let evaluation = read(context, "foo(NULL)");
        assert!(evaluation.is_always_null());
    }

    /// A strict call over a `NULL` argument is nullable but not forced `NULL`.
    #[test]
    fn a_strict_call_over_a_nullable_argument_is_nullable() {
        let context = ColumnContext::new()
            .with_column("id", int_col())
            .with_callee("abs", builtin_callee(Builtin::Abs, ScalarType::Numeric));
        let evaluation = read(context, "abs(id)");
        assert!(evaluation.may_be_null());
        assert!(!evaluation.is_always_null());
    }

    /// A callee body refines the call's reading.
    #[test]
    fn a_callee_body_refines_the_call() {
        let context = ColumnContext::new()
            .with_column("id", int_col())
            .with_callee("foo", callee(None, NullInput::Strict, CalleeKind::Normal, int4()))
            .with_body(0, int(7));
        let evaluation = read(context, "foo(id)");
        assert!(evaluation.may_be(&PgValue::Integer(7)));
    }

    /// The `length` and `octet_length` folds reach their builtin arms.
    #[test]
    fn length_and_octet_length_calls_fold() {
        let context =
            ColumnContext::new().with_callee("length", builtin_callee(Builtin::Length, int4()));
        assert_eq!(read(context, "length('ab')").constant(), Some(&PgValue::Integer(2)));
        let context = ColumnContext::new()
            .with_callee("octet_length", builtin_callee(Builtin::OctetLength, int4()));
        assert_eq!(read(context, "octet_length('héllo')").constant(), Some(&PgValue::Integer(6)));
    }

    /// The `lower`, `upper`, `btrim`, `abs`, `concat` and `concat_ws` arms.
    #[test]
    fn the_builtin_calls_reach_their_fold_arms() {
        let context = ColumnContext::new()
            .with_callee("lower", builtin_callee(Builtin::Lower, default_text()));
        assert_eq!(read(context, "lower('abc')").ty(), default_text());
        let context = ColumnContext::new()
            .with_callee("upper", builtin_callee(Builtin::Upper, default_text()));
        assert_eq!(read(context, "upper('abc')").ty(), default_text());
        let context = ColumnContext::new()
            .with_callee("btrim", builtin_callee(Builtin::Btrim, default_text()));
        assert_eq!(
            read(context, "btrim('  hi  ')").constant(),
            Some(&PgValue::Text("hi".to_string()))
        );
        let context = ColumnContext::new()
            .with_callee("abs", builtin_callee(Builtin::Abs, ScalarType::Numeric));
        assert_eq!(read(context, "abs(-5)").constant(), Some(&PgValue::Integer(5)));
        let context = ColumnContext::new()
            .with_callee("abs", builtin_callee(Builtin::Abs, ScalarType::Numeric));
        assert!(read(context, "abs((-32768)::smallint)").always_raises());
        let context = ColumnContext::new()
            .with_callee("concat", builtin_callee(Builtin::Concat, default_text()));
        assert_eq!(
            read(context, "concat('a', 'b')").constant(),
            Some(&PgValue::Text("ab".to_string()))
        );
        let context = ColumnContext::new()
            .with_callee("concat_ws", builtin_callee(Builtin::ConcatWs, default_text()));
        assert_eq!(
            read(context, "concat_ws(',', 'a', 'b')").constant(),
            Some(&PgValue::Text("a,b".to_string()))
        );
    }

    /// A decisive branch stops the `CASE` search.
    #[test]
    fn a_decisive_case_branch_stops_the_search() {
        let decisive = read(ColumnContext::new(), "CASE WHEN true THEN 2 ELSE 3 END");
        assert_eq!(decisive.constant(), Some(&PgValue::Integer(2)));
        let skipped = read(ColumnContext::new(), "CASE WHEN true THEN 2 WHEN false THEN 3 END");
        assert_eq!(skipped.constant(), Some(&PgValue::Integer(2)));
    }

    /// The `CASE` else reads when the end is reachable, `NULL` without it.
    #[test]
    fn a_case_reads_the_else_when_the_end_is_reachable() {
        let context = ColumnContext::new().with_column("id", int_col());
        let with_else = read(context, "CASE WHEN id = 1 THEN 2 ELSE 3 END");
        assert!(with_else.may_be(&PgValue::Integer(2)));
        assert!(with_else.may_be(&PgValue::Integer(3)));
        let context = ColumnContext::new().with_column("id", int_col());
        let no_else = read(context, "CASE WHEN id = 1 THEN 2 END");
        assert!(no_else.may_be_null());
    }

    /// Non-decisive branches keep every branch result open.
    #[test]
    fn non_decisive_case_branches_keep_their_results() {
        let context = ColumnContext::new().with_column("id", int_col());
        let evaluation = read(context, "CASE WHEN id = 1 THEN 2 WHEN id = 2 THEN 3 END");
        assert!(evaluation.may_be(&PgValue::Integer(2)));
        assert!(evaluation.may_be(&PgValue::Integer(3)));
        assert!(evaluation.may_be_null());
    }

    /// A simple `CASE` matches the operand.
    #[test]
    fn a_simple_case_matches_the_operand() {
        let context = ColumnContext::new().with_column("id", int_col());
        let evaluation = read(context, "CASE id WHEN 1 THEN 2 END");
        assert!(evaluation.may_be(&PgValue::Integer(2)));
    }

    /// A `CASE` of untyped `NULL`s alone resolves to `text`, which `IS
    /// UNKNOWN` refuses.
    #[test]
    fn a_case_of_untyped_nulls_resolves_to_text() {
        let evaluation = read(ColumnContext::new(), "CASE WHEN true THEN NULL END");
        assert_eq!(evaluation.ty(), default_text());
        let tested =
            read(ColumnContext::new(), "(CASE WHEN true THEN NULL ELSE NULL END) IS UNKNOWN");
        assert!(tested.may_raise());
    }

    /// `ANY` keeps the raise of an operand it cannot list, and an operand of
    /// a scalar type is not the array it needs.
    #[test]
    fn any_keeps_the_raise_of_an_unlisted_operand() {
        let context =
            ColumnContext::new().with_column("t", text_len(Bound::Unbounded, Bound::Unbounded));
        assert!(read(context, "5 = ANY (CAST(t AS INT[]))").may_raise());
        let context = ColumnContext::new().with_column("i", int_col());
        assert!(read(context, "5 > ANY (i)").may_raise());
    }

    /// A simple `CASE` reads an untyped operand as `text`, which a number does
    /// not compare with.
    #[test]
    fn a_simple_case_reads_an_untyped_operand_as_text() {
        assert!(read(ColumnContext::new(), "CASE '1' WHEN 0 THEN 1 END").may_raise());
        assert!(read(ColumnContext::new(), "CASE NULL WHEN -1 THEN 1 END").may_raise());
        let matched = read(ColumnContext::new(), "CASE '1' WHEN '1' THEN 1 END");
        assert_eq!(matched.constant(), Some(&PgValue::Integer(1)));
        assert!(!matched.may_raise());
    }

    /// Parentheses around the array of an `ANY` leave its members read.
    #[test]
    fn any_reads_a_parenthesised_array() {
        assert!(read(ColumnContext::new(), "5 <> ALL ((ARRAY[NULL]))").may_raise());
        let evaluation = read(ColumnContext::new(), "5 = ANY ((ARRAY[1, 2]))");
        assert!(!evaluation.may_be_true() && evaluation.may_be_false());
    }

    /// `IN` over a list reads the members, including a `NULL`.
    #[test]
    fn in_list_reads_the_members() {
        assert!(
            read(ColumnContext::new().with_column("id", int_col()), "id IN (1, 2)").may_be_true()
        );
        assert!(
            read(ColumnContext::new().with_column("id", int_col()), "id IN (1, NULL)")
                .may_be_true()
        );
        assert!(
            read(ColumnContext::new().with_column("id", int_col()), "id NOT IN (1, 2)")
                .may_be_true()
        );
    }

    /// `IN` over a subquery reads the query's columns.
    #[test]
    fn in_subquery_reads_the_query() {
        let context =
            ColumnContext::new().with_column("id", int_col()).with_query(Rows::ONE, int_col());
        assert!(read(context, "id IN (SELECT id FROM t)").may_be_true());
        let context = ColumnContext::new().with_column("id", int_col());
        assert!(read(context, "id IN (SELECT id FROM t)").may_be_null());
    }

    /// `= ANY` over a row constructor reads the members and the raise.
    #[test]
    fn any_over_a_tuple_reads_the_members_and_the_raise() {
        let evaluation = read(ColumnContext::new(), "1 = ANY ((1, 2))");
        assert_eq!(evaluation.constant_value(), Some(&PgValue::Boolean(true)));
        assert!(evaluation.may_raise());
    }

    /// `IN UNNEST` stays unknown.
    #[test]
    fn in_unnest_is_unknown() {
        let context = ColumnContext::new().with_column("id", int_col());
        let evaluation = read(context, "id IN UNNEST(ARRAY[1, 2])");
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
    }

    /// `IN` of an impossible value is impossible.
    #[test]
    fn in_of_an_impossible_value_is_impossible() {
        let context =
            ColumnContext::new().with_column("x", Evaluation::impossible(ScalarType::Boolean));
        let evaluation = read(context, "x IN (1, 2)");
        assert!(evaluation.is_impossible());
    }

    /// `LIKE` stays unknown with `NULL` propagation.
    #[test]
    fn like_stays_unknown_with_null_propagation() {
        let context =
            ColumnContext::new().with_column("name", text_len(Bound::Unbounded, Bound::Unbounded));
        let nullable = read(context, "name LIKE 'a%'");
        assert!(nullable.may_be_null());
        assert!(nullable.may_be_true());
        let constant = read(ColumnContext::new(), "'abc' LIKE 'a%'");
        assert!(!constant.may_be_null());
    }

    /// A pattern ending with its escape character, an escape of more than one
    /// character and a malformed `SIMILAR TO` pattern may raise, and a `NULL`
    /// escape reads `NULL`.
    #[test]
    fn a_like_pattern_and_escape_may_raise() {
        let context = || {
            ColumnContext::new().with_column("name", text_len(Bound::Unbounded, Bound::Unbounded))
        };
        assert!(read(context(), "name LIKE 'a\\'").may_raise());
        assert!(read(context(), "name ILIKE 'a#' ESCAPE '#'").may_raise());
        assert!(read(context(), "name LIKE 'a' ESCAPE 'ab'").may_raise());
        assert!(read(context(), "name SIMILAR TO '('").may_raise());
        assert!(!read(context(), "name LIKE 'a\\%'").may_raise());
        assert!(!read(context(), "name LIKE 'a#' ESCAPE ''").may_raise());
        assert!(read(ColumnContext::new(), "'x' LIKE 'a' ESCAPE NULL").may_be_null());
        // A pattern or an escape the evaluator does not know may raise.
        let columns = || context().with_column("p", text_len(Bound::Unbounded, Bound::Unbounded));
        assert!(read(columns(), "name LIKE p").may_raise());
        assert!(read(columns(), "name LIKE 'a' ESCAPE p").may_raise());
    }

    /// `LIKE` of an impossible value is impossible.
    #[test]
    fn like_of_an_impossible_value_is_impossible() {
        let context =
            ColumnContext::new().with_column("x", Evaluation::impossible(ScalarType::Boolean));
        let evaluation = read(context, "x LIKE 'a%'");
        assert!(evaluation.is_impossible());
    }

    /// `ANY` and `ALL` over a list.
    #[test]
    fn any_and_all_over_a_list() {
        assert!(
            read(ColumnContext::new().with_column("id", int_col()), "id = ANY(ARRAY[1, 2])")
                .may_be_true()
        );
        assert!(
            read(ColumnContext::new().with_column("id", int_col()), "id = ALL(ARRAY[1, 2])")
                .may_be_false()
        );
    }

    /// `= ANY` over a fractional member keeps the member, so an integer may
    /// differ from it.
    #[test]
    fn any_over_a_fractional_member_may_be_false_for_an_integer() {
        let context = ColumnContext::new().with_column("id", int_col());
        let evaluation = read(context, "id = ANY(ARRAY[1.5])");
        assert!(evaluation.may_be_false());
    }

    /// `ANY` with a `NULL` member is `NULL` on its true rows.
    #[test]
    fn any_with_a_null_member_is_null_on_true() {
        let context = ColumnContext::new().with_column("id", int_col());
        let evaluation = read(context, "id = ANY(ARRAY[1, NULL])");
        assert!(evaluation.may_be_null());
    }

    /// An empty `ANY` reads false and an empty `ALL` reads true.
    #[test]
    fn empty_any_reads_false_and_empty_all_reads_true() {
        let context =
            ColumnContext::new().with_column("id", int_col()).with_query(Rows::NONE, int_col());
        let any = read(context, "id = ANY(SELECT id FROM t)");
        assert!(!any.may_be_true());
        assert!(any.may_be_false());
        assert!(!any.may_be_null());
        let context =
            ColumnContext::new().with_column("id", int_col()).with_query(Rows::NONE, int_col());
        let all = read(context, "id = ALL(SELECT id FROM t)");
        assert!(!all.may_be_false());
        assert!(all.may_be_true());
        assert!(!all.may_be_null());
    }

    /// `ANY` over a non-constant member stays unknown.
    #[test]
    fn any_over_a_non_constant_member_is_unknown() {
        let context =
            ColumnContext::new().with_column("id", int_col()).with_column("other", int_col());
        let evaluation = read(context, "id = ANY(ARRAY[other])");
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
    }

    /// `ANY` over a subquery reads the query's columns and emptiness.
    #[test]
    fn any_over_a_subquery_reads_the_query() {
        let context = ColumnContext::new()
            .with_column("id", int_col())
            .with_query(Rows { none: true, one: true, many: false }, int_col());
        assert!(read(context, "id = ANY(SELECT id FROM t)").may_be_true());
        let context =
            ColumnContext::new().with_column("id", int_col()).with_query(Rows::ONE, int_col());
        assert!(read(context, "id = ANY(SELECT id FROM t)").may_be_true());
        let context = ColumnContext::new().with_column("id", int_col());
        assert!(read(context, "id = ANY(SELECT id FROM t)").may_be_null());
    }

    /// A subquery's rows may hold any of the values its column may hold, so
    /// an `ANY` row may fail every member and an `ALL` row may satisfy every
    /// member.
    #[test]
    fn a_subquery_row_may_hold_any_of_its_values() {
        let id = Evaluation::known(PgValue::Integer(1), int4());
        let column = Evaluation::new(
            int4(),
            false,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)])),
        );
        let rows = Rows { none: false, one: true, many: true };
        let context =
            || ColumnContext::new().with_column("id", id.clone()).with_query(rows, column.clone());
        assert!(read(context(), "id = ANY(SELECT id FROM t)").may_be_false());
        assert!(read(context(), "id = ALL(SELECT id FROM t)").may_be_true());
    }

    /// A subquery column that may be `NULL` need not hold a `NULL` row, so an
    /// `ANY` may still fail every row and an `ALL` may still satisfy every row.
    #[test]
    fn a_nullable_subquery_column_need_not_hold_a_null_row() {
        let id = Evaluation::known(PgValue::Integer(3), int4());
        let column = Evaluation::new(
            int4(),
            true,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)])),
        );
        let rows = Rows { none: false, one: true, many: true };
        let context =
            || ColumnContext::new().with_column("id", id.clone()).with_query(rows, column.clone());
        assert!(read(context(), "id = ANY(SELECT id FROM t)").may_be_false());
        assert!(read(context(), "id <> ALL(SELECT id FROM t)").may_be_true());
        // A column holding only `NULL` reads `NULL` on every row set.
        let nulls = Evaluation::null(int4());
        let context = ColumnContext::new().with_column("id", id.clone()).with_query(rows, nulls);
        assert!(read(context, "id = ANY(SELECT id FROM t)").is_always_null());
        // An array's `NULL` element is in every member set.
        let context = || ColumnContext::new().with_column("id", id.clone());
        assert!(!read(context(), "id = ANY(ARRAY[1, NULL])").may_be_false());
        assert!(!read(context(), "id <> ALL(ARRAY[1, NULL])").may_be_true());
    }

    /// A subquery column that may be `NULL` may hold only `NULL` rows, which
    /// read `NULL` whatever the members a value matches.
    #[test]
    fn a_nullable_subquery_column_may_hold_only_null_rows() {
        let id = Evaluation::known(PgValue::Integer(1), int4());
        let column = Evaluation::new(
            int4(),
            true,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)])),
        );
        let rows = Rows { none: false, one: true, many: true };
        let context =
            || ColumnContext::new().with_column("id", id.clone()).with_query(rows, column.clone());
        assert!(read(context(), "id = ANY(SELECT id FROM t)").may_be_null());
        assert!(read(context(), "id <> ALL(SELECT id FROM t)").may_be_null());
        assert!(read(context(), "id < ANY(SELECT id FROM t)").may_be_null());
    }

    /// `ANY` of an impossible value is impossible.
    #[test]
    fn any_of_an_impossible_value_is_impossible() {
        let context =
            ColumnContext::new().with_column("x", Evaluation::impossible(ScalarType::Boolean));
        let evaluation = read(context, "x = ANY(ARRAY[1])");
        assert!(evaluation.is_impossible());
    }

    /// `EXISTS` reads the subquery's row count.
    #[test]
    fn exists_reads_the_row_count() {
        assert!(
            read(ColumnContext::new().with_query(Rows::ONE, int_col()), "EXISTS (SELECT 1)")
                .may_be_true()
        );
        assert!(
            !read(ColumnContext::new().with_query(Rows::ONE, int_col()), "EXISTS (SELECT 1)")
                .may_be_false()
        );
        let context = ColumnContext::new().with_query(Rows::NONE, int_col());
        assert!(read(context, "EXISTS (SELECT 1)").may_be_false());
        let context = ColumnContext::new().with_query(Rows::NONE, int_col());
        assert!(read(context, "NOT EXISTS (SELECT 1)").may_be_true());
        let context = ColumnContext::new().with_query(Rows::ANY, int_col());
        let unknown = read(context, "EXISTS (SELECT 1)");
        assert!(unknown.may_be_true());
        assert!(unknown.may_be_false());
    }

    /// A scalar subquery reads the query's column and row count.
    #[test]
    fn a_scalar_subquery_reads_the_query() {
        let context = ColumnContext::new();
        assert_eq!(read(context, "(SELECT 1)").ty(), ScalarType::Unknown);
        let context = ColumnContext::new()
            .with_query(Rows { none: false, one: false, many: true }, int_col());
        assert!(read(context, "(SELECT 1)").may_raise());
        let context = ColumnContext::new().with_query(Rows::ONE, int_col());
        assert!(matches!(read(context, "(SELECT 1)").values(), Values::Unlisted(_)));
    }

    /// `IS UNKNOWN` reads as `IS NULL`.
    #[test]
    fn is_unknown_reads_as_is_null() {
        let context = ColumnContext::new().with_column("id", int_col());
        let evaluation = read(context, "id IS UNKNOWN");
        assert!(evaluation.may_be_true());
        assert!(evaluation.may_be_false());
    }

    /// `IS DISTINCT FROM` is never `NULL`.
    #[test]
    fn is_distinct_from_is_never_null() {
        assert!(
            !read(ColumnContext::new().with_column("id", int_col()), "id IS DISTINCT FROM 1")
                .may_be_null()
        );
        assert!(
            !read(ColumnContext::new().with_column("id", int_col()), "id IS NOT DISTINCT FROM 1")
                .may_be_null()
        );
    }

    /// `IS DISTINCT FROM` of an impossible side is impossible.
    #[test]
    fn is_distinct_from_an_impossible_side_is_impossible() {
        let context =
            ColumnContext::new().with_column("x", Evaluation::impossible(ScalarType::Boolean));
        let evaluation = read(context, "x IS DISTINCT FROM 1");
        assert!(evaluation.is_impossible());
    }

    /// The operand comparison of a `CASE` carries its type refusal.
    #[test]
    fn a_case_operand_comparison_refusal_raises() {
        let context = ColumnContext::new().with_column(
            "d",
            Evaluation::new(
                ScalarType::Date,
                true,
                false,
                Values::Unlisted(crate::domain::Unlisted::any()),
            ),
        );
        assert!(read(context, "CASE d WHEN 1 THEN 2 END").may_raise());
        // A branch no row reaches still refuses its types.
        assert!(read(ColumnContext::new(), "CASE WHEN false THEN true + 1 ELSE 1 END").may_raise());
    }

    /// `abs` over an integer range holding the type's minimum may raise.
    #[test]
    fn abs_over_a_range_holding_the_minimum_may_raise() {
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let small = Evaluation::new(
            int4,
            true,
            false,
            Values::Unlisted(Unlisted {
                numbers: Interval {
                    low: Bound::Included(PgNumeric::from(0)),
                    high: Bound::Included(PgNumeric::from(100)),
                },
                ..Unlisted::any()
            }),
        );
        let context = || {
            ColumnContext::new()
                .with_column("x", int_col())
                .with_column("y", small.clone())
                .with_callee("abs", builtin_callee(Builtin::Abs, int4))
        };
        assert!(read(context(), "abs(x)").may_raise());
        assert!(!read(context(), "abs(y)").may_raise());
        // An untyped literal is read through a number type's input.
        assert!(read(context(), "abs('abc')").may_raise());
    }

    /// Branch, member and list values are read as the type they join to.
    #[test]
    fn joined_values_take_the_joined_type() {
        let read = |sql: &str| read(ColumnContext::new().with_column("x", int_col()), sql);
        let literal = read("CASE WHEN false THEN -1 ELSE '1' END");
        assert!(literal.may_be(&PgValue::Integer(1)));
        assert!(read("CASE WHEN false THEN '' ELSE 0.0 END").may_raise());
        let float = read("CASE WHEN true THEN 2147483647 ELSE 1.5::real END");
        assert_eq!(float.ty(), ScalarType::Float);
        assert!(float.constant_value().is_none());
        assert!(read("x = ANY (ARRAY['1'])").may_raise());
        assert!(read("'a' = ANY (ARRAY[x])").may_raise());
        assert!(read("'1' IN ('', 1)").may_raise());
        assert!(!read("'1' IN ('2', 1)").may_raise());
        // Branches of untyped literals alone join to `text`.
        let text = read("CASE WHEN x = 1 THEN 'a' ELSE 'b' END");
        assert_eq!(text.ty(), ScalarType::Text(crate::scalar::TextRules::DEFAULT));
        assert!(text.may_be(&PgValue::Text("a".to_string())));
    }

    /// A call no known function answers may raise.
    #[test]
    fn a_call_without_a_known_callee_may_raise() {
        assert!(read(ColumnContext::new(), "f(1)").may_raise());
    }

    /// The arguments' raise and the strictness null wrap the body's reading.
    #[test]
    fn a_body_refinement_keeps_the_wrapper_outcomes() {
        let context = ColumnContext::new()
            .with_callee("f", callee(None, NullInput::Strict, CalleeKind::Normal, int4()))
            .with_body(0, Evaluation::known(PgValue::Integer(1), int4()));
        assert!(read(context, "f(1 / 0)").may_raise());
        let context = ColumnContext::new()
            .with_callee("g", callee(None, NullInput::Strict, CalleeKind::Normal, int4()))
            .with_body(0, Evaluation::raising(int4()));
        assert!(read(context, "g(5)").may_raise());
        let context = ColumnContext::new()
            .with_column("id", int_col())
            .with_callee("foo", callee(None, NullInput::Strict, CalleeKind::Normal, int4()))
            .with_body(0, Evaluation::known(PgValue::Integer(7), int4()));
        let evaluation = read(context, "foo(id)");
        assert!(evaluation.may_be(&PgValue::Integer(7)));
        assert!(evaluation.may_be_null());
    }

    /// A folded builtin keeps the error its arguments may raise.
    #[test]
    fn a_folded_builtin_keeps_its_arguments_error() {
        let context = ColumnContext::new().with_callee(
            "length",
            callee(Some(Builtin::Length), NullInput::Strict, CalleeKind::Normal, int4()),
        );
        assert!(read(context, "length('a' || 1 / 0)").may_raise());
    }

    /// `IS [NOT] UNKNOWN` reads its operand as a condition.
    #[test]
    fn is_unknown_reads_its_operand_as_a_condition() {
        let context = || ColumnContext::new().with_column("x", int_col());
        assert!(read(context(), "x IS UNKNOWN").may_raise());
        assert!(read(context(), "x IS NOT UNKNOWN").may_raise());
        assert!(read(ColumnContext::new(), "'a' IS UNKNOWN").may_raise());
        assert!(!read(ColumnContext::new(), "'t' IS NOT UNKNOWN").may_raise());
    }

    /// Every `ANY` member is read at the joined type, also past a member that
    /// is not a constant.
    #[test]
    fn every_any_member_is_read_at_the_joined_type() {
        let context = ColumnContext::new().with_column("x", int_col());
        assert!(read(context, "1 = ANY (ARRAY[x, 'a'])").may_raise());
    }

    /// `IS TRUE` and a `CASE` condition read an untyped literal as `boolean`.
    #[test]
    fn an_untyped_condition_reads_as_boolean() {
        assert!(read(ColumnContext::new(), "'t' IS TRUE").may_be_true());
        let case = read(ColumnContext::new(), "CASE WHEN '1' THEN 1 END");
        assert!(case.may_be(&PgValue::Integer(1)));
    }

    /// A type an engine refuses where it reads a condition, a pattern or an
    /// array reads the raise.
    #[test]
    fn a_refused_operand_type_reads_the_raise() {
        for sql in [
            "CASE WHEN true THEN 1 WHEN 1 THEN 2 END",
            "1 IS TRUE",
            "1 LIKE 'a'",
            "'a' LIKE 1",
            "1 = ANY ((1, 2))",
            "CASE WHEN true THEN 1 ELSE true END",
        ] {
            assert!(read(ColumnContext::new(), sql).may_raise(), "{sql}");
        }
    }

    /// A fold reads only an argument that is never `NULL`.
    #[test]
    fn a_nullable_argument_is_not_folded() {
        let five = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Listed(BTreeSet::from([PgValue::Integer(5)])),
        );
        assert!(super::fold_abs(core::slice::from_ref(&five), int4()).is_none());
        assert!(super::fold_concat(&[text("a"), five]).is_none());
    }

    /// `CASE` and `ANY` keep every member on the type that holds them all.
    #[test]
    fn branches_and_members_join_their_types() {
        let flag = Evaluation::new(
            ScalarType::Boolean,
            true,
            false,
            Values::Unlisted(crate::domain::Unlisted::any()),
        );
        let case =
            read(ColumnContext::new().with_column("b", flag), "CASE WHEN b THEN 1 ELSE 2.5 END");
        assert!(case.may_be(&PgValue::Numeric(PgNumeric::parse("2.5").expect("a number"))));
        assert!(read(ColumnContext::new(), "1 = ANY (ARRAY[1.5])").may_be_false());
        assert!(read(ColumnContext::new(), "1 = ALL (ARRAY[1.5, 1])").may_be_false());
    }
}
