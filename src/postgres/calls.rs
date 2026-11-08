//! PostgreSQL's recognitions of calls by name, before the context is asked.

use alloc::{collections::BTreeSet, vec::Vec};

use sql_values::PgValue;

use crate::{
    domain::{Evaluation, Unlisted, Values},
    forward::fold::reinterpret_unknown,
    scalar::{ScalarType, TextRules},
    support,
};

/// `COALESCE`, `NULLIF`, `GREATEST` and `LEAST`, recognised by name under
/// PostgreSQL before the context is asked.
pub(super) fn name_recognised(
    name: &str,
    argument_evaluations: &[Evaluation],
    error: bool,
) -> Option<Evaluation> {
    if !matches!(name, "coalesce" | "nullif" | "greatest" | "least") {
        return None;
    }
    let first = argument_evaluations.first()?;
    let Some((resolved, unsure)) = resolved_type(argument_evaluations) else {
        return Some(Evaluation::raising(first.ty()));
    };
    // `NULLIF` compares through `=`, which PostgreSQL has between a `date`
    // and a `timestamp`, so a `date` left argument keeps its type.
    let ty = if name == "nullif" && first.ty() == ScalarType::Date {
        ScalarType::Date
    } else {
        resolved
    };
    // Every argument is read as the resolved type, and an untyped literal
    // its input function cannot read raises the whole call.
    let coerced: Vec<Evaluation>;
    let arguments = if argument_evaluations.iter().any(|argument| converts(argument, ty)) {
        coerced = argument_evaluations.iter().map(|argument| coerce(argument, ty)).collect();
        let unreadable = argument_evaluations.iter().zip(&coerced).any(|(argument, coerced)| {
            argument.ty() == ScalarType::UnknownLiteral && coerced.always_raises()
        });
        if unreadable {
            return Some(Evaluation::raising(ty));
        }
        coerced.as_slice()
    } else {
        argument_evaluations
    };
    let error = error || unsure || arguments.iter().any(Evaluation::may_raise);
    match name {
        "nullif" => nullif(arguments, ty, error),
        "coalesce" => coalesce(arguments, ty, error),
        _ => Some(extremum(name == "least", arguments, ty, error)),
    }
}

/// Whether reading the argument as `ty` changes it.
fn converts(argument: &Evaluation, ty: ScalarType) -> bool {
    match argument.ty() {
        ScalarType::UnknownLiteral => ty != ScalarType::UnknownLiteral,
        ScalarType::Date => ty == ScalarType::Timestamp,
        ScalarType::Integer(_) | ScalarType::Numeric => ty == ScalarType::Float,
        _ => false,
    }
}

/// The argument read as `ty`.
fn coerce(argument: &Evaluation, ty: ScalarType) -> Evaluation {
    match argument.ty() {
        ScalarType::UnknownLiteral => reinterpret_unknown(argument.clone(), ty),
        ScalarType::Date if ty == ScalarType::Timestamp => date_as_timestamp(argument),
        // An integer always fits a `float`, a `numeric` may lie past its range.
        from @ (ScalarType::Integer(_) | ScalarType::Numeric) if ty == ScalarType::Float => {
            Evaluation::new(
                ty,
                argument.may_be_null(),
                argument.may_raise() || from == ScalarType::Numeric,
                Values::Unlisted(Unlisted::any()),
            )
        }
        _ => argument.clone(),
    }
}

/// A `date` argument read as `timestamp`, raising for a listed date past
/// the timestamp range and for an unlisted date, which may lie past it.
fn date_as_timestamp(evaluation: &Evaluation) -> Evaluation {
    let Values::Listed(listed) = evaluation.values() else {
        return evaluation.clone().with_type(ScalarType::Timestamp).with_error(true);
    };
    let mut error = evaluation.may_raise();
    let mut values = BTreeSet::new();
    for date in listed.iter().filter_map(PgValue::as_date) {
        match date.to_timestamp() {
            Ok(timestamp) => {
                values.insert(PgValue::Timestamp(timestamp));
            }
            Err(_) => error = true,
        }
    }
    Evaluation::new(ScalarType::Timestamp, evaluation.may_be_null(), error, Values::Listed(values))
}

/// The type PostgreSQL reads the arguments as and whether that reading may
/// raise, `None` for a mix of type categories it refuses.
fn resolved_type(argument_evaluations: &[Evaluation]) -> Option<(ScalarType, bool)> {
    let mut resolved: Option<ScalarType> = None;
    let mut unsure = false;
    for evaluation in argument_evaluations {
        let ty = evaluation.ty();
        // An untyped literal and an untyped `NULL` take the resolved type.
        if ty == ScalarType::UnknownLiteral
            || (ty == ScalarType::Unknown && evaluation.is_always_null())
        {
            continue;
        }
        let Some(current) = resolved else {
            resolved = Some(ty);
            continue;
        };
        let (joined, may_raise) = common_type(current, ty)?;
        resolved = Some(joined);
        unsure |= may_raise;
    }
    Some((resolved.unwrap_or(ScalarType::Text(TextRules::DEFAULT)), unsure))
}

/// The type two argument types meet at and whether reading both as it may
/// raise, `None` for two type categories PostgreSQL never matches.
fn common_type(current: ScalarType, ty: ScalarType) -> Option<(ScalarType, bool)> {
    let unmodelled = |t: ScalarType| matches!(t, ScalarType::Other | ScalarType::Unknown);
    let number = |t: ScalarType| t.is_number() || t == ScalarType::Float;
    let temporal = |t: ScalarType| matches!(t, ScalarType::Date | ScalarType::Timestamp);
    Some(match (current, ty) {
        _ if unmodelled(current) || unmodelled(ty) => (ScalarType::Other, true),
        _ if current == ty => (current, false),
        // Two implicit collations leave the comparison's collation undecided.
        (ScalarType::Text(_), ScalarType::Text(_)) => (ScalarType::Text(TextRules::UNKNOWN), true),
        (ScalarType::Integer(a), ScalarType::Integer(b)) => {
            (ScalarType::Integer(a.wider(b)), false)
        }
        _ if number(current) && number(ty) => {
            let float = current == ScalarType::Float || ty == ScalarType::Float;
            (if float { ScalarType::Float } else { ScalarType::Numeric }, false)
        }
        _ if temporal(current) && temporal(ty) => (ScalarType::Timestamp, false),
        _ => return None,
    })
}

/// `COALESCE`: the first argument not known `NULL` wins, giving its constant
/// value, or the union of the values of the arguments up to the first that
/// cannot be `NULL`, after which no argument is read. The result reads
/// `NULL` when every argument may be `NULL`.
fn coalesce(
    argument_evaluations: &[Evaluation],
    ty: ScalarType,
    error: bool,
) -> Option<Evaluation> {
    let first = argument_evaluations.iter().position(|evaluation| !evaluation.is_always_null())?;
    let evaluation = if let Some(datum) = argument_evaluations[first].constant() {
        Evaluation::known(datum.clone(), ty)
    } else {
        let reached = argument_evaluations[first..]
            .iter()
            .position(|evaluation| !evaluation.may_be_null())
            .map_or(argument_evaluations.len(), |last| first + last + 1);
        let reached = &argument_evaluations[first..reached];
        let null = reached.iter().all(Evaluation::may_be_null);
        Evaluation::new(ty, null, error, values_hull(reached.iter()))
    };
    Some(evaluation.with_error(error))
}

/// `NULLIF`: `NULL` when both constant arguments agree, the left value
/// otherwise; the left values otherwise.
fn nullif(argument_evaluations: &[Evaluation], ty: ScalarType, error: bool) -> Option<Evaluation> {
    let [left, right] = argument_evaluations else {
        return None;
    };
    let evaluation = match (left.constant(), right.constant()) {
        (Some(left_value), Some(right_value)) if left_value == right_value => Evaluation::null(ty),
        (Some(left_value), Some(_)) => Evaluation::known(left_value.clone(), ty),
        _ => {
            let null = left.may_be_null() || support::values_share(left, right);
            Evaluation::new(ty, null, error, left.values().clone())
        }
    };
    Some(evaluation.with_error(error))
}

/// `GREATEST` and `LEAST`: the extremum of the constant arguments when all
/// are known, the union of the non-`NULL` arguments' values otherwise.
fn extremum(
    least: bool,
    argument_evaluations: &[Evaluation],
    ty: ScalarType,
    error: bool,
) -> Evaluation {
    let non_null: Vec<&Evaluation> =
        argument_evaluations.iter().filter(|evaluation| !evaluation.is_always_null()).collect();
    if non_null.is_empty() {
        return Evaluation::null(ty).with_error(error);
    }
    let ordered = !matches!(ty, ScalarType::Text(rules) if !rules.byte_order)
        && !matches!(
            ty,
            ScalarType::UnknownLiteral
                | ScalarType::Unknown
                | ScalarType::Other
                | ScalarType::Float
        );
    // The extremum folds only when every argument is a constant.
    let constants: Option<Vec<&PgValue>> = if ordered {
        non_null.iter().map(|evaluation| evaluation.constant()).collect()
    } else {
        None
    };
    let best = constants.and_then(|constants| {
        constants.into_iter().reduce(|best, datum| {
            if (least && datum < best) || (!least && datum > best) { datum } else { best }
        })
    });
    let evaluation = if let Some(datum) = best {
        Evaluation::known(datum.clone(), ty)
    } else {
        let values = values_hull(non_null.iter().copied());
        let null = argument_evaluations.iter().all(Evaluation::may_be_null);
        Evaluation::new(ty, null, error, values)
    };
    evaluation.with_error(error)
}

/// The union of the evaluations' values, `NULL` and errors ignored.
fn values_hull<'a>(evaluations: impl Iterator<Item = &'a Evaluation>) -> Values {
    let mut hull: Option<Evaluation> = None;
    for evaluation in evaluations {
        let candidate = Evaluation::new(evaluation.ty(), false, false, evaluation.values().clone());
        hull = Some(match hull {
            Some(hull) => hull.join(&candidate),
            None => candidate,
        });
    }
    hull.map_or(Values::none(), |hull| hull.values().clone())
}

#[cfg(test)]
mod tests {
    use sql_values::{IntegerType, PgDate, PgNumeric, PgTimestamp, PgValue};

    use crate::{
        domain::{Evaluation, Unlisted, Values},
        scalar::{ScalarType, TextRules},
    };

    fn int(value: i64) -> Evaluation {
        Evaluation::known(PgValue::Integer(value), ScalarType::Integer(IntegerType::Integer))
    }

    fn num(text: &str) -> Evaluation {
        Evaluation::known(
            PgValue::Numeric(PgNumeric::parse(text).expect("a numeric literal")),
            ScalarType::Numeric,
        )
    }

    fn text(value: &str) -> Evaluation {
        Evaluation::known(PgValue::Text(value.to_string()), ScalarType::Text(TextRules::DEFAULT))
    }

    fn date_on(year: i32) -> Evaluation {
        let date = PgDate::from_ymd(year, 1, 1).expect("a valid date");
        Evaluation::known(PgValue::Date(date), ScalarType::Date)
    }

    fn date() -> Evaluation {
        date_on(2024)
    }

    fn timestamp() -> Evaluation {
        let midnight = PgTimestamp::from_pg_micros(0).expect("a valid timestamp");
        Evaluation::known(PgValue::Timestamp(midnight), ScalarType::Timestamp)
    }

    /// A nullable, non-raising column of `ty`.
    fn column(ty: ScalarType) -> Evaluation {
        Evaluation::new(ty, true, false, Values::Unlisted(Unlisted::any()))
    }

    /// `COALESCE` with no argument resolves nothing.
    #[test]
    fn coalesce_without_arguments_resolves_nothing() {
        assert!(super::name_recognised("coalesce", &[], false).is_none());
    }

    /// `COALESCE` keeps the first argument that is not known `NULL`.
    #[test]
    fn coalesce_keeps_the_first_non_null_argument() {
        let evaluation = super::name_recognised(
            "coalesce",
            &[Evaluation::null(ScalarType::Integer(IntegerType::Integer)), int(5)],
            false,
        )
        .expect("the call resolves");
        assert_eq!(evaluation.constant(), Some(&PgValue::Integer(5)));
    }

    /// A non-constant first `COALESCE` argument takes its values, never `NULL`.
    #[test]
    fn coalesce_over_a_column_is_never_null() {
        let evaluation = super::name_recognised(
            "coalesce",
            &[column(ScalarType::Integer(IntegerType::Integer)), int(5)],
            false,
        )
        .expect("the call resolves");
        assert!(!evaluation.may_be_null());
        assert!(evaluation.may_be(&PgValue::Integer(5)));
    }

    /// A `COALESCE` over only `NULL`-able arguments may itself be `NULL`.
    #[test]
    fn coalesce_over_only_nullable_arguments_may_be_null() {
        let ty = ScalarType::Integer(IntegerType::Integer);
        let evaluation = super::name_recognised("coalesce", &[column(ty), column(ty)], false)
            .expect("the call resolves");
        assert!(evaluation.may_be_null());
    }

    /// A `COALESCE` of a `date` and a `timestamp` resolves to `timestamp`.
    #[test]
    fn coalesce_of_date_and_timestamp_resolves_to_timestamp() {
        let evaluation = super::name_recognised("coalesce", &[date(), timestamp()], false)
            .expect("the call resolves");
        assert_eq!(evaluation.ty(), ScalarType::Timestamp);
    }

    /// Untyped literals alone resolve to `text`.
    #[test]
    fn untyped_arguments_resolve_to_text() {
        let literal = |text: &str| {
            Evaluation::known(PgValue::Text(text.to_string()), ScalarType::UnknownLiteral)
        };
        let call = super::name_recognised("coalesce", &[literal("t"), literal("f")], false)
            .expect("resolves");
        assert_eq!(call.ty(), ScalarType::Text(TextRules::DEFAULT));
    }

    /// A call outside the four constructs is left to its callees, whatever its
    /// argument types.
    #[test]
    fn another_call_is_not_resolved_by_name() {
        assert!(super::name_recognised("concat", &[text("a"), int(1)], false).is_none());
        let literal = Evaluation::known(PgValue::Text("x".to_string()), ScalarType::UnknownLiteral);
        assert!(super::name_recognised("concat", &[literal, int(1)], false).is_none());
    }

    /// A mix of type categories PostgreSQL refuses raises the call.
    #[test]
    fn a_refused_type_mix_raises() {
        let text_null = Evaluation::null(ScalarType::Text(TextRules::DEFAULT));
        let flag = Evaluation::known(PgValue::Boolean(true), ScalarType::Boolean);
        for name in ["coalesce", "nullif", "greatest", "least"] {
            for arguments in [[text_null.clone(), int(5)], [date(), int(5)], [flag.clone(), int(1)]]
            {
                let call = super::name_recognised(name, &arguments, false).expect("resolves");
                assert!(call.always_raises(), "{name}");
            }
        }
    }

    /// An argument of a type the model does not read may not match the others.
    #[test]
    fn an_unmodelled_argument_may_raise() {
        for name in ["coalesce", "nullif", "greatest", "least"] {
            let arguments = [column(ScalarType::Other), int(5)];
            let call = super::name_recognised(name, &arguments, false).expect("resolves");
            assert!(call.may_raise(), "{name}");
            assert!(!call.always_raises(), "{name}");
        }
    }

    /// A `float` argument resolves the numbers to `float` and keeps its values.
    #[test]
    fn a_float_argument_resolves_the_numbers_to_float() {
        let null_int = Evaluation::null(ScalarType::Integer(IntegerType::Integer));
        let call =
            super::name_recognised("coalesce", &[null_int, column(ScalarType::Float)], false)
                .expect("resolves");
        assert_eq!(call.ty(), ScalarType::Float);
        assert!(call.may_be(&PgValue::Numeric(PgNumeric::parse("2.5").expect("a number"))));
        let call = super::name_recognised("greatest", &[int(5), column(ScalarType::Float)], false)
            .expect("resolves");
        assert_eq!(call.ty(), ScalarType::Float);
        assert!(!call.may_raise());
        // A `numeric` past the `float` range raises reading as `float`.
        let call = super::name_recognised("least", &[num("1.5"), column(ScalarType::Float)], false)
            .expect("resolves");
        assert!(call.may_raise());
    }

    /// Integers of different widths resolve to the wider one.
    #[test]
    fn integers_resolve_to_the_wider_width() {
        let small =
            Evaluation::known(PgValue::Integer(1), ScalarType::Integer(IntegerType::Smallint));
        let big = Evaluation::known(PgValue::Integer(2), ScalarType::Integer(IntegerType::Bigint));
        let call = super::name_recognised("greatest", &[small, big], false).expect("resolves");
        assert_eq!(call.ty(), ScalarType::Integer(IntegerType::Bigint));
        assert_eq!(call.constant(), Some(&PgValue::Integer(2)));
    }

    /// Text under different collations may raise where the call compares it.
    #[test]
    fn text_under_different_collations_may_raise() {
        let default = column(ScalarType::Text(TextRules::DEFAULT));
        let bytewise = column(ScalarType::Text(TextRules::BYTEWISE));
        let call =
            super::name_recognised("greatest", &[default, bytewise], false).expect("resolves");
        assert_eq!(call.ty(), ScalarType::Text(TextRules::UNKNOWN));
        assert!(call.may_raise());
    }

    /// A `COALESCE` of `numeric` and an integer resolves to `numeric`.
    #[test]
    fn coalesce_of_numeric_and_integer_resolves_to_numeric() {
        let evaluation = super::name_recognised("coalesce", &[num("1.5"), int(5)], false)
            .expect("the call resolves");
        assert_eq!(evaluation.ty(), ScalarType::Numeric);
    }

    /// `COALESCE`, `GREATEST` and `LEAST` read a `date` argument as the
    /// `timestamp` they resolve to, which raises for a date past the
    /// timestamp range or a date column that may hold one.
    #[test]
    fn a_date_widened_to_timestamp_may_raise() {
        for name in ["coalesce", "greatest", "least"] {
            let late = super::name_recognised(name, &[date_on(294_277), timestamp()], false)
                .expect("the call resolves");
            assert!(late.may_raise(), "{name}");
            let column =
                super::name_recognised(name, &[column(ScalarType::Date), timestamp()], false)
                    .expect("the call resolves");
            assert!(column.may_raise(), "{name}");
            let early = super::name_recognised(name, &[date(), timestamp()], false)
                .expect("the call resolves");
            assert!(!early.may_raise(), "{name}");
        }
    }

    /// `NULLIF` compares a `date` with a `timestamp` without converting it,
    /// so it keeps the `date` type.
    #[test]
    fn nullif_of_a_date_and_a_timestamp_keeps_the_date() {
        let evaluation = super::name_recognised("nullif", &[date(), timestamp()], false)
            .expect("the call resolves");
        assert_eq!(evaluation.ty(), ScalarType::Date);
    }

    /// `NULLIF` with two unequal constants is the left value.
    #[test]
    fn nullif_of_unequal_constants_is_the_left() {
        let evaluation =
            super::name_recognised("nullif", &[int(5), int(6)], false).expect("the call resolves");
        assert_eq!(evaluation.constant(), Some(&PgValue::Integer(5)));
    }

    /// `NULLIF` of more than two arguments resolves nothing.
    #[test]
    fn nullif_of_three_arguments_resolves_nothing() {
        assert!(super::name_recognised("nullif", &[int(1), int(2), int(3)], false).is_none());
    }

    /// A `NULLIF` over a column is `NULL` when the column may be.
    #[test]
    fn nullif_over_a_column_may_be_null() {
        let evaluation = super::name_recognised(
            "nullif",
            &[column(ScalarType::Integer(IntegerType::Integer)), int(6)],
            false,
        )
        .expect("the call resolves");
        assert!(evaluation.may_be_null());
        assert_eq!(evaluation.ty(), ScalarType::Integer(IntegerType::Integer));
    }

    /// `GREATEST` and `LEAST` of constants give the extremum.
    #[test]
    fn greatest_and_least_of_constants() {
        let args = [int(3), int(5), int(1)];
        let greatest = super::name_recognised("greatest", &args, false).expect("the call resolves");
        assert_eq!(greatest.constant(), Some(&PgValue::Integer(5)));
        let least = super::name_recognised("least", &args, false).expect("the call resolves");
        assert_eq!(least.constant(), Some(&PgValue::Integer(1)));
    }

    /// An unorderable `GREATEST` (default-collation text) leaves the values.
    #[test]
    fn greatest_of_default_text_leaves_the_values() {
        let args = [text("a"), text("b")];
        let evaluation =
            super::name_recognised("greatest", &args, false).expect("the call resolves");
        assert!(!evaluation.may_be_null());
        assert!(evaluation.may_be(&PgValue::Text("a".to_string())));
        assert!(evaluation.may_be(&PgValue::Text("b".to_string())));
    }

    /// A `GREATEST` of only `NULL` arguments is `NULL`.
    #[test]
    fn greatest_of_only_null_arguments_is_null() {
        let args = [
            Evaluation::null(ScalarType::Integer(IntegerType::Integer)),
            Evaluation::null(ScalarType::Integer(IntegerType::Integer)),
        ];
        let evaluation =
            super::name_recognised("greatest", &args, false).expect("the call resolves");
        assert!(evaluation.is_always_null());
    }

    /// The values hull joins the non-`NULL` members and drops `NULL` and
    /// errors.
    #[test]
    fn the_values_hull_joins_members() {
        let a = int(1);
        let b = int(2);
        let values = super::values_hull([a, b].iter());
        assert!(values.admits(&PgValue::Integer(1)));
        assert!(values.admits(&PgValue::Integer(2)));
    }

    fn literal(value: &str) -> Evaluation {
        Evaluation::known(PgValue::Text(value.to_string()), ScalarType::UnknownLiteral)
    }

    /// The four calls read an untyped literal with the input function of
    /// the type they resolve to, so a literal it cannot read raises the
    /// call, as `coalesce('a', 5)` and `greatest('a', 5)` do.
    #[test]
    fn an_unreadable_literal_raises_the_call() {
        for name in ["coalesce", "greatest", "least", "nullif"] {
            let call = super::name_recognised(name, &[literal("a"), int(5)], false)
                .expect("the call resolves");
            assert!(call.always_raises(), "{name}");
        }
        let late = super::name_recognised(
            "coalesce",
            &[Evaluation::null(ScalarType::Unknown), literal("a"), int(5)],
            false,
        )
        .expect("the call resolves");
        assert!(late.always_raises());
    }

    /// A readable literal takes the resolved type before the call compares
    /// or returns it, as `nullif('5', 5)` is `NULL`, `nullif('5', 6)` is
    /// `5` and `coalesce('5', 6.5)` is the `numeric` `5`.
    #[test]
    fn a_readable_literal_takes_the_resolved_type() {
        let equal = super::name_recognised("nullif", &[literal("5"), int(5)], false)
            .expect("the call resolves");
        assert!(equal.is_always_null());
        let swapped = super::name_recognised("nullif", &[int(5), literal("5")], false)
            .expect("the call resolves");
        assert!(swapped.is_always_null());
        let unequal = super::name_recognised("nullif", &[literal("5"), int(6)], false)
            .expect("the call resolves");
        assert_eq!(unequal.constant(), Some(&PgValue::Integer(5)));
        let first = super::name_recognised("coalesce", &[literal("5"), num("6.5")], false)
            .expect("the call resolves");
        assert_eq!(first.ty(), ScalarType::Numeric);
        assert_eq!(first.constant(), Some(&PgValue::Integer(5)));
        let greatest = super::name_recognised("greatest", &[literal("3"), int(5)], false)
            .expect("the call resolves");
        assert_eq!(greatest.constant(), Some(&PgValue::Integer(5)));
    }

    /// A `COALESCE` argument that cannot be `NULL` ends the call, so the
    /// arguments after it add no value.
    #[test]
    fn coalesce_stops_at_an_argument_that_cannot_be_null() {
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let never_null = Evaluation::new(
            int4,
            false,
            false,
            Values::Listed([PgValue::Integer(1), PgValue::Integer(2)].into()),
        );
        let call = super::name_recognised("coalesce", &[never_null, int(5)], false)
            .expect("the call resolves");
        assert!(!call.may_be_null());
        assert!(!call.may_be(&PgValue::Integer(5)));
        let nullable = super::name_recognised("coalesce", &[column(int4), int(5)], false)
            .expect("the call resolves");
        assert!(nullable.may_be(&PgValue::Integer(5)));
    }
}
