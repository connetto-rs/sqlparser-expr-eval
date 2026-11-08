//! The conservative reading the default dialect bodies and the value-domain
//! constructors settle on.

extern crate alloc;

use alloc::string::String;
use core::ops::Bound;

use sqlparser::{
    ast::{
        BinaryOperator, CastKind, CharacterLength, DataType, ExactNumberInfo, Expr, Function,
        FunctionArguments, Ident, ObjectName,
    },
    dialect::GenericDialect,
};
use sqlparser_expr_eval::{
    Callee, CalleeKind, EvalDialect, Evaluation, Interval, NullInput, Outcomes, QueryEvaluation,
    RowKinds, Rows, ScalarType, TextRules, Values, Volatility,
    sql_values::{IntegerType, PgNumeric},
};

fn int4(value: i64) -> Evaluation {
    Evaluation::known(
        sqlparser_expr_eval::sql_values::PgValue::Integer(value),
        ScalarType::Integer(IntegerType::Integer),
    )
}

fn unlisted(ty: ScalarType) -> Evaluation {
    Evaluation::new(ty, false, false, Values::Unlisted(sqlparser_expr_eval::Unlisted::any()))
}

#[test]
fn the_default_dialect_answers_conservatively() {
    let dialect = GenericDialect;
    let options = ();
    let right = Evaluation::unknown(ScalarType::Integer(IntegerType::Integer));

    let comparison = dialect.comparison(int4(1), &BinaryOperator::Eq, right.clone(), &options);
    assert_eq!(comparison.rows, RowKinds::ANY);

    assert!(dialect.concat(&int4(1), &right, &options).is_none());

    let sum = dialect.arithmetic(&int4(1), &BinaryOperator::Plus, &right, &options);
    assert!(sum.ty().is_number());
    assert!(sum.may_raise());
    // The unknown operand may be NULL.
    assert!(sum.may_be_null());
    let quotient = dialect.arithmetic(&int4(2), &BinaryOperator::Divide, &right, &options);
    assert!(quotient.may_be_null());
    assert!(!dialect.unary_plus_preserves(&int4(1), &options));

    let cast = dialect.cast(
        int4(1),
        &CastKind::Cast,
        &DataType::Int(None),
        ScalarType::Integer(IntegerType::Bigint),
        |_| ScalarType::Integer(IntegerType::Bigint),
        &options,
    );
    assert!(matches!(cast.ty(), ScalarType::Integer(IntegerType::Bigint)));
    assert!(cast.may_raise());
    let null_cast = dialect.cast(
        Evaluation::null(ScalarType::Integer(IntegerType::Integer)),
        &CastKind::Cast,
        &DataType::Int(None),
        ScalarType::Integer(IntegerType::Bigint),
        |_| ScalarType::Integer(IntegerType::Bigint),
        &options,
    );
    assert!(null_cast.may_be_null() && null_cast.may_raise());
    let untyped_null = dialect.cast(
        Evaluation::null(ScalarType::Unknown),
        &CastKind::Cast,
        &DataType::Int(None),
        ScalarType::Integer(IntegerType::Bigint),
        |_| ScalarType::Integer(IntegerType::Bigint),
        &options,
    );
    assert!(untyped_null.is_always_null());
    // A non-strict cast may fail and answer `NULL`, or raise for a cast the
    // engine does not permit.
    let try_cast = dialect.cast(
        int4(1),
        &CastKind::TryCast,
        &DataType::Date,
        ScalarType::Date,
        |_| ScalarType::Date,
        &options,
    );
    assert!(try_cast.may_be_null() && try_cast.may_raise());
    // A `numeric(p, s)` typmod may refuse a `numeric` past its precision.
    let numeric = unlisted(ScalarType::Numeric);
    let bounded = dialect.cast(
        numeric,
        &CastKind::Cast,
        &DataType::Numeric(ExactNumberInfo::PrecisionAndScale(4, 1)),
        ScalarType::Numeric,
        |_| ScalarType::Numeric,
        &options,
    );
    assert!(bounded.may_raise());
    // A `float8` cast to `real` may overflow.
    let narrowed = dialect.cast(
        unlisted(ScalarType::Float),
        &CastKind::Cast,
        &DataType::Real,
        ScalarType::Float,
        |_| ScalarType::Float,
        &options,
    );
    assert!(narrowed.may_raise());
    // A `varchar(0)` typmod refuses every value.
    let empty_varchar = dialect.cast(
        unlisted(ScalarType::Text(TextRules::DEFAULT)),
        &CastKind::DoubleColon,
        &DataType::Varchar(Some(CharacterLength::IntegerLength { length: 0, unit: None })),
        ScalarType::Text(TextRules::DEFAULT),
        |_| ScalarType::Text(TextRules::DEFAULT),
        &options,
    );
    assert!(empty_varchar.may_raise());

    assert!(dialect.named_call("length", &[int4(1)], &options).is_none());
    let left_expr = Expr::Identifier(Ident::new("x"));
    let right_expr = Expr::Identifier(Ident::new("y"));
    assert!(dialect.null_equals_operand(&left_expr, &right_expr, &options).is_none());
    assert!(!dialect.null_only_on_null_input(&[], &options));
    let call = Function {
        name: ObjectName::from(Ident::new("length")),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::None,
        within_group: Vec::new(),
        filter: None,
        null_treatment: None,
        over: None,
    };
    assert!(dialect.counted_length_argument(&call, &[], &options).is_none());
}

#[test]
fn interval_endpoints_are_read_exactly() {
    let open = Interval { low: Bound::Included(PgNumeric::from(1)), high: Bound::Unbounded };
    assert!(open.contains(&PgNumeric::from(1)));
    assert!(open.contains(&PgNumeric::from(10_000)));
    let excluded = Interval {
        low: Bound::Included(PgNumeric::from(1)),
        high: Bound::Excluded(PgNumeric::from(10)),
    };
    assert!(excluded.contains(&PgNumeric::from(9)));
    assert!(!excluded.contains(&PgNumeric::from(10)));
    let empty = Interval {
        low: Bound::Included(PgNumeric::from(3)),
        high: Bound::Excluded(PgNumeric::from(3)),
    };
    assert!(empty.is_empty());
    assert!(!open.is_empty());
}

#[test]
fn unlisted_values_admit_and_meet() {
    let wide = unlisted(ScalarType::Integer(IntegerType::Integer));
    let narrow = unlisted(ScalarType::Integer(IntegerType::Smallint));
    assert!(wide.values().admits(&sqlparser_expr_eval::sql_values::PgValue::Integer(3)));
    assert!(wide.number_range().contains(&PgNumeric::from(i64::from(i32::MIN))));

    let meet = wide.meet(&narrow);
    assert!(meet.values().admits(&sqlparser_expr_eval::sql_values::PgValue::Integer(3)));
    let range = meet.number_range();
    assert!(range.contains(&PgNumeric::from(30_000)));
    assert!(!range.contains(&PgNumeric::from(40_000)));
}

#[test]
fn outcomes_combine_and_negate() {
    let both = Outcomes::TRUE.union(Outcomes::FALSE);
    assert_eq!(both.intersection(Outcomes::TRUE), Outcomes::TRUE);
    assert!(Outcomes::TRUE.intersection(Outcomes::FALSE).is_empty());
    assert!(!both.is_empty());
    assert_eq!(Outcomes::NULL.negated(), Outcomes::NULL);
    assert_ne!(Outcomes::TRUE.negated(), Outcomes::TRUE);
}

#[test]
fn evaluation_constructors_and_predicates() {
    let condition = Evaluation::condition(Outcomes::TRUE.union(Outcomes::FALSE), true);
    assert!(condition.may_be_true());
    assert!(condition.may_be_false());
    assert!(condition.may_raise());

    let null = Evaluation::null(ScalarType::Boolean);
    assert!(null.is_always_null());
    assert!(!null.may_be_true());

    let raising = Evaluation::raising(ScalarType::Boolean);
    assert!(raising.always_raises());
    assert!(!raising.is_always_null());

    let unknown = Evaluation::unknown(ScalarType::Unknown);
    assert!(unknown.is_uninformative());
    assert!(unknown.may_be_true());

    assert!(!condition.outcomes().is_empty());
}

#[test]
fn evaluation_outcomes_ranges_and_rewrites() {
    let range = int4(7).number_range();
    assert!(matches!(
        (range.low, range.high),
        (Bound::Included(value), Bound::Included(high)) if value == PgNumeric::from(7) && high == PgNumeric::from(7)
    ));
    let wide = unlisted(ScalarType::Integer(IntegerType::Integer));
    assert!(wide.number_range().contains(&PgNumeric::from(5)));
    assert!(unlisted(ScalarType::Text(TextRules::DEFAULT)).length_range().contains(&0));

    let text = Evaluation::known(
        sqlparser_expr_eval::sql_values::PgValue::Text(String::from("abc")),
        ScalarType::Text(TextRules::DEFAULT),
    );
    assert!(text.length_range().contains(&3));

    assert!(int4(1).with_null(true).may_be_null());
    assert!(int4(1).with_error(true).may_raise());
    assert!(matches!(int4(1).with_type(ScalarType::Numeric).ty(), ScalarType::Numeric));
    assert!(int4(1).with_values(Values::none()).is_impossible());

    let wide = unlisted(ScalarType::Integer(IntegerType::Integer));
    let joined = int4(1).join(&wide);
    assert!(joined.ty().is_number());
    let met = Evaluation::unknown(ScalarType::Unknown).meet(&int4(1));
    assert!(matches!(met.ty(), ScalarType::Integer(IntegerType::Integer)));
}

#[test]
fn normalisation_resolves_the_boolean_unlisted() {
    let boolean = Evaluation::new(
        ScalarType::Boolean,
        false,
        false,
        Values::Unlisted(sqlparser_expr_eval::Unlisted::any()),
    );
    assert!(boolean.may_be_true());
    assert!(boolean.may_be_false());
    // Normalisation lists the two admitted booleans.
    assert!(!boolean.values().is_empty());
}

#[test]
fn the_callee_and_query_constructors_hold_their_fields() {
    let reading = QueryEvaluation::new(Rows::ONE, Some(vec![int4(1)]), false);
    assert_eq!(reading.rows, Rows::ONE);
    assert!(reading.columns.as_deref().is_some_and(|columns| {
        matches!(columns[0].ty(), ScalarType::Integer(IntegerType::Integer))
    }));
    assert!(!reading.error);
    let callee = Callee::new(
        None,
        NullInput::Strict,
        Volatility::Immutable,
        CalleeKind::Normal,
        ScalarType::Integer(IntegerType::Integer),
        false,
        0,
    );
    assert_eq!(callee.returns, ScalarType::Integer(IntegerType::Integer));
    let rows = Rows::new(true, false, true);
    assert!(rows.none);
    assert!(rows.many);
    assert!(!rows.one);
}
