//! The PostgreSQL reading of a comparison of two evaluated operands.

use sqlparser::ast::BinaryOperator;

use crate::{
    domain::Evaluation,
    forward::fold::{
        Category, Comparison, RowKinds, comparison_category, pair_analysis, reinterpret_unknown,
    },
    scalar::{ScalarType, TextRules},
};

/// The comparison of two already-evaluated operands under PostgreSQL's
/// literal typing and value comparison.
pub(super) fn comparison(left: Evaluation, op: &BinaryOperator, right: Evaluation) -> Comparison {
    let error = left.may_raise() || right.may_raise();
    let literal =
        *left.ty() == ScalarType::UnknownLiteral || *right.ty() == ScalarType::UnknownLiteral;
    let (left, right) = resolve_unknown_sides(left, right);
    // Resolving an untyped literal against the other side may raise.
    let error = error || left.may_raise() || right.may_raise();
    // A `date` compares with a `timestamp` through `date_cmp_timestamp`,
    // which never raises, so the date side only takes the timestamp type.
    let (mut left, mut right) = (left, right);
    if *left.ty() == ScalarType::Date && *right.ty() == ScalarType::Timestamp {
        left = left.with_type(ScalarType::Timestamp);
    }
    if *right.ty() == ScalarType::Date && *left.ty() == ScalarType::Timestamp {
        right = right.with_type(ScalarType::Timestamp);
    }
    let modelled = !left.ty().is_unmodelled() && !right.ty().is_unmodelled();
    // The evaluator reads `varchar` as `text`, and PostgreSQL compares a
    // `varchar` array with no `text` array, save an untyped literal that
    // takes the other side's type.
    let text_arrays = |category: &Category<'_>| {
        matches!(category, Category::Array(element) if element.is_text()) && !literal
    };
    if left.is_always_null() || right.is_always_null() {
        let refused = modelled
            && comparison_category(left.ty(), right.ty())
                .is_none_or(|category| text_arrays(&category));
        return Comparison::new(RowKinds::NONE, error || refused);
    }
    let valid_op = matches!(
        op,
        BinaryOperator::Eq
            | BinaryOperator::NotEq
            | BinaryOperator::Lt
            | BinaryOperator::LtEq
            | BinaryOperator::Gt
            | BinaryOperator::GtEq
    );
    if !modelled {
        if valid_op && ordered_builtin(left.ty(), right.ty()) {
            return Comparison::new(RowKinds::ANY, error);
        }
        // An unmodelled type may have the operator, with any outcome.
        return crate::forward::fold::standard_comparison(&left, op, &right, true);
    }
    let Some(category) = comparison_category(left.ty(), right.ty()) else {
        return Comparison::new(RowKinds::NONE, true);
    };
    if !valid_op {
        return Comparison::new(RowKinds::NONE, true);
    }
    // Two different implicit collations leave the comparison without a
    // collation, so PostgreSQL raises on every pair of values it compares.
    if let Category::Text(rules_left, rules_right) = category
        && TextRules::conflicting(rules_left, rules_right)
    {
        return Comparison::new(RowKinds::NONE, true);
    }
    let refused = text_arrays(&category);
    Comparison::new(pair_analysis(&left, &right, &category, op), error || refused)
}

/// Whether both sides are one built-in type whose default btree operator
/// class compares any two of its values without raising. A type goes by its
/// name, as [`OtherType`](crate::OtherType) identifies it, so a bare or a
/// `pg_catalog` spelling names the built-in.
fn ordered_builtin(left: &ScalarType, right: &ScalarType) -> bool {
    let (ScalarType::Other(Some(left)), ScalarType::Other(Some(right))) = (left, right) else {
        return false;
    };
    left == right
        && matches!(
            left.name(),
            "uuid"
                | "timestamptz"
                | "time"
                | "timetz"
                | "interval"
                | "bytea"
                | "inet"
                | "cidr"
                | "macaddr"
                | "macaddr8"
                | "jsonb"
                | "bit"
                | "varbit"
                | "money"
        )
}

/// The operands with their unknown-literal types resolved against the other
/// operand, and both untyped operands read as text under the default
/// collation.
fn resolve_unknown_sides(left: Evaluation, right: Evaluation) -> (Evaluation, Evaluation) {
    match (left.ty().is_untyped(), right.ty().is_untyped()) {
        (true, false) => (reinterpret_unknown(left, right.ty()), right),
        (false, true) => {
            let right = reinterpret_unknown(right, left.ty());
            (left, right)
        }
        (true, true) => {
            let text = || ScalarType::Text(TextRules::DEFAULT);
            let left = if *left.ty() == ScalarType::UnknownLiteral {
                left.with_type(text())
            } else {
                left
            };
            let right = if *right.ty() == ScalarType::UnknownLiteral {
                right.with_type(text())
            } else {
                right
            };
            (left, right)
        }
        _ => (left, right),
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;

    use sql_values::postgres::{IntegerType, PgValue};
    use sqlparser::ast::BinaryOperator;

    use super::comparison;
    use crate::{
        domain::{Evaluation, Unlisted, Values},
        scalar::{EnumType, ScalarType, TextRules},
        test_context::{ColumnContext, read},
    };

    fn int(ty: IntegerType) -> Evaluation {
        Evaluation::new(ScalarType::Integer(ty), true, false, Values::Unlisted(Unlisted::any()))
    }

    fn numeric() -> Evaluation {
        Evaluation::new(ScalarType::Numeric, true, false, Values::Unlisted(Unlisted::any()))
    }

    fn text(rules: TextRules) -> Evaluation {
        Evaluation::new(ScalarType::Text(rules), true, false, Values::Unlisted(Unlisted::any()))
    }

    fn listed(values: impl IntoIterator<Item = PgValue>) -> BTreeSet<PgValue> {
        values.into_iter().collect()
    }

    #[test]
    fn number_pairs_compare_across_widths_and_numeric() {
        let i2 = int(IntegerType::Smallint);
        let i4 = int(IntegerType::Integer);
        let i8 = int(IntegerType::Bigint);
        for pair in
            [(i2.clone(), i8.clone()), (i4.clone(), numeric()), (i2, i4), (numeric(), numeric())]
        {
            let result = read(
                ColumnContext::new()
                    .with_column("a", pair.0.clone())
                    .with_column("b", pair.1.clone()),
                "a = b",
            );
            assert!(result.may_be_true(), "a = b may be true for {pair:?}");
            assert!(result.may_be_false());
            assert!(result.may_be_null());
            assert!(!result.may_raise());
        }
        // Values compare by value across the types, so a shared constant is
        // a certainty once both sides list it.
        let result = read(
            ColumnContext::new()
                .with_column(
                    "a",
                    Evaluation::known(
                        PgValue::Integer(3),
                        ScalarType::Integer(IntegerType::Smallint),
                    ),
                )
                .with_column(
                    "b",
                    Evaluation::known(
                        PgValue::Numeric(sql_values::postgres::PgNumeric::from(3_i64)),
                        ScalarType::Numeric,
                    ),
                ),
            "a = b",
        );
        assert!(result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_be_null());
    }

    #[test]
    fn unknown_literals_resolve_against_each_type() {
        // Against a listed value, the resolved literal decides the pair.
        let i4 = Evaluation::known(PgValue::Integer(5), ScalarType::Integer(IntegerType::Integer));
        let resolved = comparison(
            Evaluation::known(PgValue::Text("5".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            i4,
        );
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());

        let resolved = comparison(
            Evaluation::known(PgValue::Text("5".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            numeric(),
        );
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // The resolution raises on a member the type rejects.
        let resolved = comparison(
            Evaluation::known(PgValue::Text("5.5".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Integer(5), ScalarType::Integer(IntegerType::Integer)),
        );
        assert!(resolved.error);
        assert!(!resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());

        // An unfamiliar spelling is left undecided and raises nothing.
        let resolved = comparison(
            Evaluation::known(PgValue::Text("NaN".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            numeric(),
        );
        assert!(!resolved.error);
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // A boolean literal resolves against a boolean column.
        let resolved = comparison(
            Evaluation::known(PgValue::Text("t".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Boolean(true), ScalarType::Boolean),
        );
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());
        let resolved = comparison(
            Evaluation::known(PgValue::Text("x".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Boolean(true), ScalarType::Boolean),
        );
        assert!(resolved.error);

        // Date and timestamp literals resolve against their columns.
        let date = sql_values::postgres::parse_date("2026-01-01").expect("a valid date");
        let resolved = comparison(
            Evaluation::known(PgValue::Date(date), ScalarType::Date),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Text("2026-01-01".to_string()), ScalarType::UnknownLiteral),
        );
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());
        let timestamp = sql_values::postgres::parse_timestamp("2026-01-01 03:00:00")
            .expect("a valid timestamp");
        let resolved = comparison(
            Evaluation::known(PgValue::Timestamp(timestamp), ScalarType::Timestamp),
            &BinaryOperator::Eq,
            Evaluation::known(
                PgValue::Text("2026-01-01 03:00:00".to_string()),
                ScalarType::UnknownLiteral,
            ),
        );
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());
        // Every spelling `date_in` reads resolves, and one whose reading the
        // `DateStyle` field order decides is undecided, not an error.
        let resolved = comparison(
            Evaluation::known(PgValue::Date(date), ScalarType::Date),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Text("Jan 1 2026".to_string()), ScalarType::UnknownLiteral),
        );
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());
        let resolved = comparison(
            Evaluation::known(PgValue::Date(date), ScalarType::Date),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Text("01/01/2026".to_string()), ScalarType::UnknownLiteral),
        );
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());
    }

    #[test]
    fn unknown_literals_stay_open_against_setting_dependent_types() {
        let resolved = comparison(
            Evaluation::known(PgValue::Text("abc".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            Evaluation::new(ScalarType::Float, true, false, Values::Unlisted(Unlisted::any())),
        );
        assert!(resolved.error);
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // An operand of a type the model does not read may match the literal.
        let resolved = comparison(
            Evaluation::known(PgValue::Text("a".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            Evaluation::new(ScalarType::Unknown, true, false, Values::Unlisted(Unlisted::any())),
        );
        assert!(resolved.error);
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // Both untyped operands read as text under the default collation.
        let resolved = comparison(
            Evaluation::known(PgValue::Text("a".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Text("b".to_string()), ScalarType::UnknownLiteral),
        );
        assert!(!resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());
        let resolved = comparison(
            Evaluation::known(PgValue::Text("a".to_string()), ScalarType::UnknownLiteral),
            &BinaryOperator::Lt,
            Evaluation::known(PgValue::Text("b".to_string()), ScalarType::UnknownLiteral),
        );
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());
    }

    #[test]
    fn dates_compare_against_timestamps_at_midnight() {
        // A listed date reads as its midnight against a timestamp.
        let date = sql_values::postgres::parse_date("2026-01-01").expect("a valid date");
        let timestamp = sql_values::postgres::parse_timestamp("2026-01-01 00:00:00")
            .expect("a valid timestamp");
        let resolved = comparison(
            Evaluation::known(PgValue::Date(date), ScalarType::Date),
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Timestamp(timestamp), ScalarType::Timestamp),
        );
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());
        let resolved = comparison(
            Evaluation::known(PgValue::Timestamp(timestamp), ScalarType::Timestamp),
            &BinaryOperator::Gt,
            Evaluation::known(PgValue::Date(date), ScalarType::Date),
        );
        assert!(!resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // An unlisted date stays open, and the comparison never raises.
        let date_column =
            Evaluation::new(ScalarType::Date, true, false, Values::Unlisted(Unlisted::any()));
        let resolved = comparison(
            date_column,
            &BinaryOperator::Eq,
            Evaluation::known(PgValue::Timestamp(timestamp), ScalarType::Timestamp),
        );
        assert!(!resolved.error);
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // End to end, through the typed literals the parser reads.
        let result =
            read(ColumnContext::new(), "DATE '2026-01-01' = TIMESTAMP '2026-01-01 00:00:00'");
        assert!(result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_raise());
    }

    /// `date_cmp_timestamp` places a date past the timestamp range after
    /// every finite timestamp, without raising.
    #[test]
    fn a_date_past_the_timestamp_range_compares_without_raising() {
        let after =
            read(ColumnContext::new(), "DATE '5874897-12-31' > TIMESTAMP '2000-01-01 00:00:00'");
        assert!(after.may_be_true());
        assert!(!after.may_be_false());
        assert!(!after.may_raise());
        let equal =
            read(ColumnContext::new(), "DATE '5874897-12-31' = TIMESTAMP '2000-01-01 00:00:00'");
        assert!(!equal.may_be_true());
        assert!(equal.may_be_false());
        assert!(!equal.may_raise());
    }

    /// PostgreSQL holds dates from 4714-11-24 BC.
    #[test]
    fn the_earliest_date_is_4714_11_24_bc() {
        let result = read(ColumnContext::new(), "DATE '4713-01-01 BC' - 1 = DATE '4714-12-31 BC'");
        assert!(result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_raise());
    }

    fn text_of(rules: TextRules, value: &str) -> Evaluation {
        Evaluation::known(PgValue::Text(value.to_string()), ScalarType::Text(rules))
    }

    /// A quantified comparison the pair analysis cannot decide keeps both
    /// outcomes.
    #[test]
    fn an_undecided_quantified_comparison_stays_open() {
        let column = |ty| Evaluation::new(ty, true, false, Values::Unlisted(Unlisted::any()));
        let context = || {
            ColumnContext::new()
                .with_column("b", column(ScalarType::Boolean))
                .with_column("d", column(ScalarType::Date))
                .with_column("t", column(ScalarType::Text(TextRules::DEFAULT)))
        };
        for sql in [
            "b < ANY (ARRAY[true])",
            "d = ANY (ARRAY[DATE '2020-01-01'])",
            "d < ALL (ARRAY[DATE '2020-01-01'])",
            "t < ANY (ARRAY['a'])",
        ] {
            let result = read(context(), sql);
            assert!(result.may_be_true() && result.may_be_false(), "{sql}");
        }
    }

    #[test]
    fn text_compares_under_its_collation() {
        // An unlisted column against a literal keeps every open pair open.
        let resolved = comparison(
            text(TextRules::DEFAULT),
            &BinaryOperator::Eq,
            text_of(TextRules::DEFAULT, "a"),
        );
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());
        assert!(resolved.rows.all_true);
        let resolved = comparison(
            text(TextRules::UNKNOWN),
            &BinaryOperator::Eq,
            text_of(TextRules::UNKNOWN, "a"),
        );
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());
        assert!(resolved.rows.all_true);
        let resolved =
            comparison(text(TextRules::C), &BinaryOperator::Lt, text_of(TextRules::C, "a"));
        assert!(resolved.rows.all_true);
        assert!(resolved.rows.all_false);
        let resolved = comparison(
            text(TextRules::DEFAULT),
            &BinaryOperator::Lt,
            text_of(TextRules::DEFAULT, "a"),
        );
        assert!(resolved.rows.all_true);
        assert!(resolved.rows.all_false);

        // Equality decides only under an identity collation.
        let a = text_of(TextRules::DEFAULT, "a");
        let b = text_of(TextRules::DEFAULT, "b");
        let resolved = comparison(a.clone(), &BinaryOperator::Eq, b);
        assert!(!resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());
        // The unknown collation may equate the two spellings, so the
        // inequality stays open on both sides.
        let resolved = comparison(a, &BinaryOperator::NotEq, text_of(TextRules::UNKNOWN, "b"));
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // Ordering decides only under a byte collation.
        let a = text_of(TextRules::C, "a");
        let b = text_of(TextRules::C, "b");
        let resolved = comparison(a.clone(), &BinaryOperator::Lt, b.clone());
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());
        let resolved = comparison(a, &BinaryOperator::Gt, b);
        assert!(!resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());
        let a = text_of(TextRules::DEFAULT, "a");
        let b = text_of(TextRules::DEFAULT, "b");
        let resolved = comparison(a, &BinaryOperator::LtEq, b);
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // A listed column against a literal keeps every open pair open.
        let both = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            false,
            false,
            Values::Listed(listed([
                PgValue::Text("a".to_string()),
                PgValue::Text("b".to_string()),
            ])),
        );
        let resolved =
            comparison(both.clone(), &BinaryOperator::Eq, text_of(TextRules::DEFAULT, "a"));
        assert!(resolved.rows.may_be_true());
        assert!(resolved.rows.may_be_false());

        // Byte order over two listed values decides every pair.
        let resolved = comparison(
            text_of(TextRules::C, "a"),
            &BinaryOperator::LtEq,
            text_of(TextRules::C, "b"),
        );
        assert!(resolved.rows.may_be_true());
        assert!(!resolved.rows.may_be_false());
    }

    #[test]
    fn booleans_compare_by_value() {
        let flag =
            Evaluation::new(ScalarType::Boolean, true, false, Values::Unlisted(Unlisted::any()));
        let result = read(ColumnContext::new().with_column("flag", flag), "flag = true");
        assert!(result.may_be_true());
        assert!(result.may_be_false());
        assert!(result.may_be_null());

        let result = read(
            ColumnContext::new().with_column(
                "flag",
                Evaluation::known(PgValue::Boolean(true), ScalarType::Boolean),
            ),
            "flag <> true",
        );
        assert!(!result.may_be_true());
        assert!(result.may_be_false());
        assert!(!result.may_be_null());

        let result = read(
            ColumnContext::new().with_column(
                "flag",
                Evaluation::new(
                    ScalarType::Boolean,
                    true,
                    false,
                    Values::Unlisted(Unlisted::any()),
                ),
            ),
            "flag > true",
        );
        assert!(!result.may_raise());
        assert!(result.may_be_true());
        assert!(result.may_be_false());
    }

    #[test]
    fn an_always_null_side_raises_only_for_a_refused_pair() {
        let refused = read(ColumnContext::new(), "NULL::int = '5'::text");
        assert!(refused.may_raise());
        let untyped = read(ColumnContext::new(), "NULL = 1");
        assert!(untyped.is_always_null());
        assert!(!untyped.may_raise());
    }

    #[test]
    fn a_comparison_outside_its_category_raises() {
        let i4 = int(IntegerType::Integer);
        let result = read(
            ColumnContext::new().with_column("a", i4).with_column("b", text(TextRules::DEFAULT)),
            "a = b",
        );
        assert!(result.may_raise());
        assert!(!result.may_be_true());
        assert!(!result.may_be_false());
    }

    #[test]
    fn an_impossible_side_decides_before_resolution() {
        let impossible = Evaluation::impossible(ScalarType::Integer(IntegerType::Integer));
        let result = read(ColumnContext::new().with_column("a", impossible), "a = 1");
        assert!(!result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_be_null());
        assert!(!result.may_raise());
    }

    #[test]
    fn a_raising_side_leaves_no_pair() {
        let result = read(ColumnContext::new(), "1 / 0 = 1");
        assert!(result.may_raise());
        assert!(!result.may_be_true());
        assert!(!result.may_be_false());
        assert!(!result.may_be_null());
    }

    #[test]
    fn a_multi_value_unknown_literal_reinterprets_each_member() {
        let values =
            BTreeSet::from([PgValue::Text("1".to_string()), PgValue::Text("2".to_string())]);
        let literal =
            Evaluation::new(ScalarType::UnknownLiteral, false, false, Values::Listed(values));
        let resolved = crate::forward::fold::reinterpret_unknown(
            literal,
            &ScalarType::Integer(IntegerType::Integer),
        );
        let Values::Listed(listed) = resolved.values() else { panic!("the literal is listed") };
        assert_eq!(listed, &BTreeSet::from([PgValue::Integer(1), PgValue::Integer(2)]));
        let literal = Evaluation::new(
            ScalarType::UnknownLiteral,
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        let resolved = crate::forward::fold::reinterpret_unknown(
            literal,
            &ScalarType::Integer(IntegerType::Integer),
        );
        assert!(matches!(resolved.values(), Values::Unlisted(_)));
        assert!(resolved.may_be_null());
    }

    /// A column of the type `sql` spells, as a table definition spells it.
    fn spelled_column(sql: &str) -> Evaluation {
        let data_type = sqlparser::parser::Parser::new(&sqlparser::dialect::PostgreSqlDialect {})
            .try_with_sql(sql)
            .and_then(|mut parser| parser.parse_data_type())
            .expect("a data type");
        Evaluation::new(ScalarType::of(&data_type), true, false, Values::Unlisted(Unlisted::any()))
    }

    /// The array a context answers `ARRAY(query)` with for a query projecting
    /// `column`, never `NULL` and of unlisted values.
    fn query_array(column: &Evaluation) -> Evaluation {
        let ty = ScalarType::Array(crate::ArrayElement::of(column.ty()));
        Evaluation::new(ty, false, false, Values::Unlisted(Unlisted::any()))
    }

    /// The columns of `uuid`, `timestamptz`, `inet`, `json`, `point`,
    /// `citext`, `timestamp`, `jsonb` and a `public.inet`, and a query
    /// projecting `uuid`, which `ARRAY(SELECT p FROM t)` collects.
    fn typed_columns() -> ColumnContext {
        ColumnContext::new()
            .with_column("p", spelled_column("UUID"))
            .with_column("e", spelled_column("TIMESTAMP WITH TIME ZONE"))
            .with_column("i", spelled_column("pg_catalog.INET"))
            .with_column("j", spelled_column("JSON"))
            .with_column("g", spelled_column("POINT"))
            .with_column("c", spelled_column("citext"))
            .with_column("d", spelled_column("TIMESTAMP"))
            .with_column("b", spelled_column("JSONB"))
            .with_column("o", spelled_column("public.inet"))
            .with_query(crate::Rows { none: true, one: true, many: true }, spelled_column("UUID"))
            .with_expression("ARRAY(SELECT p FROM t)", query_array(&spelled_column("UUID")))
    }

    #[test]
    fn two_values_of_one_ordered_builtin_type_compare_without_raising() {
        for condition in [
            "p = p",
            "p <> p",
            "p < p",
            "p >= p",
            "p IS NOT DISTINCT FROM p",
            "p IS DISTINCT FROM p",
            "p IN (p, p)",
            "p NOT IN (p, p)",
            "p IN (SELECT p FROM t)",
            "p = ANY (SELECT p FROM t)",
            "p = ANY (ARRAY[p, p])",
            "p = ANY (ARRAY(SELECT p FROM t))",
            "p <> ALL (ARRAY(SELECT p FROM t))",
            "e = e",
            "e > CURRENT_TIMESTAMP",
            "i <= i",
            "b = b",
        ] {
            let result = read(typed_columns(), condition);
            assert!(!result.may_raise(), "{condition} never raises");
            assert!(result.may_be_true() && result.may_be_false(), "{condition} takes both truths");
        }
    }

    #[test]
    fn a_type_without_a_btree_order_or_a_pair_of_types_may_raise() {
        for condition in [
            "j = j",
            "g = g",
            "c = c",
            "o = o",
            "e > d",
            "p = b",
            "p = 'x'",
            "b = ANY (ARRAY(SELECT p FROM t))",
        ] {
            assert!(read(typed_columns(), condition).may_raise(), "{condition} may raise");
        }
    }

    /// An enum value against an array a query builds of its own enum and
    /// one of another enum, and arrays of the one enum and of two.
    #[test]
    fn an_array_element_keeps_its_enum_type() {
        let enumeration = |name: &str| {
            let ty = EnumType::new(name, ["note", "task"]);
            Evaluation::new(
                ScalarType::Enum(alloc::sync::Arc::new(ty)),
                true,
                false,
                Values::Unlisted(Unlisted::any()),
            )
        };
        let condition = "k = ANY (ARRAY(SELECT k FROM t))";
        let context = |element: &str| {
            ColumnContext::new()
                .with_column("k", enumeration("kinds"))
                .with_column("s", enumeration("states"))
                .with_expression("ARRAY(SELECT k FROM t)", query_array(&enumeration(element)))
        };
        let own = read(context("kinds"), condition);
        assert!(!own.may_raise(), "{condition} over its own enum never raises");
        assert!(own.may_be_true() && own.may_be_false(), "{condition} takes both truths");
        assert!(read(context("states"), condition).may_raise(), "another enum may raise");
        assert!(!read(context("kinds"), "ARRAY[k] = ARRAY[k]").may_raise());
        // PostgreSQL has no operator between arrays of two enums.
        assert!(read(context("kinds"), "ARRAY[k] = ARRAY[s]").may_raise());
    }
}
