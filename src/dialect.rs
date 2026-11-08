//! The dialect an expression is evaluated under, with the settings that
//! change its answers.

use core::fmt::Debug;

use sqlparser::ast::{
    BinaryOperator, CastKind, CharacterLength, DataType, ExactNumberInfo, Expr, Function,
};

use crate::{
    Callee, Comparison, Evaluation, ScalarType, Unlisted, Values,
    forward::fold::{arithmetic_type, interval_contains_zero, standard_comparison},
    support::number_bounds,
};

/// The behaviour an expression takes under one SQL dialect.
///
/// Every method answers one question the forward and backward passes ask of
/// the dialect, and the default bodies keep only what every dialect shares,
/// so a dialect that implements no method is the conservative reading. The
/// dialect's [`Options`](EvalDialect::Options) carries the settings that
/// change its answers.
pub trait EvalDialect {
    /// The settings of this dialect that change what an expression evaluates
    /// to.
    type Options: Default + Clone + Debug + PartialEq;

    /// The comparison of two already-evaluated operands, as the kinds of
    /// answer a left value may give against the right values and whether it
    /// may raise. The outcomes follow from the kinds and the operands' `NULL`.
    fn comparison(
        &self,
        left: Evaluation,
        op: &BinaryOperator,
        right: Evaluation,
        options: &Self::Options,
    ) -> Comparison {
        let _ = options;
        let error = left.may_raise() || right.may_raise();
        standard_comparison(&left, op, &right, error)
    }

    /// The result of `left || right` when the dialect folds it, `None` when
    /// the values stay unknown.
    fn concat(
        &self,
        left: &Evaluation,
        right: &Evaluation,
        options: &Self::Options,
    ) -> Option<Evaluation> {
        let _ = (left, right, options);
        None
    }

    /// The result of `left op right` on the arithmetic operands, folded or
    /// bounded by the dialect's rules.
    fn arithmetic(
        &self,
        left: &Evaluation,
        op: &BinaryOperator,
        right: &Evaluation,
        options: &Self::Options,
    ) -> Evaluation {
        let _ = options;
        let ty = arithmetic_type(left.ty(), right.ty());
        let divisor_may_be_zero = matches!(op, BinaryOperator::Divide | BinaryOperator::Modulo)
            && interval_contains_zero(&number_bounds(right.values()));
        let null = left.may_be_null() || right.may_be_null();
        Evaluation::new(ty, null || divisor_may_be_zero, true, Values::Unlisted(Unlisted::any()))
    }

    /// Whether `+inner` keeps `inner`'s values, as a sign over a number does.
    fn unary_plus_preserves(&self, inner: &Evaluation, options: &Self::Options) -> bool {
        let _ = (inner, options);
        false
    }

    /// The result of a cast of `source` to `to`, `element_type` naming the
    /// type a cast's array element spells.
    fn cast(
        &self,
        source: Evaluation,
        kind: &CastKind,
        data_type: &DataType,
        to: ScalarType,
        element_type: impl Fn(&DataType) -> ScalarType,
        options: &Self::Options,
    ) -> Evaluation {
        let _ = (element_type, options);
        // A non-strict cast may fail and answer `NULL`, or raise for a cast
        // the engine does not permit.
        if !matches!(kind, CastKind::Cast | CastKind::DoubleColon) {
            return Evaluation::unknown(to);
        }
        if source.is_always_null() {
            // A typed `NULL` may name a cast that does not exist.
            let typed = !matches!(source.ty(), ScalarType::Unknown | ScalarType::UnknownLiteral);
            let refused = typed && to != source.ty();
            return Evaluation::null(to).with_error(source.may_raise() || refused);
        }
        // A `numeric(p, s)` typmod, a narrower `float` or a `varchar(0)` may
        // refuse the value.
        let bounded = to == ScalarType::Float
            || matches!(
                data_type,
                DataType::Numeric(info) | DataType::Decimal(info) | DataType::Dec(info)
                    if *info != ExactNumberInfo::None
            )
            || matches!(
                data_type,
                DataType::Varchar(Some(CharacterLength::IntegerLength { length: 0, .. }))
                    | DataType::CharacterVarying(Some(CharacterLength::IntegerLength {
                        length: 0,
                        ..
                    }))
                    | DataType::CharVarying(Some(CharacterLength::IntegerLength { length: 0, .. }))
            );
        let error = source.may_raise() || to != source.ty() || bounded;
        source.with_type(to).with_values(Values::Unlisted(Unlisted::any())).with_error(error)
    }

    /// The result of a call the dialect recognises by name before the context
    /// is asked.
    fn named_call(
        &self,
        name: &str,
        arguments: &[Evaluation],
        options: &Self::Options,
    ) -> Option<Evaluation> {
        let _ = (name, arguments, options);
        None
    }

    /// The operand of `left = right` when the dialect reads the form as
    /// `operand IS NULL`, `None` when no side is a literal `NULL`.
    fn null_equals_operand<'a>(
        &self,
        left: &'a Expr,
        right: &'a Expr,
        options: &Self::Options,
    ) -> Option<&'a Expr> {
        let _ = (left, right, options);
        None
    }

    /// Whether a `NULL` result of a call that may reach `callees` comes from
    /// a `NULL` at one of the arguments that return `NULL`.
    fn null_only_on_null_input(&self, callees: &[Callee], options: &Self::Options) -> bool {
        let _ = (callees, options);
        false
    }

    /// The position of the argument a `length`-like call that may reach
    /// `callees` counts the characters of.
    fn counted_length_argument(
        &self,
        call: &Function,
        callees: &[Callee],
        options: &Self::Options,
    ) -> Option<usize> {
        let _ = (call, callees, options);
        None
    }
}

/// Implements [`EvalDialect`] with the default bodies and no settings on a
/// dialect whose answers are the conservative ones.
macro_rules! default_dialect {
    ($($dialect:ty),* $(,)?) => {
        $(
            impl crate::dialect::EvalDialect for $dialect {
                type Options = ();
            }
        )*
    };
}

default_dialect! {
    sqlparser::dialect::AnsiDialect,
    sqlparser::dialect::BigQueryDialect,
    sqlparser::dialect::ClickHouseDialect,
    sqlparser::dialect::DatabricksDialect,
    sqlparser::dialect::DuckDbDialect,
    sqlparser::dialect::GenericDialect,
    sqlparser::dialect::HiveDialect,
    sqlparser::dialect::MsSqlDialect,
    sqlparser::dialect::MySqlDialect,
    sqlparser::dialect::OracleDialect,
    sqlparser::dialect::RedshiftSqlDialect,
    sqlparser::dialect::SnowflakeDialect,
    sqlparser::dialect::SparkSqlDialect,
    sqlparser::dialect::SQLiteDialect,
    sqlparser::dialect::TeradataDialect,
}
