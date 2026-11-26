//! What the evaluator asks of the scope an expression is read in.

use alloc::vec::Vec;
use core::fmt::Debug;

use sqlparser::ast::{DataType, Expr, Function, ObjectName, Query};

use super::{Evaluation, Guarantees, ScalarType, TextRules};

/// A reference the context resolved, with what it can hold.
#[derive(Debug, Clone)]
pub struct Resolved<R> {
    /// The identity guarantees about the reference are keyed by.
    pub reference: R,
    /// What the reference can hold before any guarantee narrows it.
    pub evaluation: Evaluation,
}

/// What a `NULL` argument does to a call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NullInput {
    /// Any `NULL` argument makes the call `NULL`.
    Strict,
    /// The function runs on `NULL` input, and a `NULL` at one of these
    /// argument positions makes the call `NULL`.
    CalledOnNull(Vec<usize>),
}

impl NullInput {
    /// Whether a `NULL` argument at `position` makes the call `NULL`.
    pub(crate) fn nulls_at(&self, position: usize) -> bool {
        match self {
            Self::Strict => true,
            Self::CalledOnNull(positions) => positions.contains(&position),
        }
    }
}

/// Where a `NULL` result of a call comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullResult {
    /// Only from a `NULL` argument at a position [`NullInput`] names, so a
    /// call with no such argument never returns `NULL`.
    OnlyOnNullInput,
    /// From any input, `NULL` or not.
    AnyInput,
}

/// How a function's result may change between calls with equal arguments,
/// ordered from the least volatile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Volatility {
    /// Fixed by the arguments.
    Immutable,
    /// Fixed within one statement.
    Stable,
    /// May change at every call.
    Volatile,
}

/// The kind of routine a call reaches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CalleeKind {
    /// An ordinary function.
    Normal,
    /// An aggregate.
    Aggregate,
    /// A window function.
    Window,
}

/// The PostgreSQL built-in functions whose results the evaluator folds, each
/// read as raising only where its fold does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Builtin {
    /// `length(text)`, `char_length`, `character_length`: characters.
    Length,
    /// `octet_length(text)`: bytes in the database encoding.
    OctetLength,
    /// `lower(text)`.
    Lower,
    /// `upper(text)`.
    Upper,
    /// `btrim(text[, text])`.
    Btrim,
    /// `abs` over integers and numeric.
    Abs,
    /// `concat("any", ...)`.
    Concat,
    /// `concat_ws(text, "any", ...)`.
    ConcatWs,
    /// `count`.
    Count,
}

/// One function a call may reach.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct Callee {
    /// The built-in this is, when the evaluator knows how to fold it.
    pub builtin: Option<Builtin>,
    /// What a `NULL` argument does.
    pub null_input: NullInput,
    /// Where a `NULL` result comes from.
    pub null_result: NullResult,
    /// Whether equal arguments give equal results.
    pub volatility: Volatility,
    /// Ordinary, aggregate or window.
    pub kind: CalleeKind,
    /// The declared result type.
    pub returns: ScalarType,
    /// Whether the function returns a set of rows.
    pub returns_set: bool,
    /// Whether running the function may raise an error. The evaluator
    /// tightens it with what it reads of a builtin and of a body
    /// [`Context::evaluate_body`] answers.
    pub error: bool,
    /// Whether the planner may inline the function's body into the call and
    /// fold the constants of the inlined body before the call runs, which
    /// PostgreSQL does for a `LANGUAGE sql` function.
    pub inlinable: bool,
    /// A token the context reads back in [`Context::evaluate_body`].
    pub token: usize,
}

impl Callee {
    /// The callee with its declared reading.
    #[must_use]
    #[expect(
        clippy::too_many_arguments,
        reason = "every field is a separate fact a context states, and the struct is non-exhaustive"
    )]
    pub fn new(
        builtin: Option<Builtin>,
        null_input: NullInput,
        null_result: NullResult,
        volatility: Volatility,
        kind: CalleeKind,
        returns: ScalarType,
        returns_set: bool,
        error: bool,
        inlinable: bool,
        token: usize,
    ) -> Self {
        Self {
            builtin,
            null_input,
            null_result,
            volatility,
            kind,
            returns,
            returns_set,
            error,
            inlinable,
            token,
        }
    }

    /// Whether a call writing `arity` arguments may return `NULL` while
    /// every argument at a position [`NullInput`] names is non-`NULL`. A
    /// position past the written arguments names a defaulted one, whose
    /// value the call does not show.
    pub(crate) fn nulls_on_non_null_arguments(&self, arity: usize) -> bool {
        match (self.null_result, &self.null_input) {
            (NullResult::AnyInput, _) => true,
            (NullResult::OnlyOnNullInput, NullInput::Strict) => false,
            (NullResult::OnlyOnNullInput, NullInput::CalledOnNull(positions)) => {
                positions.iter().any(|&position| position >= arity)
            }
        }
    }
}

/// Which row counts a query may return.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct Rows {
    /// Whether it may return no row.
    pub none: bool,
    /// Whether it may return exactly one row.
    pub one: bool,
    /// Whether it may return more than one row.
    pub many: bool,
}

impl Rows {
    /// The row counts a query may return.
    #[must_use]
    pub fn new(none: bool, one: bool, many: bool) -> Self {
        Self { none, one, many }
    }

    /// Any number of rows.
    pub const ANY: Self = Self { none: true, one: true, many: true };
    /// No row at all.
    pub const NONE: Self = Self { none: true, one: false, many: false };
    /// Exactly one row.
    pub const ONE: Self = Self { none: false, one: true, many: false };
}

/// What a query can return.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct QueryEvaluation {
    /// The row counts it may return.
    pub rows: Rows,
    /// What each output column can hold over the rows it returns, `None`
    /// when the outputs are not enumerated.
    pub columns: Option<Vec<Evaluation>>,
    /// Whether running it may raise an error.
    pub error: bool,
}

impl QueryEvaluation {
    /// The query's reading, with its row counts, output columns and error.
    #[must_use]
    pub fn new(rows: Rows, columns: Option<Vec<Evaluation>>, error: bool) -> Self {
        Self { rows, columns, error }
    }

    /// A query about which nothing is known.
    #[must_use]
    pub fn unknown() -> Self {
        Self { rows: Rows::ANY, columns: None, error: true }
    }
}

/// The scope an expression is evaluated in.
///
/// The context resolves names, calls and subqueries, and may answer a whole
/// expression it knows better than its parts tell. The dialect an
/// expression is evaluated under is passed to the entry points, and nothing
/// the context answers depends on it.
pub trait Context {
    /// The identity of a resolved reference.
    type Reference: Clone + Ord + Debug;
    /// The error resolution may raise.
    type Error;

    /// Resolves an identifier, a compound identifier or a positional
    /// parameter, `None` when the scope holds no such name.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn reference(
        &mut self,
        expression: &Expr,
    ) -> Result<Option<Resolved<Self::Reference>>, Self::Error>;

    /// The functions `call` may reach, empty when none is known.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn callees(&mut self, call: &Function) -> Result<Vec<Callee>, Self::Error>;

    /// Evaluates the body of a `LANGUAGE sql` callee with its parameters
    /// bound to `arguments`, `None` when the body is not read.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn evaluate_body(
        &mut self,
        callee: &Callee,
        arguments: &[Evaluation],
    ) -> Result<Option<Evaluation>, Self::Error> {
        let _ = (callee, arguments);
        Ok(None)
    }

    /// What `expression` evaluates to when the context knows it better than
    /// its parts tell, `None` to evaluate it as written. The evaluator asks
    /// about every node but a reference, with its parentheses stripped, and
    /// the answer stands for the node and its operands, so the context
    /// answers for what they would raise. The answer is met with the
    /// guarantee the node's subject satisfies.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn expression(&mut self, expression: &Expr) -> Result<Option<Evaluation>, Self::Error> {
        let _ = expression;
        Ok(None)
    }

    /// What `query`, nested in the expression, can return when the outer
    /// references it reads satisfy `guarantees`.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn query(
        &mut self,
        query: &Query,
        guarantees: &Guarantees<Self::Reference>,
    ) -> Result<QueryEvaluation, Self::Error> {
        let _ = (query, guarantees);
        Ok(QueryEvaluation::unknown())
    }

    /// The type a cast or a typed literal names. A context that knows the
    /// schema answers [`ScalarType::Enum`] for an enum type, which the
    /// parser reads as a custom type.
    fn data_type(&self, data_type: &DataType) -> ScalarType {
        ScalarType::of(data_type)
    }

    /// Whether a cast or a typed literal of `data_type` checks the value it
    /// converts to, as one to a PostgreSQL domain with a constraint, or to
    /// an array of such a domain, does. The planner never folds such a cast,
    /// unless it casts an untyped literal to an array, whose input function
    /// runs the elements' checks as the literal is parsed.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn cast_checks(&mut self, data_type: &DataType) -> Result<bool, Self::Error> {
        let _ = data_type;
        Ok(false)
    }

    /// What the checks of `data_type` make of `converted`, the value a cast
    /// or a typed literal converts to, asked only when
    /// [`cast_checks`](Self::cast_checks) answers `true`. A PostgreSQL
    /// context reads a domain's constraints over `converted`, `NULL`
    /// included, and answers a raise wherever one may refuse it. The
    /// default may refuse any value. The evaluator keeps what the
    /// conversion raises.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn checked_cast(
        &mut self,
        data_type: &DataType,
        converted: Evaluation,
    ) -> Result<Evaluation, Self::Error> {
        let _ = data_type;
        Ok(converted.with_error(true))
    }

    /// The rules of the collation `name` names, under the identity a column
    /// declared with it carries, [`TextRules::UNKNOWN`] when the context
    /// cannot identify it. The evaluator marks them
    /// [`explicit`](TextRules::explicit) itself.
    ///
    /// # Errors
    ///
    /// Returns the context's error when resolution fails.
    fn collation(&mut self, name: &ObjectName) -> Result<TextRules, Self::Error> {
        let _ = name;
        Ok(TextRules::UNKNOWN)
    }

    /// Whether the engine may plan `reference` as a constant, which a strict
    /// operator over a `NULL` one folds to `NULL` before its other operands
    /// run. A PostgreSQL context answers `true` for a function parameter and
    /// a PL/pgSQL variable, which a custom plan or an inlined SQL function
    /// binds as a constant.
    fn may_plan_as_constant(&self, reference: &Self::Reference) -> bool {
        let _ = reference;
        false
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use sql_values::postgres::IntegerType;
    use sqlparser::{
        ast::{Expr, Function, Ident, ObjectNamePart},
        dialect::{GenericDialect, PostgreSqlDialect},
    };

    use super::*;
    use crate::{
        CalleeKind, Evaluation, NullInput, PostgreSqlOptions, ScalarType, TextRules, Values,
        Volatility, forward::evaluate, guarantees::Guarantees, test_context::parse,
    };

    /// A context that answers only `reference` and `callees`, so the default
    /// bodies of the remaining methods carry the reading.
    #[derive(Default)]
    struct Minimal;

    impl Context for Minimal {
        type Reference = &'static str;
        type Error = Infallible;

        fn reference(
            &mut self,
            expression: &Expr,
        ) -> Result<Option<Resolved<&'static str>>, Infallible> {
            match expression {
                Expr::Identifier(Ident { value, .. }) if value.as_str() == "x" => {
                    Ok(Some(Resolved {
                        reference: "x",
                        evaluation: Evaluation::unknown(ScalarType::Text(TextRules::DEFAULT)),
                    }))
                }
                _ => Ok(None),
            }
        }

        fn callees(&mut self, call: &Function) -> Result<Vec<Callee>, Infallible> {
            let Some(name) = call.name.0.last().and_then(ObjectNamePart::as_ident) else {
                return Ok(Vec::new());
            };
            if name.value.as_str() != "f" {
                return Ok(Vec::new());
            }
            Ok(Vec::from([Callee {
                builtin: None,
                null_input: NullInput::Strict,
                null_result: NullResult::AnyInput,
                volatility: Volatility::Immutable,
                kind: CalleeKind::Normal,
                returns: ScalarType::Integer(IntegerType::Integer),
                returns_set: false,
                error: false,
                inlinable: false,
                token: 0,
            }]))
        }
    }

    #[test]
    fn a_call_with_no_body_reads_the_callee() {
        let evaluation =
            evaluate(&parse("f('a')"), &GenericDialect, &(), &mut Minimal, &Guarantees::new())
                .unwrap();
        assert_eq!(*evaluation.ty(), ScalarType::Integer(IntegerType::Integer));
        // An unknown routine may return `NULL` on non-`NULL` input, and `f`
        // is declared not to raise
        assert!(evaluation.may_be_null());
        assert!(!evaluation.may_raise());
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    }

    /// A reference plans at run time by default, so it never folds a strict
    /// operator to `NULL` beside an operand that always raises.
    #[test]
    fn a_reference_plans_at_run_time_by_default() {
        let evaluation = evaluate(
            &parse("x || (CASE WHEN x IS NULL THEN 1 ELSE 2 END / 0)"),
            &PostgreSqlDialect {},
            &PostgreSqlOptions::default(),
            &mut Minimal,
            &Guarantees::new(),
        )
        .unwrap();
        assert!(evaluation.always_raises());
    }

    /// A context that names no collation leaves an explicit one unknown,
    /// over text and over a text array alike.
    #[test]
    fn an_unnamed_collation_is_unknown() {
        let explicit = TextRules { explicit: true, ..TextRules::UNKNOWN };
        for (sql, ty) in [
            (r#"x COLLATE "C""#, ScalarType::Text(explicit)),
            (r#"ARRAY[x] COLLATE "C""#, ScalarType::Array(crate::ArrayElement::Text(explicit))),
        ] {
            let evaluation =
                evaluate(&parse(sql), &GenericDialect, &(), &mut Minimal, &Guarantees::new())
                    .unwrap();
            assert_eq!(*evaluation.ty(), ty, "{sql}");
        }
    }

    #[test]
    fn a_nested_query_reads_unknown() {
        let evaluation = evaluate(
            &parse("EXISTS (SELECT 1)"),
            &GenericDialect,
            &(),
            &mut Minimal,
            &Guarantees::new(),
        )
        .unwrap();
        // The default query reads a row count of any, so `EXISTS` may be
        // true and false, never `NULL`, and the read may raise
        assert!(evaluation.may_raise());
        assert!(!evaluation.may_be_null());
        assert_eq!(
            evaluation.values(),
            &Values::Listed(alloc::collections::BTreeSet::from([
                sql_values::postgres::PgValue::Boolean(true),
                sql_values::postgres::PgValue::Boolean(false)
            ]))
        );
    }

    #[test]
    fn a_cast_reads_the_default_type() {
        let evaluation =
            evaluate(&parse("1::text"), &GenericDialect, &(), &mut Minimal, &Guarantees::new())
                .unwrap();
        assert_eq!(*evaluation.ty(), ScalarType::Text(TextRules::DEFAULT));
        assert!(!evaluation.is_always_null());
        // The default cast is conservative and may raise on a type change
        assert!(evaluation.may_raise());
        // A `NULL` source keeps its values absent
        let evaluation =
            evaluate(&parse("NULL::text"), &GenericDialect, &(), &mut Minimal, &Guarantees::new())
                .unwrap();
        assert!(evaluation.is_always_null());
        assert!(!evaluation.may_raise());
    }

    /// A context whose every type checks its values, reading none of the
    /// checks.
    struct Checking;

    impl Context for Checking {
        type Reference = &'static str;
        type Error = Infallible;

        fn reference(
            &mut self,
            _expression: &Expr,
        ) -> Result<Option<Resolved<&'static str>>, Infallible> {
            Ok(None)
        }

        fn callees(&mut self, _call: &Function) -> Result<Vec<Callee>, Infallible> {
            Ok(Vec::new())
        }

        fn cast_checks(&mut self, _data_type: &DataType) -> Result<bool, Infallible> {
            Ok(true)
        }
    }

    /// A type whose checks the context does not read may refuse any value,
    /// `NULL` included.
    #[test]
    fn an_unread_check_may_refuse_any_value() {
        for sql in ["5::int", "NULL::int", "x::int", "f(1)::int"] {
            let evaluation = evaluate(
                &parse(sql),
                &PostgreSqlDialect {},
                &PostgreSqlOptions::default(),
                &mut Checking,
                &Guarantees::new(),
            )
            .unwrap();
            assert!(evaluation.may_raise() && !evaluation.always_raises(), "{sql}");
        }
    }

    #[test]
    fn a_column_and_an_unnamed_call_read_their_context() {
        // `x` resolves to the column the context spells
        let evaluation =
            evaluate(&parse("x"), &GenericDialect, &(), &mut Minimal, &Guarantees::new()).unwrap();
        assert_eq!(*evaluation.ty(), ScalarType::Text(TextRules::DEFAULT));
        assert!(evaluation.may_be_null());
        // `y` resolves to nothing, so the node reads unknown
        let evaluation =
            evaluate(&parse("y"), &GenericDialect, &(), &mut Minimal, &Guarantees::new()).unwrap();
        assert!(matches!(evaluation.ty(), ScalarType::Unknown | ScalarType::UnknownLiteral));
        // A call the context does not know keeps any value of an unknown type
        let evaluation =
            evaluate(&parse("g('a')"), &GenericDialect, &(), &mut Minimal, &Guarantees::new())
                .unwrap();
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
        // A call whose name the context cannot read as an identifier reaches
        // no callee at all
        let call = Expr::Function(sqlparser::ast::Function {
            name: sqlparser::ast::ObjectName(vec![sqlparser::ast::ObjectNamePart::Function(
                sqlparser::ast::ObjectNamePartFunction { name: Ident::new("f"), args: Vec::new() },
            )]),
            uses_odbc_syntax: false,
            parameters: sqlparser::ast::FunctionArguments::None,
            args: sqlparser::ast::FunctionArguments::None,
            within_group: Vec::new(),
            filter: None,
            null_treatment: None,
            over: None,
        });
        let evaluation =
            evaluate(&call, &GenericDialect, &(), &mut Minimal, &Guarantees::new()).unwrap();
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
    }

    #[test]
    fn the_rows_constructor_keeps_each_row_count() {
        let rows = Rows::new(true, false, true);
        assert!(rows.none);
        assert!(!rows.one);
        assert!(rows.many);
    }
}
