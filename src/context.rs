//! What the evaluator asks of the scope an expression is read in.

use alloc::vec::Vec;
use core::fmt::Debug;

use sqlparser::ast::{DataType, Expr, Function, Query};

use super::{Evaluation, Guarantees, ScalarType};

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

/// How a function's result may change between calls with equal arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// The PostgreSQL built-in functions whose results the evaluator folds.
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
    /// Whether equal arguments give equal results.
    pub volatility: Volatility,
    /// Ordinary, aggregate or window.
    pub kind: CalleeKind,
    /// The declared result type.
    pub returns: ScalarType,
    /// Whether the function returns a set of rows.
    pub returns_set: bool,
    /// A token the context reads back in [`Context::evaluate_body`].
    pub token: usize,
}

impl Callee {
    /// The callee with its declared reading.
    #[must_use]
    pub fn new(
        builtin: Option<Builtin>,
        null_input: NullInput,
        volatility: Volatility,
        kind: CalleeKind,
        returns: ScalarType,
        returns_set: bool,
        token: usize,
    ) -> Self {
        Self { builtin, null_input, volatility, kind, returns, returns_set, token }
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
/// The context only resolves names, calls and subqueries. The dialect an
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

    /// The type a cast or a typed literal names.
    fn data_type(&self, data_type: &DataType) -> ScalarType {
        ScalarType::of(data_type)
    }
}

#[cfg(test)]
mod tests {
    use core::convert::Infallible;

    use sql_values::postgres::IntegerType;
    use sqlparser::{
        ast::{Expr, Function, Ident, ObjectNamePart},
        dialect::GenericDialect,
    };

    use super::*;
    use crate::{
        CalleeKind, Evaluation, NullInput, ScalarType, TextRules, Values, Volatility,
        forward::evaluate, guarantees::Guarantees, test_context::parse,
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
                volatility: Volatility::Immutable,
                kind: CalleeKind::Normal,
                returns: ScalarType::Integer(IntegerType::Integer),
                returns_set: false,
                token: 0,
            }]))
        }
    }

    #[test]
    fn a_call_with_no_body_reads_the_callee() {
        let evaluation =
            evaluate(&parse("f('a')"), &GenericDialect, &(), &mut Minimal, &Guarantees::new())
                .unwrap();
        assert_eq!(evaluation.ty(), ScalarType::Integer(IntegerType::Integer));
        // An unknown routine may return `NULL` on non-`NULL` input, and `f`
        // is declared not to raise
        assert!(evaluation.may_be_null());
        assert!(!evaluation.may_raise());
        assert!(matches!(evaluation.values(), Values::Unlisted(_)));
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
        assert_eq!(evaluation.ty(), ScalarType::Text(TextRules::DEFAULT));
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

    #[test]
    fn a_column_and_an_unnamed_call_read_their_context() {
        // `x` resolves to the column the context spells
        let evaluation =
            evaluate(&parse("x"), &GenericDialect, &(), &mut Minimal, &Guarantees::new()).unwrap();
        assert_eq!(evaluation.ty(), ScalarType::Text(TextRules::DEFAULT));
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
