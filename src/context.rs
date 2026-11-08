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
    use alloc::collections::BTreeSet;
    use core::{convert::Infallible, ops::Bound};

    use sql_values::{IntegerType, PgNumeric};
    use sqlparser::{
        ast::{Expr, Function, Statement},
        dialect::GenericDialect,
        parser::Parser,
    };

    use super::*;
    use crate::{
        CalleeKind, Evaluation, Interval, NullInput, ScalarType, TextRules, Values, Volatility,
        guarantees::Guarantees,
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
            _expression: &Expr,
        ) -> Result<Option<Resolved<&'static str>>, Infallible> {
            Ok(None)
        }

        fn callees(&mut self, _call: &Function) -> Result<Vec<Callee>, Infallible> {
            Ok(Vec::new())
        }
    }

    #[test]
    fn the_default_bodies_answer_conservatively() {
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
        let argument = Evaluation::new(
            ScalarType::Text(TextRules::DEFAULT),
            true,
            false,
            Values::Unlisted(crate::domain::Unlisted::any()),
        );
        let mut context = Minimal;
        assert!(context.evaluate_body(&callee, &[argument]).unwrap().is_none());
        let statement = Parser::parse_sql(&GenericDialect, "SELECT 1").unwrap().pop().unwrap();
        let Statement::Query(query) = statement else { panic!("the statement is a query") };
        let reading = context.query(&query, &Guarantees::new()).unwrap();
        assert_eq!(reading.rows, Rows::ANY);
        assert!(reading.columns.is_none());
        assert!(reading.error);
        assert_eq!(QueryEvaluation::unknown().rows, reading.rows);
        assert_eq!(QueryEvaluation::unknown().error, reading.error);
        let type_of = context.data_type(&sqlparser::ast::DataType::Integer(None));
        assert_eq!(type_of, ScalarType::Integer(IntegerType::Integer));
        let name = sqlparser::ast::Expr::Identifier(sqlparser::ast::Ident::new("name"));
        assert!(context.reference(&name).unwrap().is_none());
        let value = sqlparser::ast::Expr::Value(
            sqlparser::ast::Value::Number("1".to_string(), false).into(),
        );
        assert!(context.reference(&value).unwrap().is_none());
        let call = sqlparser::ast::Function {
            name: sqlparser::ast::ObjectName(vec![sqlparser::ast::ObjectNamePart::Identifier(
                sqlparser::ast::Ident::new("length"),
            )]),
            uses_odbc_syntax: false,
            parameters: sqlparser::ast::FunctionArguments::None,
            args: sqlparser::ast::FunctionArguments::None,
            within_group: vec![],
            filter: None,
            null_treatment: None,
            over: None,
        };
        assert!(context.callees(&call).unwrap().is_empty());
    }

    #[test]
    fn the_rows_constructor_keeps_each_row_count() {
        let rows = Rows::new(true, false, true);
        assert!(rows.none);
        assert!(!rows.one);
        assert!(rows.many);
    }

    #[test]
    fn the_integer_normalisation_drops_the_out_of_range_unlisted() {
        let beyond = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::none(),
        )
        .with_values(Values::Unlisted(crate::domain::Unlisted {
            excluded: BTreeSet::new(),
            numbers: Interval {
                low: Bound::Included(PgNumeric::from(1_i64 << 40)),
                high: Bound::Included(PgNumeric::from((1_i64 << 40) + 5)),
            },
            length: Interval::unbounded(),
        }));
        assert!(beyond.is_impossible());
    }

    #[test]
    fn the_query_constructor_holds_its_fields() {
        let reading = QueryEvaluation::new(
            Rows::ONE,
            Some(vec![Evaluation::unknown(ScalarType::Boolean)]),
            true,
        );
        assert_eq!(reading.rows, Rows::ONE);
        assert!(matches!(reading.columns.as_deref().unwrap()[0].ty(), ScalarType::Boolean));
        assert!(reading.error);
    }
}
