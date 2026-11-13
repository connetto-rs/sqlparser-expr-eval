# sqlparser-expr-eval

[![CI](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval/graph/badge.svg)](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/blob/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/sqlparser-expr-eval.svg)](https://crates.io/crates/sqlparser-expr-eval)
[![docs.rs](https://docs.rs/sqlparser-expr-eval/badge.svg)](https://docs.rs/sqlparser-expr-eval)

Reads a [`sqlparser`](https://docs.rs/sqlparser) expression into what it can evaluate to over every row: whether it can be `NULL`, which values it can take, and whether it can raise. Every answer over-approximates, so unknown is always a safe answer. Values and their PostgreSQL rules come from [`sql-values`](https://docs.rs/sql-values), re-exported as [`sql_values`]. The crate is `no_std`, needing only `alloc`.

An [`EvalDialect`] answers the questions an expression's meaning turns on, and carries its settings in `Options`. A [`Context`] resolves the columns, calls and subqueries an expression reads. The dialect and its settings are passed to the entry points; nothing the context answers depends on them.

```rust
use sqlparser::{
    ast::{Expr, Function, FunctionArguments, Ident, ObjectName, ObjectNamePart},
    dialect::PostgreSqlDialect,
    parser::Parser,
};
use sqlparser_expr_eval::{
    Callee, Context, Evaluation, Guarantees, Outcomes, PostgreSqlOptions, Resolved,
    ScalarType, TextRules, Unlisted, Values, evaluate, implied,
    sql_values::postgres::IntegerType,
};

/// `id INT NOT NULL` and `note TEXT`.
struct Teams;

impl Context for Teams {
    type Reference = &'static str;
    type Error = core::convert::Infallible;

    fn reference(&mut self, expression: &Expr) -> Result<Option<Resolved<&'static str>>, Self::Error> {
        let Expr::Identifier(name) = expression else { return Ok(None) };
        let (reference, ty, null) = match name.value.as_str() {
            "id" => ("id", ScalarType::Integer(IntegerType::Integer), false),
            "note" => ("note", ScalarType::Text(TextRules::DEFAULT), true),
            _ => return Ok(None),
        };
        let evaluation = Evaluation::new(ty, null, false, Values::Unlisted(Unlisted::any()));
        Ok(Some(Resolved { reference, evaluation }))
    }

    fn callees(&mut self, _call: &Function) -> Result<Vec<Callee>, Self::Error> {
        Ok(Vec::new())
    }
}

let dialect = PostgreSqlDialect {};
let options = PostgreSqlOptions::default();
let parse = |sql: &str| Parser::new(&dialect).try_with_sql(sql)?.parse_expr();
let read = |sql: &str| {
    let Ok(evaluation) = evaluate(&parse(sql)?, &dialect, &options, &mut Teams, &Guarantees::new());
    Ok::<_, sqlparser::parser::ParserError>(evaluation)
};

assert!(!read("id NOT IN (1, NULL)")?.may_be_true());
assert!(read("note = NULL")?.is_always_null());
assert!(read("2147483647 + 1")?.always_raises());
assert_eq!(read("7.0 / 2")?.constant().map(ToString::to_string).as_deref(), Some("3.5000000000000000"));

let Ok(guarantees) = implied(
    &parse("note IN ('read', 'write')")?,
    Outcomes::TRUE,
    &dialect,
    &options,
    &mut Teams,
    &Guarantees::new(),
);
let note = guarantees.reference(&"note").expect("the condition restricts note");
assert!(!note.may_be_null());
assert!(matches!(note.values(), Values::Listed(listed) if listed.len() == 2));

// A name the scope does not hold and a call it does not declare admit nothing.
let mut scope = Teams;
assert!(scope
    .reference(&Expr::Identifier(Ident::new("other")))
    .unwrap()
    .is_none());
assert!(scope.callees(&Function {
    name: ObjectName(vec![ObjectNamePart::Identifier(Ident::new("length"))]),
    uses_odbc_syntax: false,
    parameters: FunctionArguments::None,
    args: FunctionArguments::None,
    within_group: vec![],
    filter: None,
    null_treatment: None,
    over: None,
})
.unwrap()
.is_empty());
# Ok::<(), sqlparser::parser::ParserError>(())
```

Under [`PostgreSqlDialect`](sqlparser::dialect::PostgreSqlDialect) the evaluator folds with PostgreSQL's literal typing, overflow, numeric scale, casts and built-ins. Every other dialect keeps only what every dialect shares, three-valued logic and `NULL` propagation, and leaves values unknown. A dialect is any type implementing [`EvalDialect`], chosen at runtime, as in this engine that runs a query under PostgreSQL or the generic reading:

```rust
use sqlparser::{
    ast::{BinaryOperator, CastKind, DataType, Expr, Function},
    dialect::{GenericDialect, PostgreSqlDialect},
    parser::Parser,
};
use sqlparser_expr_eval::{
    Callee, Comparison, Context, EvalDialect, Evaluation, Guarantees, PostgreSqlOptions, Resolved,
    ScalarType, evaluate,
};

/// The engine a query runs under, chosen at runtime, holding the settings
/// of the dialect it wraps.
enum Engine {
    Postgres(PostgreSqlDialect, PostgreSqlOptions),
    Generic(GenericDialect),
}

impl EvalDialect for Engine {
    type Options = ();

    fn comparison(
        &self,
        left: Evaluation,
        op: &BinaryOperator,
        right: Evaluation,
        _options: &(),
    ) -> Comparison {
        match self {
            Self::Postgres(dialect, options) => dialect.comparison(left, op, right, options),
            Self::Generic(dialect) => dialect.comparison(left, op, right, &()),
        }
    }

    fn concat(&self, left: &Evaluation, right: &Evaluation, _options: &()) -> Option<Evaluation> {
        match self {
            Self::Postgres(dialect, options) => dialect.concat(left, right, options),
            Self::Generic(dialect) => dialect.concat(left, right, &()),
        }
    }

    fn arithmetic(
        &self,
        left: &Evaluation,
        op: &BinaryOperator,
        right: &Evaluation,
        _options: &(),
    ) -> Evaluation {
        match self {
            Self::Postgres(dialect, options) => dialect.arithmetic(left, op, right, options),
            Self::Generic(dialect) => dialect.arithmetic(left, op, right, &()),
        }
    }

    fn unary_plus_preserves(&self, inner: &Evaluation, _options: &()) -> bool {
        match self {
            Self::Postgres(dialect, options) => dialect.unary_plus_preserves(inner, options),
            Self::Generic(dialect) => dialect.unary_plus_preserves(inner, &()),
        }
    }

    fn cast(
        &self,
        source: Evaluation,
        kind: &CastKind,
        data_type: &DataType,
        to: ScalarType,
        element_type: impl Fn(&DataType) -> ScalarType,
        _options: &(),
    ) -> Evaluation {
        match self {
            Self::Postgres(dialect, options) => {
                dialect.cast(source, kind, data_type, to, &element_type, options)
            }
            Self::Generic(dialect) => dialect.cast(source, kind, data_type, to, &element_type, &()),
        }
    }

    fn named_call(&self, name: &str, arguments: &[Evaluation], _options: &()) -> Option<Evaluation> {
        match self {
            Self::Postgres(dialect, options) => dialect.named_call(name, arguments, options),
            Self::Generic(dialect) => dialect.named_call(name, arguments, &()),
        }
    }

    fn null_equals_operand<'a>(
        &self,
        left: &'a Expr,
        right: &'a Expr,
        _options: &(),
    ) -> Option<&'a Expr> {
        match self {
            Self::Postgres(dialect, options) => dialect.null_equals_operand(left, right, options),
            Self::Generic(dialect) => dialect.null_equals_operand(left, right, &()),
        }
    }
}

/// A query with no columns of its own.
struct Empty;
impl Context for Empty {
    type Reference = ();
    type Error = sqlparser::parser::ParserError;
    fn reference(&mut self, _expression: &Expr) -> Result<Option<Resolved<()>>, Self::Error> {
        Ok(None)
    }
    fn callees(&mut self, _call: &Function) -> Result<Vec<Callee>, Self::Error> {
        Ok(Vec::new())
    }
}

let parse = |sql: &str| Parser::new(&GenericDialect).try_with_sql(sql)?.parse_expr();
let postgres = |sql: &str| {
    evaluate(
        &parse(sql)?,
        &Engine::Postgres(PostgreSqlDialect {}, PostgreSqlOptions::default()),
        &(),
        &mut Empty,
        &Guarantees::new(),
    )
};
let generic = |sql: &str| evaluate(&parse(sql)?, &Engine::Generic(GenericDialect), &(), &mut Empty, &Guarantees::new());
// The same expressions, read under the two engines.
assert!(postgres("2147483647 + 1")?.always_raises());
assert!(generic("1 + 2")?.may_raise());
assert!(postgres("1 = 1")?.may_be_true());
assert!(generic("1 = 1")?.may_be_true());
assert_eq!(postgres("'a' || 'b'")?.constant().map(ToString::to_string).as_deref(), Some("ab"));
assert!(!generic("'a' || 'b'")?.may_raise());
assert_eq!(postgres("+1")?.constant().map(ToString::to_string).as_deref(), Some("1"));
assert!(!generic("+1")?.may_be_null());
assert!(matches!(postgres("1::text")?.ty(), ScalarType::Text(_)));
assert!(matches!(generic("1::text")?.ty(), ScalarType::Text(_)));
assert!(!postgres("coalesce(1)")?.may_be_null());
assert!(generic("coalesce(1)")?.may_be_null());
assert!(postgres("1 = NULL")?.is_always_null());
assert!(generic("1 = NULL")?.is_always_null());
let open = postgres("x = 1")?;
assert!(open.may_be_null());
assert!(generic("x = 1")?.may_be_null());

# Ok::<(), sqlparser::parser::ParserError>(())
```
