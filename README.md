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
    dialect::GenericDialect,
    parser::Parser,
};
use sqlparser_expr_eval::{
    Callee, Context, Evaluation, Guarantees, Outcomes, Resolved,
    ScalarType, TextRules, Unlisted, Values, evaluate, implied,
    sql_values::IntegerType,
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

let dialect = GenericDialect;
let options = ();
let parse = |sql: &str| Parser::new(&dialect).try_with_sql(sql)?.parse_expr();
let read = |sql: &str| {
    let Ok(evaluation) = evaluate(&parse(sql)?, &dialect, &options, &mut Teams, &Guarantees::new());
    Ok::<_, sqlparser::parser::ParserError>(evaluation)
};

assert!(!read("id NOT IN (1, NULL)")?.may_be_true());
assert!(read("note = NULL")?.is_always_null());
assert!(read("2147483647 + 1")?.may_raise());
assert!(read("7.0 / 2")?.may_raise());

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

The generic reading keeps only what every dialect shares, three-valued logic and `NULL` propagation, and leaves values unknown. A dialect is any type implementing [`EvalDialect`], chosen at runtime, as in this engine that picks its reading when the query runs:

```rust
use sqlparser::{
    ast::{BinaryOperator, Expr, Function},
    dialect::GenericDialect,
    parser::Parser,
};
use sqlparser_expr_eval::{
    Callee, Comparison, Context, EvalDialect, Evaluation, Guarantees, Resolved, Values, evaluate,
};

/// The dialect an engine picks at runtime, reading as the generic dialect.
struct Engine;

impl EvalDialect for Engine {
    type Options = ();

    fn comparison(
        &self,
        left: Evaluation,
        op: &BinaryOperator,
        right: Evaluation,
        _options: &Self::Options,
    ) -> Comparison {
        GenericDialect.comparison(left, op, right, &())
    }

    fn arithmetic(
        &self,
        left: &Evaluation,
        op: &BinaryOperator,
        right: &Evaluation,
        _options: &Self::Options,
    ) -> Evaluation {
        GenericDialect.arithmetic(left, op, right, &())
    }
}

/// A context with no columns of its own.
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
let sum = evaluate(&parse("1 + 2")?, &Engine, &(), &mut Empty, &Guarantees::new())?;
assert!(sum.may_raise());
assert!(matches!(sum.values(), Values::Unlisted(_)));
let eq = evaluate(&parse("1 = 1")?, &Engine, &(), &mut Empty, &Guarantees::new())?;
assert!(eq.may_be_true());
let call = evaluate(&parse("length(1)")?, &Engine, &(), &mut Empty, &Guarantees::new())?;
assert!(call.may_be_null());
let open = evaluate(&parse("x = 1")?, &Engine, &(), &mut Empty, &Guarantees::new())?;
assert!(open.may_be_null());

# Ok::<(), sqlparser::parser::ParserError>(())
```
