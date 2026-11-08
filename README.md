# sqlparser-expr-eval

[![CI](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval/graph/badge.svg)](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/blob/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/sqlparser-expr-eval.svg)](https://crates.io/crates/sqlparser-expr-eval)
[![docs.rs](https://docs.rs/sqlparser-expr-eval/badge.svg)](https://docs.rs/sqlparser-expr-eval)

Reads a [`sqlparser`](https://docs.rs/sqlparser) expression into what it can evaluate to over every row: whether it can be `NULL`, which values it can take, and whether it can raise. Every answer over-approximates, so unknown is always a safe answer. Values and their PostgreSQL rules come from [`sql-values`](https://docs.rs/sql-values), re-exported as [`sql_values`]. The crate is `no_std`, needing only `alloc`.

An [`EvalDialect`] answers the questions an expression's meaning turns on, and carries its settings in `Options`. A [`Context`] resolves the columns, calls and subqueries an expression reads. The dialect and its settings are passed to the entry points; nothing the context answers depends on them.

```rust
use sqlparser::ast::{Expr, Function, FunctionArguments, Ident, ObjectName, ObjectNamePart};
use sqlparser::parser::Parser;
use sqlparser_expr_eval::{
    Callee, Context, Evaluation, Guarantees, Resolved, ScalarType, TextRules,
    Unlisted, Values, evaluate,
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

let parse = |sql: &str| {
    Parser::new(&sqlparser::dialect::GenericDialect).try_with_sql(sql)?.parse_expr()
};
let read = |sql: &str| {
    let Ok(evaluation) =
        evaluate(&parse(sql)?, &sqlparser::dialect::GenericDialect, &(), &mut Teams, &Guarantees::new());
    Ok::<_, sqlparser::parser::ParserError>(evaluation)
};

assert!(!read("id NOT IN (1, NULL)")?.may_be_true());
assert!(read("note = NULL")?.is_always_null());

// The conservative dialect leaves values unknown, but the outcome still holds.
let sum = read("1 + 2")?;
assert!(sum.may_raise());
assert!(matches!(sum.values(), Values::Unlisted(_)));

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

A dialect whose answers are the conservative ones implements no method and carries no settings in `Options`. The conservative reading keeps three-valued logic and `NULL` propagation and leaves values unknown, so an unknown routine may return `NULL` on non-`NULL` input and an arithmetic result may raise.
