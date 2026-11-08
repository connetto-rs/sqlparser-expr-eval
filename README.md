# sqlparser-expr-eval

[![CI](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval/graph/badge.svg)](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/blob/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/sqlparser-expr-eval.svg)](https://crates.io/crates/sqlparser-expr-eval)
[![docs.rs](https://docs.rs/sqlparser-expr-eval/badge.svg)](https://docs.rs/sqlparser-expr-eval)

What a SQL expression can evaluate to over every row: an [`Evaluation`] of its [`ScalarType`], its `NULL`-ness, whether it can raise, and the [`Values`] it can take, read into what the subjects an expression reads are known to satisfy by [`Guarantees`]. Every answer over-approximates, so unknown is always a safe answer. Values and their PostgreSQL rules come from [`sql-values`](https://docs.rs/sql-values), re-exported as [`sql_values`]. The crate is `no_std`, needing only `alloc`.

An [`EvalDialect`] answers the questions an expression's meaning turns on; a dialect whose answers are the conservative ones implements no method and carries no settings in `Options`. A [`Context`] resolves the columns, calls and subqueries an expression reads. The dialect and its settings are passed to the entry points; nothing the context answers depends on them.

```rust
use sqlparser::ast::DataType;
use sqlparser_expr_eval::{
    Callee, CalleeKind, Context, Evaluation, Guarantees, NullInput, QueryEvaluation,
    Resolved, Rows, ScalarType, TextRules, Unlisted, Values, Volatility,
    sql_values::IntegerType,
};
use core::convert::Infallible;

/// A scope that resolves a single `id` column.
struct Teams {
    /// Whether the scope has seen the column.
    seen: bool,
}

impl Context for Teams {
    type Reference = &'static str;
    type Error = Infallible;

    fn reference(
        &mut self,
        expression: &sqlparser::ast::Expr,
    ) -> Result<Option<Resolved<&'static str>>, Self::Error> {
        let sqlparser::ast::Expr::Identifier(name) = expression else {
            return Ok(None);
        };
        if name.value.as_str() != "id" {
            return Ok(None);
        }
        self.seen = true;
        let evaluation = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            false,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        Ok(Some(Resolved { reference: "id", evaluation }))
    }

    fn callees(&mut self, _call: &sqlparser::ast::Function) -> Result<Vec<Callee>, Self::Error> {
        Ok(Vec::new())
    }
}

let mut teams = Teams { seen: false };
let column = teams
    .reference(&sqlparser::ast::Expr::Identifier(sqlparser::ast::Ident::new("id")))
    .unwrap()
    .expect("the column resolves");
assert!(teams.seen);
assert!(!column.evaluation.may_be_null());

// An expression that is not an identifier and a name the scope does not
// hold admit no column.
let unbound = teams
    .reference(&sqlparser::ast::Expr::Value(
        sqlparser::ast::Value::Number("1".to_string(), false).into(),
    ))
    .unwrap();
assert!(unbound.is_none());
assert!(teams
    .reference(&sqlparser::ast::Expr::Identifier(sqlparser::ast::Ident::new("name")))
    .unwrap()
    .is_none());

// The scope declares no callee.
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
assert!(teams.callees(&call).unwrap().is_empty());

// A callee the scope declares, with what a `NULL` argument does to it.
let callee = Callee::new(
    None,
    NullInput::Strict,
    Volatility::Immutable,
    CalleeKind::Normal,
    ScalarType::Text(TextRules::DEFAULT),
    false,
    0,
);
assert_eq!(callee.returns, ScalarType::Text(TextRules::DEFAULT));

// A query the scope cannot answer reads every row count and may raise.
let reading = QueryEvaluation::unknown();
assert_eq!(reading.rows, Rows::ANY);
assert!(reading.error);

// The type a cast spells.
let integer = ScalarType::of(&DataType::Integer(None));
assert_eq!(integer, ScalarType::Integer(IntegerType::Integer));

// The guarantees are empty until a condition narrows a subject.
let guarantees: Guarantees<&'static str> = Guarantees::new();
assert!(guarantees.is_empty());
```
