# sqlparser-expr-eval

[![CI](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval/graph/badge.svg)](https://codecov.io/gh/connetto-rs/sqlparser-expr-eval)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/connetto-rs/sqlparser-expr-eval/blob/main/LICENSE)
[![crates.io](https://img.shields.io/crates/v/sqlparser-expr-eval.svg)](https://crates.io/crates/sqlparser-expr-eval)
[![docs.rs](https://docs.rs/sqlparser-expr-eval/badge.svg)](https://docs.rs/sqlparser-expr-eval)

The scalar types a SQL expression is read as: a [`ScalarType`], with the [`TextRules`] a text collation allows, over the values and types of [`sql-values`](https://docs.rs/sql-values), re-exported as [`sql_values`]. The crate is `no_std`, needing only `alloc`.

```rust
use sqlparser::ast::DataType;
use sqlparser_expr_eval::{ScalarType, TextRules, sql_values::IntegerType};

let bigint = ScalarType::of(&DataType::BigInt(None));
assert_eq!(bigint, ScalarType::Integer(IntegerType::Bigint));
assert_eq!(bigint.integer_bounds(), Some((i64::MIN, i64::MAX)));
assert!(ScalarType::Text(TextRules::DEFAULT).is_text());
```
