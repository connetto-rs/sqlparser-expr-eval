//! The forward and backward passes over PostgreSQL expressions that grow in
//! the shapes real conditions take.
#![expect(
    clippy::expect_used,
    clippy::panic,
    reason = "a benchmark whose input fails to parse or resolve measures nothing"
)]

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sqlparser::dialect::PostgreSqlDialect;
use sqlparser_expr_eval::{
    Builtin, Guarantees, Outcomes, PostgreSqlOptions, ScalarType, TextRules, evaluate, implied,
    sql_values::postgres::IntegerType,
};

#[path = "../tests/common/mod.rs"]
mod common;

use common::{Ctx, builtin_callee, column, parse};

const INTEGER: ScalarType = ScalarType::Integer(IntegerType::Integer);
const TEXT: ScalarType = ScalarType::Text(TextRules::DEFAULT);

/// The sizes an expression whose cost grows linearly is measured at.
const LINEAR: &[usize] = &[8, 64, 512];
/// The sizes an expression whose cost grows faster is measured at, kept
/// small enough for one run to finish in minutes.
const STEEP: &[usize] = &[8, 16, 32, 64];

/// A context with a nullable integer `x`, a nullable text `s`, the integers
/// `c0` to `c{count - 1}` that are never `NULL`, and the text built-ins.
fn context(count: usize) -> Ctx {
    let mut columns = vec![("x".into(), column(INTEGER, true)), ("s".into(), column(TEXT, true))];
    columns.extend((0..count).map(|index| (format!("c{index}"), column(INTEGER, false))));
    Ctx::with_columns(columns)
        .with_callee("length", builtin_callee(Builtin::Length, INTEGER))
        .with_callee("lower", builtin_callee(Builtin::Lower, TEXT))
}

/// The `size` terms `term(0)` to `term(size - 1)` joined by `separator`.
fn joined(size: usize, separator: &str, term: impl Fn(usize) -> String) -> String {
    (0..size).map(term).collect::<Vec<_>>().join(separator)
}

/// Measures `evaluate` on each expression of `group`, built from each of
/// `sizes`.
fn forward_group(
    criterion: &mut Criterion,
    group: &str,
    sizes: &[usize],
    sql: impl Fn(usize) -> String,
) {
    let mut group = criterion.benchmark_group(group);
    for &size in sizes {
        let expression = parse(&sql(size));
        let mut context = context(0);
        let guarantees = Guarantees::new();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &expression,
            |bencher, expression| {
                bencher.iter(|| {
                    evaluate(
                        black_box(expression),
                        &PostgreSqlDialect {},
                        &PostgreSqlOptions::default(),
                        &mut context,
                        &guarantees,
                    )
                    .expect("the context resolves")
                });
            },
        );
    }
    group.finish();
}

fn or_chain(criterion: &mut Criterion) {
    forward_group(criterion, "forward/or_chain", STEEP, |size| {
        joined(size, " OR ", |index| format!("x = {index}"))
    });
}

fn in_list(criterion: &mut Criterion) {
    forward_group(criterion, "forward/in_list", LINEAR, |size| {
        format!("x IN ({})", joined(size, ", ", |index| index.to_string()))
    });
}

fn case_branches(criterion: &mut Criterion) {
    forward_group(criterion, "forward/case", STEEP, |size| {
        format!(
            "CASE {} ELSE NULL END",
            joined(size, " ", |index| format!("WHEN x = {index} THEN 'v{index}'"))
        )
    });
}

fn arithmetic_fold(criterion: &mut Criterion) {
    forward_group(criterion, "forward/arithmetic", LINEAR, |size| {
        format!("{} > x", joined(size, " + ", |index| format!("{index} * 3")))
    });
}

fn text_calls(criterion: &mut Criterion) {
    forward_group(criterion, "forward/text_calls", LINEAR, |size| {
        joined(size, " AND ", |index| {
            format!("(lower(s) = 'v{index}' OR length(s) < {index} OR s LIKE 'v{index}%')")
        })
    });
}

/// Measures `implied` taking `TRUE` for each expression of `group`, built
/// from each of `sizes` over a context holding that many integer columns.
fn backward_group(
    criterion: &mut Criterion,
    group: &str,
    sizes: &[usize],
    sql: impl Fn(usize) -> String,
) {
    let mut group = criterion.benchmark_group(group);
    for &size in sizes {
        let expression = parse(&sql(size));
        let mut context = context(size + 1);
        let guarantees = Guarantees::new();
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &expression,
            |bencher, expression| {
                bencher.iter(|| {
                    implied(
                        black_box(expression),
                        Outcomes::TRUE,
                        &PostgreSqlDialect {},
                        &PostgreSqlOptions::default(),
                        &mut context,
                        &guarantees,
                    )
                    .expect("the context resolves")
                });
            },
        );
    }
    group.finish();
}

fn relation_chain(criterion: &mut Criterion) {
    backward_group(criterion, "backward/relation_chain", STEEP, |size| {
        joined(size, " AND ", |index| format!("c{index} < c{}", index + 1))
    });
}

fn bounds(criterion: &mut Criterion) {
    backward_group(criterion, "backward/bounds", STEEP, |size| {
        joined(size, " AND ", |index| format!("c{index} BETWEEN {index} AND {}", index + 10))
    });
}

fn disjoint_values(criterion: &mut Criterion) {
    backward_group(criterion, "backward/or_of_equalities", LINEAR, |size| {
        joined(size, " OR ", |index| format!("(x = {index} AND c{index} IS NOT NULL)"))
    });
}

criterion_group!(
    benches,
    or_chain,
    in_list,
    case_branches,
    arithmetic_fold,
    text_calls,
    relation_chain,
    bounds,
    disjoint_values
);
criterion_main!(benches);
