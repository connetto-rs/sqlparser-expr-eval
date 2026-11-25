//! The minimal form search over conditions up to the connectives it admits.
#![expect(clippy::expect_used, reason = "a benchmark whose input fails to parse measures nothing")]

use core::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use sqlparser::{dialect::PostgreSqlDialect, parser::Parser};
use sqlparser_expr_eval::{Atom, MAX_CONNECTIVES, Site, Volatility, minimal};

/// Measures `minimal` at every site on each condition of `group`, built from
/// each of `sizes` over nullable atoms.
fn group(
    criterion: &mut Criterion,
    group: &str,
    sizes: impl IntoIterator<Item = usize>,
    sql: impl Fn(usize) -> String,
) {
    let mut group = criterion.benchmark_group(group);
    let read =
        |_: &_| Atom { may_be_null: true, may_raise: false, volatility: Volatility::Immutable };
    for size in sizes {
        let condition = Parser::new(&PostgreSqlDialect {})
            .try_with_sql(&sql(size))
            .and_then(|mut parser| parser.parse_expr())
            .expect("the condition parses");
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &condition,
            |bencher, condition| {
                bencher.iter(|| {
                    for site in [Site::Check, Site::Filter, Site::Value] {
                        black_box(minimal(black_box(condition), site, read));
                    }
                });
            },
        );
    }
    group.finish();
}

/// `(c0 AND c1) OR (c0 AND NOT c1) OR (c2 AND c3) OR ...` over `pairs`
/// pairs, which drops every odd atom.
fn redundant(criterion: &mut Criterion) {
    group(criterion, "minimal/redundant", 1..=(MAX_CONNECTIVES + 1) / 4, |pairs| {
        (0..pairs)
            .map(|pair| {
                let (even, odd) = (2 * pair, 2 * pair + 1);
                format!("(c{even} AND c{odd}) OR (c{even} AND NOT c{odd})")
            })
            .collect::<Vec<_>>()
            .join(" OR ")
    });
}

/// `(c0 AND NOT c1) OR (c1 AND NOT c2) OR ... OR (cN AND NOT c0)`, whose
/// prime implicants cover it cyclically.
fn ring(criterion: &mut Criterion) {
    group(criterion, "minimal/ring", 2..=MAX_CONNECTIVES.div_ceil(2), |atoms| {
        (0..atoms)
            .map(|index| format!("(c{index} AND NOT c{})", (index + 1) % atoms))
            .collect::<Vec<_>>()
            .join(" OR ")
    });
}

criterion_group!(benches, redundant, ring);
criterion_main!(benches);
