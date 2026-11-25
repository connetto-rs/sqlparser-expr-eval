//! Minimal forms of conditions under three-valued logic, at each site a
//! condition sits in.
#![cfg(feature = "std")]

use sqlparser::{ast::Expr, dialect::PostgreSqlDialect, parser::Parser};
use sqlparser_expr_eval::{Atom, MAX_CONNECTIVES, Site, Volatility, minimal};

const SITES: [Site; 3] = [Site::Check, Site::Filter, Site::Value];

const NULLABLE: Atom =
    Atom { may_be_null: true, may_raise: false, volatility: Volatility::Immutable };
const NON_NULL: Atom =
    Atom { may_be_null: false, may_raise: false, volatility: Volatility::Immutable };

fn parse(sql: &str) -> Expr {
    Parser::new(&PostgreSqlDialect {}).try_with_sql(sql).unwrap().parse_expr().unwrap()
}

/// The form `minimal` answers for `sql` at `site`, with the atoms it drops,
/// reading each atom through `read`.
fn answer(sql: &str, site: Site, read: impl FnMut(&Expr) -> Atom) -> Option<(String, Vec<String>)> {
    let condition = parse(sql);
    minimal(&condition, site, read).map(|minimal| {
        (minimal.form.to_string(), minimal.dropped.iter().map(ToString::to_string).collect())
    })
}

/// The answer that writes `form` and drops `dropped`.
#[expect(clippy::unnecessary_wraps, reason = "compared with what `answer` returns")]
fn form(form: &str, dropped: &[&str]) -> Option<(String, Vec<String>)> {
    Some((form.to_string(), dropped.iter().map(ToString::to_string).collect()))
}

/// Reads the atom written `name` as `special` and every other one as `rest`.
fn reading(name: &str, special: Atom, rest: Atom) -> impl FnMut(&Expr) -> Atom {
    move |atom| if atom.to_string() == name { special } else { rest }
}

// PostgreSQL 18.7 answers `(a AND b) OR (a AND NOT b)` NULL and `a` true
// where `a` is true and `b` is NULL.
#[test]
fn dropped_atom_agrees_only_where_null_admits() {
    let sql = "(a AND b) OR (a AND NOT b)";
    assert_eq!(answer(sql, Site::Check, |_| NULLABLE), form("a", &["b"]));
    assert_eq!(answer(sql, Site::Filter, |_| NULLABLE), None);
    assert_eq!(answer(sql, Site::Value, |_| NULLABLE), None);
    for site in SITES {
        assert_eq!(answer(sql, site, |_| NON_NULL), form("a", &["b"]));
    }
}

// PostgreSQL 18.7 answers `a AND (NOT a OR b)` NULL and `a AND b` false
// where `a` is NULL and `b` is false.
#[test]
fn shorter_form_agrees_only_where_null_rejects() {
    let sql = "a AND (NOT a OR b)";
    assert_eq!(answer(sql, Site::Check, |_| NULLABLE), None);
    assert_eq!(answer(sql, Site::Filter, |_| NULLABLE), form("a AND b", &[]));
    assert_eq!(answer(sql, Site::Value, |_| NULLABLE), None);
    for site in SITES {
        assert_eq!(answer(sql, site, |_| NON_NULL), form("a AND b", &[]));
    }
}

#[test]
fn atom_that_may_raise_is_kept() {
    let raising = Atom { may_raise: true, ..NON_NULL };
    for site in SITES {
        let read = reading("b", raising, NON_NULL);
        assert_eq!(answer("(a AND b) OR (a AND NOT b)", site, read), None);
        let read = reading("a", raising, NON_NULL);
        assert_eq!(answer("(a AND b) OR (a AND NOT b)", site, read), form("a", &["b"]));
    }
}

#[test]
fn volatile_atom_written_twice_stays_two_atoms() {
    let volatile = Atom { volatility: Volatility::Volatile, ..NON_NULL };
    let sql = "random() < 0.5 OR random() < 0.5";
    assert_eq!(answer(sql, Site::Value, |_| volatile), None);
    assert_eq!(answer(sql, Site::Value, |_| NON_NULL), form("random() < 0.5", &[]));
    let stable = Atom { volatility: Volatility::Stable, ..NON_NULL };
    assert_eq!(answer(sql, Site::Value, |_| stable), form("random() < 0.5", &[]));
}

#[test]
fn volatile_atom_is_kept() {
    let volatile = Atom { volatility: Volatility::Volatile, ..NON_NULL };
    let sql = "a OR (a AND random() < 0.5)";
    let read = reading("random() < 0.5", volatile, NON_NULL);
    assert_eq!(answer(sql, Site::Value, read), None);
    assert_eq!(answer(sql, Site::Value, |_| NON_NULL), form("a", &["random() < 0.5"]));
}

#[test]
fn opaque_atom_is_kept_and_never_merged() {
    assert_eq!(answer("a OR a", Site::Check, |_| Atom::OPAQUE), None);
    assert_eq!(answer("a OR (a AND b)", Site::Check, reading("b", Atom::OPAQUE, NON_NULL)), None);
}

#[test]
fn negated_comparison_is_one_atom_negated() {
    for (sql, negated) in [
        ("x < y", "x >= y"),
        ("x > y", "x <= y"),
        ("x = y", "x <> y"),
        ("x IS TRUE", "x IS NOT TRUE"),
        ("x IS FALSE", "x IS NOT FALSE"),
        ("x IS DISTINCT FROM y", "x IS NOT DISTINCT FROM y"),
    ] {
        for (atom, negation) in [(sql, negated), (negated, sql)] {
            assert_eq!(
                answer(&format!("NOT ({atom})"), Site::Value, |_| NULLABLE),
                form(negation, &[])
            );
            let excluded = format!("{atom} OR {negation}");
            assert_eq!(answer(&excluded, Site::Value, |_| NON_NULL), form("true", &[atom]));
            assert_eq!(answer(&excluded, Site::Check, |_| NULLABLE), form("true", &[atom]));
            assert_eq!(answer(&excluded, Site::Filter, |_| NULLABLE), None);
            let contradiction = format!("{atom} AND {negation}");
            assert_eq!(answer(&contradiction, Site::Filter, |_| NULLABLE), form("false", &[atom]));
            assert_eq!(answer(&contradiction, Site::Check, |_| NULLABLE), None);
        }
    }
}

// PostgreSQL 18.7 answers both `r IS NULL` and `r IS NOT NULL` false on
// `ROW(1, NULL)`.
#[test]
fn is_null_and_is_not_null_stay_two_atoms() {
    for site in SITES {
        assert_eq!(answer("r IS NULL OR r IS NOT NULL", site, |_| NON_NULL), None);
        assert_eq!(answer("r IS NULL AND r IS NOT NULL", site, |_| NON_NULL), None);
        assert_eq!(answer("NOT (r IS NULL)", site, |_| NON_NULL), None);
    }
}

#[test]
fn constants_are_read_as_constants() {
    assert_eq!(answer("a AND TRUE", Site::Value, |_| NULLABLE), form("a", &[]));
    assert_eq!(answer("a AND FALSE", Site::Value, |_| NULLABLE), form("false", &["a"]));
    assert_eq!(answer("a OR NULL", Site::Filter, |_| NULLABLE), form("a", &["NULL"]));
    assert_eq!(answer("a OR NULL", Site::Check, |_| NULLABLE), form("true", &["a", "NULL"]));
    assert_eq!(answer("a AND NULL AND a", Site::Value, |_| NULLABLE), form("a AND NULL", &[]));
}

#[test]
fn form_reads_no_atom_more_often() {
    // Each two-level form reads `a` or `d` twice.
    let sql = "NOT (NOT (a AND (b OR c)) AND NOT d)";
    assert_eq!(answer(sql, Site::Value, |_| NON_NULL), None);
}

#[test]
fn minimal_condition_has_no_answer() {
    for sql in ["a", "a AND b", "a OR NOT b", "(a OR b) AND c", "x IS NULL"] {
        for site in SITES {
            assert_eq!(answer(sql, site, |_| NULLABLE), None, "{sql}");
        }
    }
}

#[test]
fn form_parenthesises_what_precedence_needs() {
    assert_eq!(
        answer("(a AND b) OR (a AND c)", Site::Value, |_| NULLABLE),
        form("a AND (b OR c)", &[]),
    );
    assert_eq!(
        answer("(a OR b) AND (a OR c)", Site::Value, |_| NULLABLE),
        form("a OR b AND c", &[]),
    );
    assert_eq!(
        answer("NOT (x IS NULL) AND NOT (x IS NULL)", Site::Value, |_| NULLABLE),
        form("NOT (x IS NULL)", &[]),
    );
}

#[test]
fn connective_cap_leaves_a_condition_alone() {
    let repeated = |connectives: usize| vec!["a"; connectives + 1].join(" OR ");
    assert_eq!(answer(&repeated(MAX_CONNECTIVES), Site::Value, |_| NULLABLE), form("a", &[]));
    assert_eq!(answer(&repeated(MAX_CONNECTIVES + 1), Site::Value, |_| NULLABLE), None);
}
