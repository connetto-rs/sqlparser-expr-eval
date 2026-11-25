//! Minimal forms of `AND`, `OR` and `NOT` trees under three-valued logic.

use quine_mc_cluskey::Bool;
use sqlparser::ast::{BinaryOperator, Expr, UnaryOperator, Value};

use crate::Volatility;

/// The most `AND` and `OR` connectives a condition [`minimal`] reads may
/// hold, which bounds both the Quine–McCluskey search and the truth table.
pub const MAX_CONNECTIVES: usize = 7;

// A condition reads at most one atom more than its connectives, and
// Quine–McCluskey names at most 32.
const _: () = assert!(MAX_CONNECTIVES < 32);

/// Where a condition sits, which decides the outcomes that act alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    /// A `CHECK` or domain constraint, which admits true and `NULL` alike.
    Check,
    /// A policy, `WHERE`, `ON`, index predicate or PL/pgSQL condition, which
    /// rejects false and `NULL` alike.
    Filter,
    /// A stored or returned value, where true, false and `NULL` all differ.
    Value,
}

/// What the caller knows of an atom, a maximal subexpression that is not
/// `AND`, `OR`, `NOT`, a parenthesis or a `TRUE`, `FALSE` or `NULL` literal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Atom {
    /// Whether it may be `NULL`.
    pub may_be_null: bool,
    /// Whether it may raise.
    pub may_raise: bool,
    /// How its value may change between two reads in one statement.
    pub volatility: Volatility,
}

impl Atom {
    /// An atom nothing is known of.
    pub const OPAQUE: Self =
        Self { may_be_null: true, may_raise: true, volatility: Volatility::Volatile };
}

/// The minimal form of a condition.
#[derive(Debug, Clone, PartialEq)]
pub struct Minimal<'a> {
    /// The form, built from the atoms the condition writes.
    pub form: Expr,
    /// The atoms the condition reads and the form does not, each at its
    /// first occurrence.
    pub dropped: Vec<&'a Expr>,
}

/// Answers the shortest form Quine–McCluskey finds for `condition` that
/// agrees with it at `site` and improves on it as clippy's `nonminimal_bool`
/// counts, or else `condition` with its negated comparisons written as their
/// negations, reading each atom through `read`.
///
/// Two occurrences are one atom when they are written alike, spans aside, and
/// `read` does not answer them volatile. A comparison and its negation are
/// one atom negated, for `<` and `>=`, `>` and `<=`, `=` and `<>`, `IS TRUE`
/// and `IS NOT TRUE`, `IS FALSE` and `IS NOT FALSE`, and `IS DISTINCT FROM`
/// and `IS NOT DISTINCT FROM`. `IS NULL` and `IS NOT NULL` stay two atoms,
/// since on a row with a `NULL` field both are false.
///
/// Quine–McCluskey proposes the candidates over two-valued logic, reading a
/// `NULL` literal as the constant that reaches the top as true in a check and
/// as false in a filter, and a
/// candidate is kept when a truth table over the atoms agrees with
/// `condition` on every row. An atom takes true and false, and `NULL` too
/// when it may be `NULL`, independently of the others. A kept candidate is
/// shorter in atoms read, negations and operators, reads no atom more often,
/// and drops no atom that may raise or is volatile. A `NULL` literal the form
/// drops is listed with the dropped atoms.
///
/// A kept candidate also passes the improvement test of clippy's
/// `nonminimal_bool`, reading some atom fewer times, or writing fewer
/// negations and as many operators, or fewer operators and as many
/// negations. An operator is an `AND` or `OR` connective, a `NOT` over an
/// `AND` or `OR` group, or a `TRUE` or `FALSE` constant, and a negation is
/// any other `NOT`, a negated comparison counting as one however it is
/// written. So `NOT a AND NOT b` and `NOT (x < y OR x < z)` have no answer,
/// since De Morgan's laws only trade negations for an operator or back.
///
/// When no candidate is kept, the answer writes each `NOT` directly over a
/// comparison that has a negation as that negation, as clippy suggests for
/// each such `NOT` it reaches, so `a AND NOT (x < y)` is answered
/// `a AND x >= y`.
///
/// A condition with more than [`MAX_CONNECTIVES`] connectives has no answer,
/// and neither has one already minimal.
///
/// ```
/// use sqlparser::{dialect::PostgreSqlDialect, parser::Parser};
/// use sqlparser_expr_eval::{Atom, Site, Volatility, minimal};
///
/// let condition = Parser::new(&PostgreSqlDialect {})
///     .try_with_sql("(a AND b) OR (a AND NOT b)")
///     .and_then(|mut parser| parser.parse_expr())
///     .expect("the condition parses");
/// let nullable = Atom { may_be_null: true, may_raise: false, volatility: Volatility::Immutable };
///
/// // A CHECK admits the NULL the condition takes where `a` is true and `b`
/// // is NULL, and a WHERE clause rejects it.
/// let minimal_form = minimal(&condition, Site::Check, |_| nullable).expect("`a` agrees");
/// assert_eq!(minimal_form.form.to_string(), "a");
/// assert_eq!(minimal_form.dropped.len(), 1);
/// assert!(minimal(&condition, Site::Filter, |_| nullable).is_none());
/// ```
pub fn minimal(
    condition: &Expr,
    site: Site,
    read: impl FnMut(&Expr) -> Atom,
) -> Option<Minimal<'_>> {
    let mut reader = Reader { leaves: Vec::new(), null: None, read, connectives: 0 };
    let read_condition = reader.condition(condition)?;
    let mut leaves = reader.leaves;
    // The `NULL` literal is the last leaf of the truth table. Quine–McCluskey
    // reads it as an atom in a value, and elsewhere as whichever constant
    // reaches the top as the outcome the site takes `NULL` for.
    let (written, proposed) = match reader.null {
        None => (read_condition, None),
        Some(expression) => {
            let null = Bool::Term(u8::try_from(leaves.len()).ok()?);
            leaves.push(Leaf { expression, atom: NULL_LITERAL, values: NULL });
            let proposed = match site {
                Site::Check => Some(substitute(&read_condition, &Bool::True, &Bool::False)),
                Site::Filter => Some(substitute(&read_condition, &Bool::False, &Bool::True)),
                Site::Value => None,
            };
            (substitute(&read_condition, &null, &null), proposed)
        }
    };
    let proposed = proposed.as_ref().unwrap_or(&written);
    let mut written_reads = vec![0; leaves.len()];
    reads(&written, &mut written_reads);
    let written_stats = Stats::of(&written);

    let mut candidates = proposed.simplify();
    for simple in Bool::Not(Box::new(proposed.clone())).simplify() {
        if matches!(simple, Bool::And(_) | Bool::Or(_)) {
            candidates.push(Bool::Not(Box::new(simple.clone())));
        }
        candidates.push(!simple);
    }
    let mut candidates: Vec<(Stats, Bool)> = candidates
        .into_iter()
        .map(|candidate| (Stats::of(&candidate), candidate))
        .filter(|(stats, _)| stats.cost() < written_stats.cost())
        .collect();
    candidates.sort_by_key(|(stats, _)| stats.cost());

    let mut candidate_reads = vec![0; leaves.len()];
    for (stats, candidate) in candidates {
        candidate_reads.fill(0);
        reads(&candidate, &mut candidate_reads);
        let mut dropped = Vec::new();
        let mut kept = true;
        for ((leaf, &written), &read) in leaves.iter().zip(&written_reads).zip(&candidate_reads) {
            if read > written {
                kept = false;
            } else if read == 0 {
                kept &= !leaf.atom.may_raise && leaf.atom.volatility != Volatility::Volatile;
                dropped.push(leaf.expression);
            }
        }
        if kept && stats.improves_on(written_stats) && agrees(&written, &candidate, &leaves, site) {
            return Some(Minimal { form: build(&candidate, &leaves), dropped });
        }
    }
    let form = with_negated_comparisons(condition)?;
    Some(Minimal { form, dropped: Vec::new() })
}

/// One of true, false and `NULL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Truth {
    True,
    False,
    Null,
}

impl Truth {
    fn not(self) -> Self {
        match self {
            Self::True => Self::False,
            Self::False => Self::True,
            Self::Null => Self::Null,
        }
    }

    fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::False, _) | (_, Self::False) => Self::False,
            (Self::True, Self::True) => Self::True,
            _ => Self::Null,
        }
    }

    fn or(self, other: Self) -> Self {
        self.not().and(other.not()).not()
    }

    /// Whether `self` and `other` act alike at `site`.
    fn agrees(self, other: Self, site: Site) -> bool {
        match site {
            Site::Check => (self == Self::False) == (other == Self::False),
            Site::Filter => (self == Self::True) == (other == Self::True),
            Site::Value => self == other,
        }
    }
}

const NON_NULL: &[Truth] = &[Truth::True, Truth::False];
const NULLABLE: &[Truth] = &[Truth::True, Truth::False, Truth::Null];
const NULL: &[Truth] = &[Truth::Null];

const NULL_LITERAL: Atom =
    Atom { may_be_null: true, may_raise: false, volatility: Volatility::Immutable };

/// The term a `NULL` literal reads as until every atom is read.
const NULL_TERM: u8 = u8::MAX;

/// An atom or a `NULL` literal, at its first occurrence.
struct Leaf<'a> {
    expression: &'a Expr,
    atom: Atom,
    /// The values the truth table assigns it.
    values: &'static [Truth],
}

/// Reads a condition into a Quine–McCluskey expression over its atoms, with
/// [`NULL_TERM`] for a `NULL` literal.
struct Reader<'a, F> {
    leaves: Vec<Leaf<'a>>,
    /// The first `NULL` literal read.
    null: Option<&'a Expr>,
    read: F,
    /// The `AND` and `OR` connectives read.
    connectives: usize,
}

impl<'a, F: FnMut(&Expr) -> Atom> Reader<'a, F> {
    fn condition(&mut self, expression: &'a Expr) -> Option<Bool> {
        match expression {
            Expr::Nested(inner) => self.condition(inner),
            Expr::BinaryOp { left, op: op @ (BinaryOperator::And | BinaryOperator::Or), right } => {
                self.connective()?;
                let mut operands = Vec::new();
                self.operands(op, left, &mut operands)?;
                self.operands(op, right, &mut operands)?;
                Some(if *op == BinaryOperator::And {
                    Bool::And(operands)
                } else {
                    Bool::Or(operands)
                })
            }
            Expr::UnaryOp { op: UnaryOperator::Not, expr } => {
                Some(Bool::Not(Box::new(self.condition(expr)?)))
            }
            Expr::Value(value) if matches!(value.value, Value::Boolean(_)) => {
                Some(if value.value == Value::Boolean(true) { Bool::True } else { Bool::False })
            }
            Expr::Value(value) if value.value == Value::Null => {
                self.null.get_or_insert(expression);
                Some(Bool::Term(NULL_TERM))
            }
            _ => self.leaf(expression),
        }
    }

    /// Pushes the operands of the `op` chain `expression` belongs to.
    fn operands(
        &mut self,
        op: &BinaryOperator,
        expression: &'a Expr,
        operands: &mut Vec<Bool>,
    ) -> Option<()> {
        match unnested(expression) {
            Expr::BinaryOp { left, op: inner, right } if inner == op => {
                self.connective()?;
                self.operands(op, left, operands)?;
                self.operands(op, right, operands)
            }
            _ => {
                operands.push(self.condition(expression)?);
                Some(())
            }
        }
    }

    fn connective(&mut self) -> Option<()> {
        self.connectives += 1;
        (self.connectives <= MAX_CONNECTIVES).then_some(())
    }

    fn leaf(&mut self, expression: &'a Expr) -> Option<Bool> {
        for (index, leaf) in self.leaves.iter().enumerate() {
            if leaf.atom.volatility == Volatility::Volatile {
                continue;
            }
            let term = Bool::Term(u8::try_from(index).ok()?);
            if leaf.expression == expression {
                return Some(term);
            }
            if negates(leaf.expression, expression) {
                return Some(Bool::Not(Box::new(term)));
            }
        }
        let term = Bool::Term(u8::try_from(self.leaves.len()).ok()?);
        let atom = (self.read)(expression);
        let values = if atom.may_be_null { NULLABLE } else { NON_NULL };
        self.leaves.push(Leaf { expression, atom, values });
        Some(term)
    }
}

/// `expression` with `positive` in place of [`NULL_TERM`] under an even
/// number of `NOT`s and `negative` under an odd number.
fn substitute(expression: &Bool, positive: &Bool, negative: &Bool) -> Bool {
    match expression {
        Bool::Term(NULL_TERM) => positive.clone(),
        Bool::True => Bool::True,
        Bool::False => Bool::False,
        Bool::Term(index) => Bool::Term(*index),
        Bool::Not(inner) => Bool::Not(Box::new(substitute(inner, negative, positive))),
        Bool::And(operands) => {
            Bool::And(
                operands.iter().map(|operand| substitute(operand, positive, negative)).collect(),
            )
        }
        Bool::Or(operands) => {
            Bool::Or(
                operands.iter().map(|operand| substitute(operand, positive, negative)).collect(),
            )
        }
    }
}

/// The expression parentheses wrap.
fn unnested(mut expression: &Expr) -> &Expr {
    while let Expr::Nested(inner) = expression {
        expression = inner;
    }
    expression
}

/// The comparison negating `op`.
fn negated_operator(op: &BinaryOperator) -> Option<BinaryOperator> {
    Some(match op {
        BinaryOperator::Lt => BinaryOperator::GtEq,
        BinaryOperator::GtEq => BinaryOperator::Lt,
        BinaryOperator::Gt => BinaryOperator::LtEq,
        BinaryOperator::LtEq => BinaryOperator::Gt,
        BinaryOperator::Eq => BinaryOperator::NotEq,
        BinaryOperator::NotEq => BinaryOperator::Eq,
        _ => return None,
    })
}

/// The comparison negating `atom`, when it has one.
fn negation(atom: &Expr) -> Option<Expr> {
    Some(match atom {
        Expr::BinaryOp { left, op, right } => {
            Expr::BinaryOp { left: left.clone(), op: negated_operator(op)?, right: right.clone() }
        }
        Expr::IsTrue(operand) => Expr::IsNotTrue(operand.clone()),
        Expr::IsNotTrue(operand) => Expr::IsTrue(operand.clone()),
        Expr::IsFalse(operand) => Expr::IsNotFalse(operand.clone()),
        Expr::IsNotFalse(operand) => Expr::IsFalse(operand.clone()),
        Expr::IsDistinctFrom(left, right) => Expr::IsNotDistinctFrom(left.clone(), right.clone()),
        Expr::IsNotDistinctFrom(left, right) => Expr::IsDistinctFrom(left.clone(), right.clone()),
        _ => return None,
    })
}

/// Whether `expression` is the comparison negating `atom`.
fn negates(atom: &Expr, expression: &Expr) -> bool {
    match (atom, expression) {
        (
            Expr::BinaryOp { left, op, right },
            Expr::BinaryOp { left: other_left, op: other_op, right: other_right },
        ) => {
            negated_operator(op).as_ref() == Some(other_op)
                && left == other_left
                && right == other_right
        }
        (Expr::IsTrue(operand), Expr::IsNotTrue(other))
        | (Expr::IsNotTrue(operand), Expr::IsTrue(other))
        | (Expr::IsFalse(operand), Expr::IsNotFalse(other))
        | (Expr::IsNotFalse(operand), Expr::IsFalse(other)) => operand == other,
        (Expr::IsDistinctFrom(left, right), Expr::IsNotDistinctFrom(other_left, other_right))
        | (Expr::IsNotDistinctFrom(left, right), Expr::IsDistinctFrom(other_left, other_right)) => {
            left == other_left && right == other_right
        }
        _ => false,
    }
}

/// Counts the reads of each leaf in `expression` into `counts`.
fn reads(expression: &Bool, counts: &mut [usize]) {
    match expression {
        Bool::True | Bool::False => {}
        Bool::Term(index) => counts[usize::from(*index)] += 1,
        Bool::Not(inner) => reads(inner, counts),
        Bool::And(operands) | Bool::Or(operands) => {
            for operand in operands {
                reads(operand, counts);
            }
        }
    }
}

/// What clippy's `nonminimal_bool` weighs a condition by, besides the reads
/// of each atom.
#[derive(Clone, Copy, Default)]
struct Stats {
    /// The atoms and `NULL` literals read.
    reads: usize,
    /// The `NOT`s over anything but an `AND` or `OR` group, each negated
    /// comparison among them.
    negations: usize,
    /// The `AND` and `OR` connectives, the `NOT`s over an `AND` or `OR` group
    /// and the `TRUE` and `FALSE` constants.
    operators: usize,
}

impl Stats {
    /// What clippy's `terminal_stats` counts in `expression`.
    fn of(expression: &Bool) -> Self {
        let mut stats = Self::default();
        stats.count(expression);
        stats
    }

    fn count(&mut self, expression: &Bool) {
        match expression {
            Bool::True | Bool::False => self.operators += 1,
            Bool::Term(_) => self.reads += 1,
            Bool::Not(inner) => {
                if matches!(**inner, Bool::And(_) | Bool::Or(_)) {
                    self.operators += 1;
                } else {
                    self.negations += 1;
                }
                self.count(inner);
            }
            Bool::And(operands) | Bool::Or(operands) => {
                self.operators += operands.len().saturating_sub(1);
                for operand in operands {
                    self.count(operand);
                }
            }
        }
    }

    /// The atoms, literals, negations and operators counted.
    fn cost(self) -> usize {
        self.reads + self.negations + self.operators
    }

    /// Whether `self` improves on `written` under the test of clippy's
    /// `nonminimal_bool`, given that it reads no atom more often.
    fn improves_on(self, written: Self) -> bool {
        self.reads < written.reads
            || (self.negations < written.negations && self.operators == written.operators)
            || (self.operators < written.operators && self.negations == written.negations)
    }
}

/// `expression` with each `NOT` its connectives reach directly over a
/// comparison that has a negation written as that negation, when there is
/// one.
fn with_negated_comparisons(expression: &Expr) -> Option<Expr> {
    match expression {
        Expr::Nested(inner) => Some(Expr::Nested(Box::new(with_negated_comparisons(inner)?))),
        Expr::BinaryOp { left, op: op @ (BinaryOperator::And | BinaryOperator::Or), right } => {
            let (written_left, written_right) =
                (with_negated_comparisons(left), with_negated_comparisons(right));
            if written_left.is_none() && written_right.is_none() {
                return None;
            }
            Some(Expr::BinaryOp {
                left: Box::new(written_left.unwrap_or_else(|| (**left).clone())),
                op: op.clone(),
                right: Box::new(written_right.unwrap_or_else(|| (**right).clone())),
            })
        }
        Expr::UnaryOp { op: UnaryOperator::Not, expr } => {
            negation(unnested(expr)).or_else(|| {
                Some(Expr::UnaryOp {
                    op: UnaryOperator::Not,
                    expr: Box::new(with_negated_comparisons(expr)?),
                })
            })
        }
        _ => None,
    }
}

/// What `expression` evaluates to when each leaf takes its value in `row`.
fn evaluate(expression: &Bool, row: &[Truth]) -> Truth {
    match expression {
        Bool::True => Truth::True,
        Bool::False => Truth::False,
        Bool::Term(index) => row[usize::from(*index)],
        Bool::Not(inner) => evaluate(inner, row).not(),
        Bool::And(operands) => {
            operands.iter().fold(Truth::True, |truth, operand| truth.and(evaluate(operand, row)))
        }
        Bool::Or(operands) => {
            operands.iter().fold(Truth::False, |truth, operand| truth.or(evaluate(operand, row)))
        }
    }
}

/// Whether `candidate` acts as `written` at `site` on every row of the truth
/// table over `leaves`.
fn agrees(written: &Bool, candidate: &Bool, leaves: &[Leaf<'_>], site: Site) -> bool {
    let mut digits = vec![0; leaves.len()];
    let mut row: Vec<Truth> = leaves.iter().map(|leaf| leaf.values[0]).collect();
    loop {
        if !evaluate(written, &row).agrees(evaluate(candidate, &row), site) {
            return false;
        }
        let mut position = 0;
        loop {
            let Some(leaf) = leaves.get(position) else {
                return true;
            };
            digits[position] += 1;
            if let Some(&value) = leaf.values.get(digits[position]) {
                row[position] = value;
                break;
            }
            digits[position] = 0;
            row[position] = leaf.values[0];
            position += 1;
        }
    }
}

/// Writes `expression` over the written leaves.
fn build(expression: &Bool, leaves: &[Leaf<'_>]) -> Expr {
    match expression {
        Bool::True => Expr::Value(Value::Boolean(true).with_empty_span()),
        Bool::False => Expr::Value(Value::Boolean(false).with_empty_span()),
        Bool::Term(index) => leaves[usize::from(*index)].expression.clone(),
        Bool::Not(inner) => {
            if let Bool::Term(index) = **inner
                && let Some(negation) = negation(leaves[usize::from(index)].expression)
            {
                return negation;
            }
            let operand = match build(inner, leaves) {
                operand @ (Expr::Identifier(_)
                | Expr::CompoundIdentifier(_)
                | Expr::Function(_)
                | Expr::Value(_)) => operand,
                operand => Expr::Nested(Box::new(operand)),
            };
            Expr::UnaryOp { op: UnaryOperator::Not, expr: Box::new(operand) }
        }
        Bool::And(operands) => connect(operands, &BinaryOperator::And, leaves),
        Bool::Or(operands) => connect(operands, &BinaryOperator::Or, leaves),
    }
}

/// Joins `operands` with `op` in the order the condition first reads them,
/// parenthesising an `OR` under an `AND`.
fn connect(operands: &[Bool], op: &BinaryOperator, leaves: &[Leaf<'_>]) -> Expr {
    let mut ordered: Vec<&Bool> = operands.iter().collect();
    ordered.sort_by_key(|operand| operand.terms().trailing_zeros());
    let mut written = ordered.into_iter().map(|operand| {
        let built = build(operand, leaves);
        if *op == BinaryOperator::And && matches!(operand, Bool::Or(_)) {
            Expr::Nested(Box::new(built))
        } else {
            built
        }
    });
    let empty = *op == BinaryOperator::And;
    let first = written.next().unwrap_or(Expr::Value(Value::Boolean(empty).with_empty_span()));
    written.fold(first, |left, right| {
        Expr::BinaryOp { left: Box::new(left), op: op.clone(), right: Box::new(right) }
    })
}

#[cfg(test)]
mod tests {
    use sqlparser::{ast::Expr, dialect::PostgreSqlDialect, parser::Parser};

    use super::{Truth, negates, negation};

    const T: Truth = Truth::True;
    const F: Truth = Truth::False;
    const N: Truth = Truth::Null;

    /// `a`, `b`, `a AND b`, `a OR b` and `NOT a` on PostgreSQL 18.7.
    const CONNECTIVES: [[Truth; 5]; 9] = [
        [F, F, F, F, T],
        [F, N, F, N, T],
        [F, T, F, T, T],
        [N, F, F, N, N],
        [N, N, N, N, N],
        [N, T, N, T, N],
        [T, F, F, T, F],
        [T, N, N, T, F],
        [T, T, T, T, F],
    ];

    const COMPARISONS: [&str; 8] = [
        "x < y",
        "x >= y",
        "x > y",
        "x <= y",
        "x = y",
        "x <> y",
        "x IS DISTINCT FROM y",
        "x IS NOT DISTINCT FROM y",
    ];

    /// [`COMPARISONS`] on PostgreSQL 18.7, over integers and over rows with
    /// `NULL` fields.
    const COMPARED: [(&str, [Truth; 8]); 12] = [
        ("1, 1", [F, T, F, T, T, F, F, T]),
        ("1, 2", [T, F, F, T, F, T, T, F]),
        ("1, NULL", [N, N, N, N, N, N, T, F]),
        ("2, 1", [F, T, T, F, F, T, T, F]),
        ("2, 2", [F, T, F, T, T, F, F, T]),
        ("2, NULL", [N, N, N, N, N, N, T, F]),
        ("NULL, 1", [N, N, N, N, N, N, T, F]),
        ("NULL, 2", [N, N, N, N, N, N, T, F]),
        ("NULL, NULL", [N, N, N, N, N, N, F, T]),
        ("ROW(1, NULL), ROW(1, NULL)", [N, N, N, N, N, N, F, T]),
        ("ROW(1, NULL), ROW(2, NULL)", [T, F, F, T, F, T, T, F]),
        ("ROW(1, NULL), ROW(1, 2)", [N, N, N, N, N, N, T, F]),
    ];

    const BOOLEAN_TESTS: [&str; 4] = ["x IS TRUE", "x IS NOT TRUE", "x IS FALSE", "x IS NOT FALSE"];

    /// [`BOOLEAN_TESTS`] on PostgreSQL 18.7.
    const TESTED: [(&str, [Truth; 4]); 3] =
        [("true", [T, F, F, T]), ("false", [F, T, T, F]), ("NULL", [F, T, F, T])];

    fn parse(sql: &str) -> Expr {
        Parser::new(&PostgreSqlDialect {}).try_with_sql(sql).unwrap().parse_expr().unwrap()
    }

    /// Asserts that the negation of each of `atoms` is recorded in `rows`
    /// with the negated value of the atom.
    fn assert_negations<const WIDTH: usize>(atoms: [&str; WIDTH], rows: &[(&str, [Truth; WIDTH])]) {
        for (index, atom) in atoms.iter().enumerate() {
            let written = parse(atom);
            let negated = negation(&written).expect("the atom has a negation");
            assert!(negates(&written, &negated), "{atom}");
            let column = atoms
                .iter()
                .position(|other| *other == negated.to_string())
                .expect("the negation is recorded");
            for (operands, values) in rows {
                assert_eq!(values[index].not(), values[column], "{atom} on {operands}");
            }
        }
    }

    #[test]
    fn connectives_match_postgresql() {
        for [left, right, and, or, not] in CONNECTIVES {
            assert_eq!(left.and(right), and, "{left:?} AND {right:?}");
            assert_eq!(left.or(right), or, "{left:?} OR {right:?}");
            assert_eq!(left.not(), not, "NOT {left:?}");
        }
    }

    #[test]
    fn negated_comparisons_match_postgresql() {
        assert_negations(COMPARISONS, &COMPARED);
    }

    #[test]
    fn negated_boolean_tests_match_postgresql() {
        assert_negations(BOOLEAN_TESTS, &TESTED);
    }
}
