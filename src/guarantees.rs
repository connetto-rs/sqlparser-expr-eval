//! What is known about the subjects an expression reads.

use alloc::collections::BTreeMap;

use sqlparser::ast::Expr;

use super::Evaluation;

/// What a guarantee is about.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[expect(
    clippy::large_enum_variant,
    reason = "box indirection would allocate on every subject clone and lookup"
)]
pub enum Subject<R> {
    /// A reference the context resolved, such as a column.
    Reference(R),
    /// Any other expression, matched as written, such as a call reading a
    /// session setting.
    Expression(Expr),
}

/// The evaluations a set of subjects is known to stay within.
///
/// A subject without an entry is unconstrained. The guarantees are
/// impossible when no row satisfies them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guarantees<R> {
    impossible: bool,
    references: BTreeMap<R, Evaluation>,
    expressions: BTreeMap<Expr, Evaluation>,
}

impl<R: Ord + Clone> Default for Guarantees<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ord + Clone> Guarantees<R> {
    /// No guarantee at all.
    #[must_use]
    pub fn new() -> Self {
        Self { impossible: false, references: BTreeMap::new(), expressions: BTreeMap::new() }
    }

    /// Guarantees no row satisfies.
    #[must_use]
    pub fn impossible() -> Self {
        Self { impossible: true, references: BTreeMap::new(), expressions: BTreeMap::new() }
    }

    /// Whether no row satisfies the guarantees.
    #[must_use]
    pub fn is_impossible(&self) -> bool {
        self.impossible
    }

    /// Whether the guarantees constrain nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.impossible && self.references.is_empty() && self.expressions.is_empty()
    }

    /// The guarantee about `subject`, `None` when it is unconstrained.
    #[must_use]
    pub fn get(&self, subject: &Subject<R>) -> Option<&Evaluation> {
        match subject {
            Subject::Reference(reference) => self.references.get(reference),
            Subject::Expression(expression) => self.expressions.get(expression),
        }
    }

    /// The guarantee about the reference `reference`.
    #[must_use]
    pub fn reference(&self, reference: &R) -> Option<&Evaluation> {
        self.references.get(reference)
    }

    /// The guarantee about the expression `expression`, matched as written.
    #[must_use]
    pub fn expression(&self, expression: &Expr) -> Option<&Evaluation> {
        self.expressions.get(expression)
    }

    /// Every guarantee about a reference.
    pub fn references(&self) -> impl Iterator<Item = (&R, &Evaluation)> {
        self.references.iter()
    }

    /// Every guarantee about an expression.
    pub fn expressions(&self) -> impl Iterator<Item = (&Expr, &Evaluation)> {
        self.expressions.iter()
    }

    /// Narrows `subject` to the outcomes `evaluation` allows as well.
    ///
    /// The guarantees become impossible when the subject is left with no
    /// value and cannot be `NULL`. Whether it may raise is irrelevant to a
    /// guarantee, which states what it holds when it holds a value.
    pub fn restrict(&mut self, subject: Subject<R>, evaluation: Evaluation) {
        if self.impossible {
            return;
        }
        let narrowed = match self.get(&subject) {
            Some(known) => known.meet(&evaluation),
            None => evaluation,
        };
        if narrowed.values().is_empty() && !narrowed.may_be_null() {
            *self = Self::impossible();
            return;
        }
        match subject {
            Subject::Reference(reference) => {
                self.references.insert(reference, narrowed);
            }
            Subject::Expression(expression) => {
                self.expressions.insert(expression, narrowed);
            }
        }
    }

    /// The guarantees holding when both hold.
    #[must_use]
    pub fn meet(&self, other: &Self) -> Self {
        if self.impossible || other.impossible {
            return Self::impossible();
        }
        let mut met = self.clone();
        for (reference, evaluation) in &other.references {
            met.restrict(Subject::Reference(reference.clone()), evaluation.clone());
        }
        for (expression, evaluation) in &other.expressions {
            met.restrict(Subject::Expression(expression.clone()), evaluation.clone());
        }
        met
    }

    /// The guarantees holding when either holds: a subject only one side
    /// constrains is left unconstrained.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        if self.impossible {
            return other.clone();
        }
        if other.impossible {
            return self.clone();
        }
        Self {
            impossible: false,
            references: self
                .references
                .iter()
                .filter_map(|(reference, evaluation)| {
                    other
                        .references
                        .get(reference)
                        .map(|theirs| (reference.clone(), evaluation.join(theirs)))
                })
                .collect(),
            expressions: self
                .expressions
                .iter()
                .filter_map(|(expression, evaluation)| {
                    other
                        .expressions
                        .get(expression)
                        .map(|theirs| (expression.clone(), evaluation.join(theirs)))
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use sql_values::{IntegerType, PgValue};
    use sqlparser::ast::{Expr, Ident};

    use super::*;
    use crate::{ScalarType, TextRules, Values};

    fn ref_subject(name: &str) -> Subject<String> {
        Subject::Reference(name.to_string())
    }

    fn int4_known(value: i64) -> Evaluation {
        Evaluation::known(PgValue::Integer(value), ScalarType::Integer(IntegerType::Integer))
    }

    #[test]
    fn new_is_default_and_empty() {
        assert_eq!(Guarantees::<String>::default(), Guarantees::new());
        assert!(!Guarantees::<String>::new().is_impossible());
        assert!(Guarantees::<String>::new().is_empty());
        assert!(Guarantees::<String>::impossible().is_impossible());
        assert!(!Guarantees::<String>::impossible().is_empty());
    }

    #[test]
    fn restrict_stores_by_reference_and_expression() {
        let mut guarantees = Guarantees::<String>::new();
        guarantees.restrict(ref_subject("x"), int4_known(1));
        guarantees.restrict(
            Subject::Expression(Expr::Identifier(Ident::new("f"))),
            Evaluation::known(
                PgValue::Text(alloc::string::String::from("a")),
                ScalarType::Text(TextRules::DEFAULT),
            ),
        );
        assert_eq!(guarantees.reference(&"x".to_string()), Some(&int4_known(1)));
        assert_eq!(guarantees.reference(&"y".to_string()), None);
        assert_eq!(guarantees.get(&Subject::Reference("x".to_string())), Some(&int4_known(1)));
        assert_eq!(
            guarantees.expression(&Expr::Identifier(Ident::new("f"))),
            Some(&Evaluation::known(
                PgValue::Text(alloc::string::String::from("a")),
                ScalarType::Text(TextRules::DEFAULT)
            ))
        );
        assert_eq!(guarantees.expression(&Expr::Identifier(Ident::new("g"))), None);
        // The iterators list every stored subject
        assert_eq!(
            guarantees.references().map(|(reference, _)| reference.as_str()).collect::<Vec<_>>(),
            ["x"]
        );
        assert_eq!(guarantees.expressions().count(), 1);
        assert!(!guarantees.is_empty());
    }

    #[test]
    fn restrict_leaves_no_value_and_no_null_becomes_impossible() {
        let mut guarantees = Guarantees::<String>::new();
        guarantees.restrict(ref_subject("x"), int4_known(1));
        // Meeting a disjoint value leaves no value the non-`NULL` subject can
        // take
        guarantees.restrict(ref_subject("x"), int4_known(2));
        assert!(guarantees.is_impossible());
        // A `NULL`-able subject survives the same meet, left with no value
        let mut guarantees = Guarantees::<String>::new();
        let nullable = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Listed(alloc::collections::BTreeSet::from([PgValue::Integer(1)])),
        );
        let other = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Listed(alloc::collections::BTreeSet::from([PgValue::Integer(2)])),
        );
        guarantees.restrict(ref_subject("x"), nullable);
        guarantees.restrict(ref_subject("x"), other);
        assert!(!guarantees.is_impossible());
        let x = guarantees.reference(&"x".to_string()).expect("the meet names x");
        assert!(x.may_be_null());
        assert!(x.values().is_empty());
        // Restricting impossible guarantees is a no-op
        let mut impossible = Guarantees::<String>::impossible();
        impossible.restrict(ref_subject("x"), int4_known(1));
        assert!(impossible.is_impossible());
        assert!(impossible.reference(&"x".to_string()).is_none());
    }

    #[test]
    fn meet_adopts_the_other_side() {
        let mut left = Guarantees::<String>::new();
        left.restrict(ref_subject("x"), int4_known(1));
        let mut right = Guarantees::<String>::new();
        right.restrict(ref_subject("y"), int4_known(2));
        right.restrict(Subject::Expression(Expr::Identifier(Ident::new("f"))), int4_known(3));
        let met = left.meet(&right);
        assert_eq!(met.reference(&"x".to_string()), Some(&int4_known(1)));
        assert_eq!(met.reference(&"y".to_string()), Some(&int4_known(2)));
        assert_eq!(met.expression(&Expr::Identifier(Ident::new("f"))), Some(&int4_known(3)));
        // An impossible side meets to impossible
        assert!(left.meet(&Guarantees::impossible()).is_impossible());
        assert!(Guarantees::<String>::impossible().meet(&left).is_impossible());
    }

    #[test]
    fn join_keeps_only_the_shared_subjects() {
        let mut left = Guarantees::<String>::new();
        left.restrict(ref_subject("x"), int4_known(1));
        left.restrict(ref_subject("left_only"), int4_known(4));
        left.restrict(Subject::Expression(Expr::Identifier(Ident::new("f"))), int4_known(5));
        let mut right = Guarantees::<String>::new();
        right.restrict(ref_subject("x"), int4_known(1));
        right.restrict(ref_subject("right_only"), int4_known(6));
        right.restrict(Subject::Expression(Expr::Identifier(Ident::new("f"))), int4_known(5));
        let joined = left.join(&right);
        assert_eq!(joined.reference(&"x".to_string()), Some(&int4_known(1)));
        assert_eq!(joined.reference(&"left_only".to_string()), None);
        assert_eq!(joined.reference(&"right_only".to_string()), None);
        assert_eq!(joined.expression(&Expr::Identifier(Ident::new("f"))), Some(&int4_known(5)));
        // An impossible side joins to the other side
        let mut other = Guarantees::<String>::new();
        other.restrict(ref_subject("y"), int4_known(7));
        assert_eq!(Guarantees::<String>::impossible().join(&other), other);
        assert_eq!(left.join(&Guarantees::impossible()), left);
    }
}
