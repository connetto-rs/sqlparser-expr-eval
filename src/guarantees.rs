//! What is known about the subjects an expression reads.

use alloc::{collections::BTreeMap, vec::Vec};

use sqlparser::ast::Expr;

use super::{Evaluation, Order, Relation, Values};

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

/// The evaluations a set of subjects is known to stay within, and the
/// relations pairs of them are known to stand in.
///
/// A subject without an entry is unconstrained, and so is a pair without a
/// relation. The guarantees are impossible when no row satisfies them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Guarantees<R> {
    impossible: bool,
    references: BTreeMap<R, Evaluation>,
    expressions: BTreeMap<Expr, Evaluation>,
    /// Per order, the relation of each subject to each other one, kept in
    /// both directions so the neighbours of a subject read in one lookup.
    relations: BTreeMap<Order, Network<R>>,
}

/// The relation of each subject to the subjects it is related to.
type Network<R> = BTreeMap<Subject<R>, BTreeMap<Subject<R>, Relation>>;

impl<R: Ord + Clone> Default for Guarantees<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Ord + Clone> Guarantees<R> {
    /// No guarantee at all.
    #[must_use]
    pub fn new() -> Self {
        Self {
            impossible: false,
            references: BTreeMap::new(),
            expressions: BTreeMap::new(),
            relations: BTreeMap::new(),
        }
    }

    /// Guarantees no row satisfies.
    #[must_use]
    pub fn impossible() -> Self {
        Self { impossible: true, ..Self::new() }
    }

    /// Whether no row satisfies the guarantees.
    #[must_use]
    pub fn is_impossible(&self) -> bool {
        self.impossible
    }

    /// Whether the guarantees constrain nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !self.impossible
            && self.references.is_empty()
            && self.expressions.is_empty()
            && self.relations.is_empty()
    }

    /// Whether some pair of subjects stands in a known relation.
    pub(crate) fn has_relations(&self) -> bool {
        !self.relations.is_empty()
    }

    /// The orders `left` may stand in to `right` under `order` on the rows
    /// where both hold a value. A subject stands equal to itself.
    #[must_use]
    pub fn relation(&self, order: Order, left: &Subject<R>, right: &Subject<R>) -> Relation {
        if left == right {
            return Relation::EQUAL;
        }
        self.relations
            .get(&order)
            .and_then(|network| network.get(left))
            .and_then(|related| related.get(right))
            .copied()
            .unwrap_or(Relation::ANY)
    }

    /// Narrows the orders `left` may stand in to `right` under `order` to
    /// `relation` as well, with every relation the narrowing implies.
    ///
    /// The relation of `a` to `c` narrows to its composition through `b`
    /// whenever `b` cannot be `NULL`, since a row where `b` is `NULL` leaves
    /// `a` and `c` free. A pair left with no order cannot both hold a value,
    /// so the guarantees become impossible when neither side may be `NULL`,
    /// and otherwise restrict the one side that may be `NULL` to `NULL`.
    pub fn relate(
        &mut self,
        order: Order,
        left: Subject<R>,
        right: Subject<R>,
        relation: Relation,
    ) {
        if self.impossible {
            return;
        }
        let mut pending = Vec::from([(left, right, relation)]);
        while let Some((left, right, relation)) = pending.pop() {
            let known = self.relation(order, &left, &right);
            let narrowed = known.intersection(relation);
            if narrowed == known {
                continue;
            }
            if narrowed.is_empty() {
                self.separate(&left, &right);
                if self.impossible {
                    return;
                }
            }
            if left == right {
                continue;
            }
            let (left_non_null, right_non_null) = (self.non_null(&left), self.non_null(&right));
            let network = self.relations.entry(order).or_default();
            network.entry(left.clone()).or_default().insert(right.clone(), narrowed);
            network.entry(right.clone()).or_default().insert(left.clone(), narrowed.mirrored());
            if narrowed.is_empty() {
                continue;
            }
            // Every triangle the narrowed pair belongs to narrows its other
            // two sides through it.
            if right_non_null {
                for (other, beyond) in &network[&right] {
                    let through = narrowed.composed(*beyond);
                    if *other != left && through != Relation::ANY {
                        pending.push((left.clone(), other.clone(), through));
                    }
                }
            }
            if left_non_null {
                for (other, before) in &network[&left] {
                    let through = before.mirrored().composed(narrowed);
                    if *other != right && through != Relation::ANY {
                        pending.push((other.clone(), right.clone(), through));
                    }
                }
            }
        }
    }

    /// Whether the guarantees hold `subject` to a value on every row.
    fn non_null(&self, subject: &Subject<R>) -> bool {
        self.get(subject).is_some_and(|evaluation| !evaluation.may_be_null())
    }

    /// Applies that `left` and `right` never both hold a value.
    fn separate(&mut self, left: &Subject<R>, right: &Subject<R>) {
        let to_null = match (self.non_null(left), self.non_null(right)) {
            (true, true) => {
                *self = Self::impossible();
                return;
            }
            (true, false) => right,
            (false, true) => left,
            (false, false) if left == right => left,
            (false, false) => return,
        };
        if let Some(evaluation) = self.get(to_null) {
            let null = evaluation.clone().with_values(Values::none());
            self.restrict(to_null.clone(), null);
        }
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
        for (order, network) in &other.relations {
            for (left, related) in network {
                for (right, relation) in related.range(left..) {
                    met.relate(*order, left.clone(), right.clone(), *relation);
                }
            }
        }
        met
    }

    /// The guarantees holding when either holds: a subject only one side
    /// constrains is left unconstrained, and so is a pair only one side
    /// relates.
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
            relations: self
                .relations
                .iter()
                .filter_map(|(order, network)| {
                    let theirs = other.relations.get(order)?;
                    let joined: Network<R> = network
                        .iter()
                        .filter_map(|(left, related)| {
                            let their_related = theirs.get(left)?;
                            let joined: BTreeMap<_, _> = related
                                .iter()
                                .filter_map(|(right, relation)| {
                                    let union = relation.union(*their_related.get(right)?);
                                    (union != Relation::ANY).then(|| (right.clone(), union))
                                })
                                .collect();
                            (!joined.is_empty()).then(|| (left.clone(), joined))
                        })
                        .collect();
                    (!joined.is_empty()).then_some((*order, joined))
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use sql_values::postgres::{IntegerType, PgValue};
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

    /// Guarantees holding each named integer to a value, never `NULL` unless
    /// its name is listed in `nullable`.
    fn integers(names: &[&str], nullable: &[&str]) -> Guarantees<String> {
        let mut guarantees = Guarantees::new();
        for name in names {
            let null = nullable.contains(name);
            let evaluation = Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                null,
                false,
                Values::Unlisted(crate::Unlisted::any()),
            );
            guarantees.restrict(ref_subject(name), evaluation);
        }
        guarantees
    }

    #[test]
    fn relations_compose_only_through_an_operand_holding_a_value() {
        for (nullable, expected) in [(&[][..], Relation::LESS), (&["b"][..], Relation::ANY)] {
            let mut guarantees = integers(&["a", "b", "c"], nullable);
            guarantees.relate(Order::Number, ref_subject("a"), ref_subject("b"), Relation::LESS);
            guarantees.relate(Order::Number, ref_subject("b"), ref_subject("c"), Relation::LESS);
            let derived = guarantees.relation(Order::Number, &ref_subject("a"), &ref_subject("c"));
            assert_eq!(derived, expected, "b nullable: {nullable:?}");
            assert_eq!(
                guarantees.relation(Order::Number, &ref_subject("c"), &ref_subject("a")),
                expected.mirrored()
            );
            // Another order keeps nothing of these.
            assert_eq!(
                guarantees.relation(Order::Time, &ref_subject("a"), &ref_subject("c")),
                Relation::ANY
            );
        }
    }

    #[test]
    fn an_emptied_relation_leaves_the_nullable_side_null() {
        let held_null = |guarantees: &Guarantees<String>, name: &str| {
            guarantees.reference(&name.to_string()).is_some_and(|evaluation| {
                evaluation.values().is_empty() && evaluation.may_be_null()
            })
        };
        // The nullable side goes `NULL` whichever side of the pair it is on.
        for (left, right) in [("a", "n"), ("n", "a")] {
            let mut guarantees = integers(&["a", "n"], &["n"]);
            guarantees.relate(Order::Number, ref_subject(left), ref_subject(right), Relation::LESS);
            guarantees.relate(
                Order::Number,
                ref_subject(left),
                ref_subject(right),
                Relation::GREATER,
            );
            assert!(!guarantees.is_impossible());
            assert!(held_null(&guarantees, "n"), "{left} then {right}");
            assert!(!held_null(&guarantees, "a"));
        }
        // A nullable subject unequal to itself is `NULL`.
        let mut guarantees = integers(&["n"], &["n"]);
        guarantees.relate(Order::Number, ref_subject("n"), ref_subject("n"), Relation::LESS);
        assert!(held_null(&guarantees, "n"));
        // Two nullable sides, or a side nothing is guaranteed about, only
        // keep that no row holds both.
        for guarantees in [integers(&["n", "m"], &["n", "m"]), integers(&["n"], &[])] {
            let mut guarantees = guarantees;
            guarantees.relate(Order::Number, ref_subject("n"), ref_subject("m"), Relation::LESS);
            guarantees.relate(Order::Number, ref_subject("n"), ref_subject("m"), Relation::EQUAL);
            assert!(!guarantees.is_impossible());
            assert!(!held_null(&guarantees, "m"));
            assert_eq!(
                guarantees.relation(Order::Number, &ref_subject("n"), &ref_subject("m")),
                Relation::NONE
            );
        }
        // Two sides holding a value make the guarantees impossible, and an
        // impossible guarantee keeps no relation.
        let mut guarantees = integers(&["a", "b"], &[]);
        guarantees.relate(Order::Number, ref_subject("a"), ref_subject("b"), Relation::LESS);
        guarantees.relate(Order::Number, ref_subject("a"), ref_subject("b"), Relation::GREATER);
        assert!(guarantees.is_impossible());
        guarantees.relate(Order::Number, ref_subject("a"), ref_subject("b"), Relation::LESS);
        assert_eq!(
            guarantees.relation(Order::Number, &ref_subject("a"), &ref_subject("b")),
            Relation::ANY
        );
    }

    #[test]
    fn join_unites_shared_relations_and_meet_closes_them() {
        let less_or_equal = Relation::LESS.union(Relation::EQUAL);
        let mut left = integers(&["a", "b", "c"], &[]);
        left.relate(Order::Number, ref_subject("a"), ref_subject("b"), Relation::LESS);
        left.relate(Order::Number, ref_subject("b"), ref_subject("c"), Relation::LESS);
        let mut right = integers(&["a", "b", "c"], &[]);
        right.relate(Order::Number, ref_subject("a"), ref_subject("b"), Relation::EQUAL);
        let joined = left.join(&right);
        let relation = |guarantees: &Guarantees<String>, from: &str, to: &str| {
            guarantees.relation(Order::Number, &ref_subject(from), &ref_subject(to))
        };
        assert_eq!(relation(&joined, "a", "b"), less_or_equal);
        assert_eq!(relation(&joined, "b", "c"), Relation::ANY);
        let mut other = integers(&[], &[]);
        other.relate(Order::Number, ref_subject("c"), ref_subject("a"), less_or_equal);
        assert!(left.meet(&other).is_impossible());
        assert_eq!(relation(&right.meet(&other), "c", "b"), less_or_equal);
        assert!(!other.is_empty());
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
