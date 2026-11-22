//! What is known about the subjects an expression reads.

use alloc::vec::Vec;
use core::fmt;

use rpds::RedBlackTreeMapSync;
use sqlparser::ast::Expr;

use super::{Evaluation, Order, Relation, Values};

/// A persistent ordered map, so a copy of the guarantees shares their storage.
type Map<K, V> = RedBlackTreeMapSync<K, V>;

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
pub struct Guarantees<R> {
    impossible: bool,
    references: Map<R, Evaluation>,
    expressions: Map<Expr, Evaluation>,
    /// Per order, the relation of each subject to each other one, kept in
    /// both directions so the neighbours of a subject read in one lookup.
    /// Each network is closed under composition through every subject that
    /// cannot be `NULL`, which `relate` relies on to look one step out only.
    relations: Map<Order, Network<R>>,
}

/// The relation of one subject to each subject it is related to.
type Row<R> = Map<Subject<R>, Relation>;

/// The relation of each subject to the subjects it is related to.
type Network<R> = Map<Subject<R>, Row<R>>;

/// Pairs whose every triangle still has to narrow through them.
type Pending<'s, R> = Vec<(&'s Subject<R>, &'s Subject<R>)>;

impl<R: Ord> Clone for Guarantees<R> {
    fn clone(&self) -> Self {
        Self {
            impossible: self.impossible,
            references: self.references.clone(),
            expressions: self.expressions.clone(),
            relations: self.relations.clone(),
        }
    }
}

impl<R: Ord> PartialEq for Guarantees<R> {
    fn eq(&self, other: &Self) -> bool {
        self.impossible == other.impossible
            && self.references == other.references
            && self.expressions == other.expressions
            && self.relations == other.relations
    }
}

impl<R: Ord> Eq for Guarantees<R> {}

impl<R: Ord + fmt::Debug> fmt::Debug for Guarantees<R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Guarantees")
            .field("impossible", &self.impossible)
            .field("references", &Entries(&self.references))
            .field("expressions", &Entries(&self.expressions))
            .field("relations", &Entries(&self.relations))
            .finish()
    }
}

/// Formats a persistent map, and the maps nested in it, as their entries.
struct Entries<'a, T>(&'a T);

impl<K: Ord + fmt::Debug, V> fmt::Debug for Entries<'_, Map<K, V>>
where
    for<'a> Entries<'a, V>: fmt::Debug,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_map()
            .entries(self.0.iter().map(|(key, value)| (key, Entries(value))))
            .finish()
    }
}

impl fmt::Debug for Entries<'_, Evaluation> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl fmt::Debug for Entries<'_, Relation> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
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
        Self {
            impossible: false,
            references: Map::new_sync(),
            expressions: Map::new_sync(),
            relations: Map::new_sync(),
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
    /// where both hold a value. A subject stands equal to itself, and a
    /// subject held `NULL` stands in no order to any other.
    #[must_use]
    pub fn relation(&self, order: Order, left: &Subject<R>, right: &Subject<R>) -> Relation {
        if left == right {
            return Relation::EQUAL;
        }
        if self.null(left) || self.null(right) {
            return Relation::NONE;
        }
        self.stored(order, left, right)
    }

    /// The relation the network of `order` keeps from `left` to `right`.
    fn stored(&self, order: Order, left: &Subject<R>, right: &Subject<R>) -> Relation {
        self.relations
            .get(&order)
            .and_then(|network| network.get(left))
            .and_then(|row| row.get(right))
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
        left: &Subject<R>,
        right: &Subject<R>,
        relation: Relation,
    ) {
        if self.impossible {
            return;
        }
        if left == right {
            if !relation.equal {
                self.separate(left, right);
            }
            return;
        }
        let narrowed = match self.narrow(order, left, right, relation) {
            Some(narrowed) if !narrowed.is_empty() => narrowed,
            _ => return,
        };
        let Some(network) = self.relations.get(&order).cloned() else {
            return;
        };
        let (Some((left, left_row)), Some((right, right_row))) =
            (network.get_key_value(left), network.get_key_value(right))
        else {
            return;
        };
        let left_row = self.non_null(left).then_some(left_row);
        let after: Vec<_> =
            neighbours(right, self.non_null(right).then_some(right_row), left).collect();
        let mut pending = Vec::new();
        for (before, to_left) in neighbours(left, left_row, right) {
            let through = to_left.mirrored().composed(narrowed);
            if through == Relation::ANY {
                continue;
            }
            for &(after, from_right) in &after {
                if before == left && after == right {
                    continue;
                }
                let composed = through.composed(from_right);
                // A pair narrowed beyond the path through the new relation
                // may narrow others beyond their paths, so it needs every
                // triangle.
                if let Some(narrowed) = self.narrow_through(order, before, after, composed)
                    && narrowed != composed
                {
                    pending.push((before, after));
                }
                if self.impossible {
                    return;
                }
            }
        }
        self.propagate(order, &network, pending);
    }

    /// Narrows `left` to `right` to the composition `relation` of two
    /// relations through a third subject, the narrowed relation when it
    /// changed and holds an order.
    fn narrow_through(
        &mut self,
        order: Order,
        left: &Subject<R>,
        right: &Subject<R>,
        relation: Relation,
    ) -> Option<Relation> {
        // An empty side leaves its nullable end `NULL`, which `relation` reads.
        if left == right || relation == Relation::ANY || relation.is_empty() {
            return None;
        }
        self.narrow(order, left, right, relation).filter(|narrowed| !narrowed.is_empty())
    }

    /// Narrows the stored relation of `left` to `right` to `relation` as well,
    /// the narrowed relation when it changed.
    fn narrow(
        &mut self,
        order: Order,
        left: &Subject<R>,
        right: &Subject<R>,
        relation: Relation,
    ) -> Option<Relation> {
        let known = self.stored(order, left, right);
        let narrowed = known.intersection(relation);
        if narrowed == known {
            return None;
        }
        if let Some(network) = self.relations.get_mut(&order) {
            store(network, left, right, narrowed);
        } else {
            let mut network = Map::new_sync();
            store(&mut network, left, right, narrowed);
            self.relations.insert_mut(order, network);
        }
        if narrowed.is_empty() {
            self.separate(left, right);
        }
        Some(narrowed)
    }

    /// Narrows every triangle each pending pair belongs to through it, until
    /// nothing narrows, where `subjects` holds every subject of the network.
    fn propagate<'s>(
        &mut self,
        order: Order,
        subjects: &'s Network<R>,
        mut pending: Pending<'s, R>,
    ) {
        let queue = |pending: &mut Pending<'s, R>, left: &Subject<R>, right: &Subject<R>| {
            if let (Some((left, _)), Some((right, _))) =
                (subjects.get_key_value(left), subjects.get_key_value(right))
            {
                pending.push((left, right));
            }
        };
        while let Some((left, right)) = pending.pop() {
            let relation = self.stored(order, left, right);
            let Some(network) = self.relations.get(&order).cloned() else {
                return;
            };
            if self.non_null(right) {
                for (beyond, from_right) in network.get(right).into_iter().flat_map(Map::iter) {
                    let composed = relation.composed(*from_right);
                    if self.narrow_through(order, left, beyond, composed).is_some() {
                        queue(&mut pending, left, beyond);
                    }
                    if self.impossible {
                        return;
                    }
                }
            }
            if self.non_null(left) {
                for (before, to_left) in network.get(left).into_iter().flat_map(Map::iter) {
                    let composed = to_left.mirrored().composed(relation);
                    if self.narrow_through(order, before, right, composed).is_some() {
                        queue(&mut pending, before, right);
                    }
                    if self.impossible {
                        return;
                    }
                }
            }
        }
    }

    /// Narrows every pair of `subject`'s neighbours through `subject`, which
    /// no longer may be `NULL`.
    fn compose_through(&mut self, subject: &Subject<R>) {
        let networks = self.relations.clone();
        for (order, network) in &networks {
            let Some((subject, row)) = network.get_key_value(subject) else {
                continue;
            };
            let mut pending = Vec::new();
            for (other, relation) in row {
                if relation.is_empty() {
                    self.separate(subject, other);
                    if self.impossible {
                        return;
                    }
                } else {
                    pending.push((subject, other));
                }
            }
            self.propagate(*order, network, pending);
            if self.impossible {
                return;
            }
        }
    }

    /// Whether the guarantees hold `subject` to a value on every row.
    fn non_null(&self, subject: &Subject<R>) -> bool {
        self.get(subject).is_some_and(|evaluation| !evaluation.may_be_null())
    }

    /// Whether the guarantees hold `subject` `NULL` on every row.
    fn null(&self, subject: &Subject<R>) -> bool {
        self.get(subject).is_some_and(|evaluation| evaluation.values().is_empty())
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
        let composes =
            (!narrowed.may_be_null() && !self.non_null(&subject)).then(|| subject.clone());
        match subject {
            Subject::Reference(reference) => {
                self.references.insert_mut(reference, narrowed);
            }
            Subject::Expression(expression) => {
                self.expressions.insert_mut(expression, narrowed);
            }
        }
        if let Some(subject) = composes {
            self.compose_through(&subject);
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
            for (left, row) in network {
                for (right, relation) in row.range(left..) {
                    met.relate(*order, left, right, *relation);
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
                            let joined: Row<R> = related
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

/// `subject` standing equal to itself, then each subject but `other` its row
/// relates it to, with the relation `subject` stands in to it.
fn neighbours<'s, R: Ord>(
    subject: &'s Subject<R>,
    row: Option<&'s Row<R>>,
    other: &'s Subject<R>,
) -> impl Iterator<Item = (&'s Subject<R>, Relation)> {
    core::iter::once((subject, Relation::EQUAL)).chain(
        row.into_iter()
            .flat_map(Map::iter)
            .filter(move |(neighbour, _)| *neighbour != other)
            .map(|(neighbour, relation)| (neighbour, *relation)),
    )
}

/// Stores the relation of `left` to `right`, and its mirror.
fn store<R: Ord + Clone>(
    network: &mut Network<R>,
    left: &Subject<R>,
    right: &Subject<R>,
    relation: Relation,
) {
    for (from, to, relation) in [(left, right, relation), (right, left, relation.mirrored())] {
        match network.get_mut(from) {
            Some(row) => {
                match row.get_mut(to) {
                    Some(stored) => *stored = relation,
                    None => row.insert_mut(to.clone(), relation),
                }
            }
            None => network.insert_mut(from.clone(), Map::new_sync().insert(to.clone(), relation)),
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

    /// Any integer, `NULL` too when `null` holds.
    fn integer(null: bool) -> Evaluation {
        Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            null,
            false,
            Values::Unlisted(crate::Unlisted::any()),
        )
    }

    /// Guarantees holding each named integer to a value, never `NULL` unless
    /// its name is listed in `nullable`.
    fn integers(names: &[&str], nullable: &[&str]) -> Guarantees<String> {
        let mut guarantees = Guarantees::new();
        for name in names {
            guarantees.restrict(ref_subject(name), integer(nullable.contains(name)));
        }
        guarantees
    }

    #[test]
    fn relations_compose_only_through_an_operand_holding_a_value() {
        for (nullable, expected) in [(&[][..], Relation::LESS), (&["b"][..], Relation::ANY)] {
            let mut guarantees = integers(&["a", "b", "c"], nullable);
            guarantees.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::LESS);
            guarantees.relate(Order::Number, &ref_subject("b"), &ref_subject("c"), Relation::LESS);
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
    fn a_subject_turning_non_null_composes_its_neighbours() {
        let mut guarantees = integers(&["a", "b", "c"], &["b"]);
        guarantees.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::LESS);
        guarantees.relate(Order::Number, &ref_subject("b"), &ref_subject("c"), Relation::LESS);
        guarantees.restrict(ref_subject("b"), integer(false));
        assert_eq!(
            guarantees.relation(Order::Number, &ref_subject("a"), &ref_subject("c")),
            Relation::LESS
        );
    }

    /// A xorshift generator drawing indices below a bound.
    struct Draws(u64);

    impl Draws {
        fn below(&mut self, bound: usize) -> usize {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            let bound = u64::try_from(bound).expect("a bound fits u64");
            usize::try_from(self.0 % bound).expect("an index below a usize bound fits usize")
        }
    }

    /// The relations among `count` subjects that `constraints` leave once
    /// closed through those in `non_null` from scratch, with the subjects
    /// held `NULL`, `None` when no row satisfies them.
    fn closure(
        non_null: &[bool],
        constraints: &[(usize, usize, Relation)],
    ) -> Option<(Vec<Vec<Relation>>, Vec<bool>)> {
        let count = non_null.len();
        let mut relations = alloc::vec![alloc::vec![Relation::ANY; count]; count];
        for (index, row) in relations.iter_mut().enumerate() {
            row[index] = Relation::EQUAL;
        }
        for &(left, right, relation) in constraints {
            relations[left][right] = relations[left][right].intersection(relation);
            relations[right][left] = relations[right][left].intersection(relation.mirrored());
        }
        let mut changed = true;
        while changed {
            changed = false;
            for middle in (0..count).filter(|&middle| non_null[middle]) {
                for left in 0..count {
                    for right in 0..count {
                        let first = relations[left][middle];
                        let second = relations[middle][right];
                        if left == right || first.is_empty() || second.is_empty() {
                            continue;
                        }
                        let narrowed = relations[left][right].intersection(first.composed(second));
                        if narrowed != relations[left][right] {
                            relations[left][right] = narrowed;
                            relations[right][left] = narrowed.mirrored();
                            changed = true;
                        }
                    }
                }
            }
        }
        let mut null = alloc::vec![false; count];
        for left in 0..count {
            for right in 0..count {
                if !relations[left][right].is_empty() {
                    continue;
                }
                match (non_null[left], non_null[right]) {
                    (true, true) => return None,
                    (true, false) => null[right] = true,
                    (false, true) => null[left] = true,
                    (false, false) => null[left] |= left == right,
                }
            }
        }
        Some((relations, null))
    }

    #[test]
    fn relations_match_their_closure_whatever_the_order() {
        for seed in 0..4000_u64 {
            let mut draws = Draws(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let count = 2 + draws.below(7);
            let names: Vec<_> = (0..count).map(|index| alloc::format!("s{index}")).collect();
            let names: Vec<_> = names.iter().map(String::as_str).collect();
            let mut guarantees = integers(&names, &names);
            let mut non_null = alloc::vec![false; count];
            let mut constraints = Vec::new();
            for _ in 0..=draws.below(24) {
                let left = draws.below(count);
                if draws.below(4) == 0 {
                    non_null[left] = true;
                    guarantees.restrict(ref_subject(names[left]), integer(false));
                    continue;
                }
                let right = draws.below(count);
                let bits = draws.below(8);
                let relation =
                    Relation { less: bits & 1 != 0, equal: bits & 2 != 0, greater: bits & 4 != 0 };
                constraints.push((left, right, relation));
                let (left, right) = (ref_subject(names[left]), ref_subject(names[right]));
                guarantees.relate(Order::Number, &left, &right, relation);
            }
            let Some((relations, null)) = closure(&non_null, &constraints) else {
                assert!(guarantees.is_impossible(), "seed {seed}: {constraints:?} {non_null:?}");
                continue;
            };
            assert!(!guarantees.is_impossible(), "seed {seed}: {constraints:?} {non_null:?}");
            for left in 0..count {
                let held_null = guarantees
                    .reference(&names[left].to_string())
                    .is_some_and(|evaluation| evaluation.values().is_empty());
                assert_eq!(
                    held_null, null[left],
                    "seed {seed}: {left} {constraints:?} {non_null:?}"
                );
                for right in (0..count).filter(|&right| right != left) {
                    let expected = if null[left] || null[right] {
                        Relation::NONE
                    } else {
                        relations[left][right]
                    };
                    let actual = guarantees.relation(
                        Order::Number,
                        &ref_subject(names[left]),
                        &ref_subject(names[right]),
                    );
                    assert_eq!(
                        actual, expected,
                        "seed {seed}: {left} to {right} {constraints:?} {non_null:?}"
                    );
                }
            }
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
            guarantees.relate(
                Order::Number,
                &ref_subject(left),
                &ref_subject(right),
                Relation::LESS,
            );
            guarantees.relate(
                Order::Number,
                &ref_subject(left),
                &ref_subject(right),
                Relation::GREATER,
            );
            assert!(!guarantees.is_impossible());
            assert!(held_null(&guarantees, "n"), "{left} then {right}");
            assert!(!held_null(&guarantees, "a"));
        }
        // A nullable subject unequal to itself is `NULL`.
        let mut guarantees = integers(&["n"], &["n"]);
        guarantees.relate(Order::Number, &ref_subject("n"), &ref_subject("n"), Relation::LESS);
        assert!(held_null(&guarantees, "n"));
        // Two nullable sides, or a side nothing is guaranteed about, only
        // keep that no row holds both.
        for guarantees in [integers(&["n", "m"], &["n", "m"]), integers(&["n"], &[])] {
            let mut guarantees = guarantees;
            guarantees.relate(Order::Number, &ref_subject("n"), &ref_subject("m"), Relation::LESS);
            guarantees.relate(Order::Number, &ref_subject("n"), &ref_subject("m"), Relation::EQUAL);
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
        guarantees.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::LESS);
        guarantees.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::GREATER);
        assert!(guarantees.is_impossible());
        guarantees.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::LESS);
        assert_eq!(
            guarantees.relation(Order::Number, &ref_subject("a"), &ref_subject("b")),
            Relation::ANY
        );
    }

    #[test]
    fn join_unites_shared_relations_and_meet_closes_them() {
        let less_or_equal = Relation::LESS.union(Relation::EQUAL);
        let mut left = integers(&["a", "b", "c"], &[]);
        left.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::LESS);
        left.relate(Order::Number, &ref_subject("b"), &ref_subject("c"), Relation::LESS);
        let mut right = integers(&["a", "b", "c"], &[]);
        right.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::EQUAL);
        let joined = left.join(&right);
        let relation = |guarantees: &Guarantees<String>, from: &str, to: &str| {
            guarantees.relation(Order::Number, &ref_subject(from), &ref_subject(to))
        };
        assert_eq!(relation(&joined, "a", "b"), less_or_equal);
        assert_eq!(relation(&joined, "b", "c"), Relation::ANY);
        let mut other = integers(&[], &[]);
        other.relate(Order::Number, &ref_subject("c"), &ref_subject("a"), less_or_equal);
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
