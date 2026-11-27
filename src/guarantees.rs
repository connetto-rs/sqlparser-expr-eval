//! What is known about the subjects an expression reads.

use alloc::vec::Vec;
use core::fmt;

use rpds::RedBlackTreeMapSync;
use sqlparser::ast::Expr;

use super::{Evaluation, Order, Relation, Values};

/// A persistent ordered map, so a copy of the guarantees shares their storage.
type Map<K, V> = RedBlackTreeMapSync<K, V>;

/// What a guarantee is about.
///
/// Literals sort after every other subject, so a walk over the neighbours of
/// a subject can stop at the first literal when literals do not concern it.
#[derive(Debug, Clone, PartialEq, Eq)]
#[expect(
    clippy::large_enum_variant,
    reason = "box indirection would allocate on every subject clone and lookup"
)]
pub enum Subject<R> {
    /// A reference the context resolved, such as a column.
    Reference(R),
    /// Any other expression, matched as written, such as a call reading a
    /// session setting.
    Expression {
        /// The expression as written.
        expression: Expr,
        /// What the names the expression reads resolve to, in the order a
        /// walk of it meets them.
        references: Vec<R>,
    },
}

impl<R: Ord> Ord for Subject<R> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;
        literal(self).cmp(&literal(other)).then_with(|| {
            match (self, other) {
                (Self::Reference(left), Self::Reference(right)) => left.cmp(right),
                (Self::Reference(_), Self::Expression { .. }) => Ordering::Less,
                (Self::Expression { .. }, Self::Reference(_)) => Ordering::Greater,
                (
                    Self::Expression { expression: left, references: left_references },
                    Self::Expression { expression: right, references: right_references },
                ) => left.cmp(right).then_with(|| left_references.cmp(right_references)),
            }
        })
    }
}

impl<R: Ord> PartialOrd for Subject<R> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The evaluations a set of subjects is known to stay within, and the
/// relations pairs of them are known to stand in.
///
/// A subject without an entry is unconstrained, and so is a pair without a
/// relation. The guarantees are impossible when no row satisfies them.
pub struct Guarantees<R> {
    impossible: bool,
    references: Map<R, Evaluation>,
    /// Per expression, the guarantee about each resolution of its names.
    expressions: Map<Expr, Map<Vec<R>, Evaluation>>,
    /// Per order, the relation of each subject to each other one, kept in
    /// both directions so the neighbours of a subject read in one lookup.
    /// Each network is closed under composition through every subject that
    /// cannot be `NULL`, which `relate` relies on to look one step out only.
    relations: Map<Order, Network<R>>,
    /// Pairs of subjects that never both hold a value, under every order,
    /// kept in both directions.
    apart: Map<Subject<R>, Partners<R>>,
}

/// The subjects one subject never holds a value together with.
type Partners<R> = Map<Subject<R>, ()>;

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
            apart: self.apart.clone(),
        }
    }
}

impl<R: Ord> PartialEq for Guarantees<R> {
    fn eq(&self, other: &Self) -> bool {
        self.impossible == other.impossible
            && self.references == other.references
            && self.expressions == other.expressions
            && self.relations == other.relations
            && self.apart == other.apart
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
            .field("apart", &Entries(&self.apart))
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

impl fmt::Debug for Entries<'_, ()> {
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
            apart: Map::new_sync(),
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
            && self.apart.is_empty()
    }

    /// Whether some pair of subjects stands in a known relation.
    pub(crate) fn has_relations(&self) -> bool {
        !self.relations.is_empty() || !self.apart.is_empty()
    }

    /// The orders `left` may stand in to `right` under `order` on the rows
    /// where both hold a value. A subject stands equal to itself, and a
    /// subject held `NULL` stands in no order to any other.
    #[must_use]
    pub fn relation(&self, order: Order, left: &Subject<R>, right: &Subject<R>) -> Relation {
        if left == right {
            return Relation::EQUAL;
        }
        if self.never_both(left, right) {
            return Relation::NONE;
        }
        self.stored(order, left, right)
    }

    /// Whether `left` and `right`, two subjects, never both hold a value.
    fn never_both(&self, left: &Subject<R>, right: &Subject<R>) -> bool {
        self.null(left)
            || self.null(right)
            || self.apart.get(left).is_some_and(|partners| partners.contains_key(right))
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
        if self.never_both(left, right) {
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
        // Literals come last, so a literal before skips the literals after.
        let (mut after, literals): (Vec<_>, Vec<_>) =
            neighbours(right, self.non_null(right).then_some(right_row), left)
                .partition(|(after, _)| !literal(after));
        let plain = after.len();
        after.extend(literals);
        let mut pending = Vec::new();
        for (before, to_left) in neighbours(left, left_row, right) {
            let through = to_left.mirrored().composed(narrowed);
            if through == Relation::ANY {
                continue;
            }
            let after = if literal(before) { &after[..plain] } else { &after[..] };
            for &(after, from_right) in after {
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
        self.store(order, left, right, narrowed);
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
                for (beyond, from_right) in neighbours_of(network.get(right), literal(left)) {
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
                for (before, to_left) in neighbours_of(network.get(left), literal(right)) {
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
            let pending = row
                .iter()
                .filter(|(_, relation)| !relation.is_empty())
                .map(|(other, _)| (subject, other))
                .collect();
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
        if self.impossible {
            return;
        }
        let to_null = match (self.non_null(left), self.non_null(right)) {
            (true, true) => {
                *self = Self::impossible();
                return;
            }
            (true, false) => right,
            (false, true) => left,
            (false, false) if left == right => left,
            (false, false) => {
                self.keep_apart(left, right);
                return;
            }
        };
        match self.get(to_null) {
            Some(evaluation) => {
                let null = evaluation.clone().with_values(Values::none());
                self.restrict(to_null.clone(), null);
            }
            // A subject nothing is known about cannot be held `NULL` yet.
            None if left != right => self.keep_apart(left, right),
            None => {}
        }
    }

    /// Records that `left` and `right`, two subjects, never both hold a value.
    fn keep_apart(&mut self, left: &Subject<R>, right: &Subject<R>) {
        for (subject, partner) in [(left, right), (right, left)] {
            if let Some(partners) = self.apart.get_mut(subject) {
                partners.insert_mut(partner.clone(), ());
            } else {
                let partners = Map::new_sync().insert(partner.clone(), ());
                self.apart.insert_mut(subject.clone(), partners);
            }
        }
    }

    /// Every subject held `NULL`.
    fn null_subjects(&self) -> impl Iterator<Item = Subject<R>> + '_ {
        let references = self
            .references
            .iter()
            .filter(|(_, evaluation)| evaluation.values().is_empty())
            .map(|(reference, _)| Subject::Reference(reference.clone()));
        let expressions = self.expressions.iter().flat_map(|(expression, keyed)| {
            keyed.iter().filter(|(_, evaluation)| evaluation.values().is_empty()).map(
                |(references, _)| {
                    Subject::Expression {
                        expression: expression.clone(),
                        references: references.clone(),
                    }
                },
            )
        });
        references.chain(expressions)
    }

    /// The guarantee about `subject`, `None` when it is unconstrained.
    #[must_use]
    pub fn get(&self, subject: &Subject<R>) -> Option<&Evaluation> {
        match subject {
            Subject::Reference(reference) => self.references.get(reference),
            Subject::Expression { expression, references } => {
                self.expression(expression, references)
            }
        }
    }

    /// The guarantee about the reference `reference`.
    #[must_use]
    pub fn reference(&self, reference: &R) -> Option<&Evaluation> {
        self.references.get(reference)
    }

    /// The guarantee about the expression `expression`, matched as written,
    /// whose names resolve to `references`.
    #[must_use]
    pub fn expression(&self, expression: &Expr, references: &[R]) -> Option<&Evaluation> {
        self.expressions.get(expression)?.get(references)
    }

    /// Whether some guarantee is about `expression` as written, under any
    /// resolution of its names.
    pub(crate) fn names_expression(&self, expression: &Expr) -> bool {
        self.expressions.contains_key(expression)
    }

    /// Every guarantee about a reference.
    pub fn references(&self) -> impl Iterator<Item = (&R, &Evaluation)> {
        self.references.iter()
    }

    /// Every guarantee about an expression, with what its names resolve to.
    pub fn expressions(&self) -> impl Iterator<Item = (&Expr, &[R], &Evaluation)> {
        self.expressions.iter().flat_map(|(expression, keyed)| {
            keyed.iter().map(move |(references, evaluation)| {
                (expression, references.as_slice(), evaluation)
            })
        })
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
        let narrowed_null = narrowed.values().is_empty();
        if narrowed_null && !narrowed.may_be_null() {
            *self = Self::impossible();
            return;
        }
        let composes = !narrowed.may_be_null() && !self.non_null(&subject);
        let partners = self.apart.get(&subject).cloned();
        let follow = (composes || partners.is_some()).then(|| subject.clone());
        match subject {
            Subject::Reference(reference) => {
                self.references.insert_mut(reference, narrowed);
            }
            Subject::Expression { expression, references } => {
                if let Some(keyed) = self.expressions.get_mut(&expression) {
                    keyed.insert_mut(references, narrowed);
                } else {
                    let keyed = Map::new_sync().insert(references, narrowed);
                    self.expressions.insert_mut(expression, keyed);
                }
            }
        }
        let Some(subject) = follow else {
            return;
        };
        // A subject just held `NULL` has nothing left to separate, which
        // also ends the recursion through `separate`.
        if !narrowed_null {
            for partner in partners.iter().flat_map(Map::keys) {
                if self.non_null(&subject) || self.non_null(partner) {
                    self.separate(&subject, partner);
                }
            }
        }
        if composes && !self.impossible {
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
        for (expression, references, evaluation) in other.expressions() {
            let subject = Subject::Expression {
                expression: expression.clone(),
                references: references.to_vec(),
            };
            met.restrict(subject, evaluation.clone());
        }
        for (order, network) in &other.relations {
            for (left, row) in network {
                for (right, relation) in row.range(left..) {
                    met.relate(*order, left, right, *relation);
                }
            }
        }
        for (left, partners) in &other.apart {
            for (right, ()) in partners.range(left..) {
                met.separate(left, right);
            }
        }
        met
    }

    /// The guarantees holding when either holds: a subject only one side
    /// constrains is left unconstrained, and a pair stands in the orders
    /// either side allows it.
    #[must_use]
    pub fn join(&self, other: &Self) -> Self {
        if self.impossible {
            return other.clone();
        }
        if other.impossible {
            return self.clone();
        }
        let mut joined = Self {
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
                .filter_map(|(expression, ours)| {
                    let theirs = other.expressions.get(expression)?;
                    let joined: Map<_, _> = ours
                        .iter()
                        .filter_map(|(references, evaluation)| {
                            let theirs = theirs.get(references)?;
                            Some((references.clone(), evaluation.join(theirs)))
                        })
                        .collect();
                    (!joined.is_empty()).then(|| (expression.clone(), joined))
                })
                .collect(),
            ..Self::new()
        };
        for (first, second) in [(self, other), (other, self)] {
            for (left, partners) in &first.apart {
                for right in partners.keys() {
                    joined.keep_apart_in_both(left, right, second);
                }
            }
            for left in first.null_subjects() {
                let partners = second.apart.get(&left).into_iter().flat_map(Map::keys);
                for right in partners.cloned().chain(second.null_subjects()) {
                    joined.keep_apart_in_both(&left, &right, second);
                }
            }
        }
        // The join starts from `self`'s relations and rewrites only the pairs
        // `other` widens, so a side that adds nothing costs no copying. A side
        // holding no subject `NULL` and no pair apart relates each pair as it
        // stores it.
        let stored_only =
            |side: &Self| side.apart.is_empty() && side.null_subjects().next().is_none();
        let (ours_stored, theirs_stored) = (stored_only(self), stored_only(other));
        for (order, network) in &self.relations {
            let mut kept = network.clone();
            let theirs = other.relations.get(order);
            for (left, row) in network {
                let their_row = theirs.and_then(|network| network.get(left));
                for (right, stored) in row.range(left..) {
                    let ours =
                        if ours_stored { *stored } else { self.relation(*order, left, right) };
                    let theirs = if theirs_stored {
                        their_row.and_then(|row| row.get(right)).copied().unwrap_or(Relation::ANY)
                    } else {
                        other.relation(*order, left, right)
                    };
                    let union = ours.union(theirs);
                    if union == *stored {
                        continue;
                    }
                    if union == Relation::ANY || union.is_empty() {
                        forget(&mut kept, left, right);
                    } else {
                        store(&mut kept, left, right, union);
                    }
                }
            }
            if !kept.is_empty() {
                joined.relations.insert_mut(*order, kept);
            }
        }
        // A pair only `other` relates keeps its relation where `self` never
        // has both hold a value.
        for left in self.null_subjects().chain(self.apart.keys().cloned()) {
            for (order, network) in &other.relations {
                for right in network.get(&left).into_iter().flat_map(Map::keys) {
                    joined.keep_union(*order, &left, right, self, other);
                }
            }
        }
        joined
    }

    /// Every subject the guarantees hold an evaluation of.
    pub(crate) fn evaluated(&self) -> impl Iterator<Item = Subject<R>> + '_ {
        let references =
            self.references.keys().map(|reference| Subject::Reference(reference.clone()));
        let expressions = self.expressions.iter().flat_map(|(expression, keyed)| {
            keyed.keys().map(|references| {
                Subject::Expression {
                    expression: expression.clone(),
                    references: references.clone(),
                }
            })
        });
        references.chain(expressions)
    }

    /// Forgets every subject `keep` refuses, with its evaluation, relations
    /// and pairs held apart, keeping what held among the kept subjects.
    ///
    /// `keep` sees each subject the guarantees name once.
    pub fn retain(&mut self, mut keep: impl FnMut(&Subject<R>) -> bool) {
        if self.impossible {
            return;
        }
        let mut forgotten = Vec::new();
        forgotten.extend(self.evaluated().filter(|subject| !keep(subject)));
        let mut unevaluated: Vec<_> = self
            .relations
            .values()
            .flat_map(Map::keys)
            .chain(self.apart.keys())
            .filter(|subject| self.get(subject).is_none())
            .collect();
        unevaluated.sort_unstable();
        unevaluated.dedup();
        forgotten.extend(unevaluated.into_iter().filter(|subject| !keep(subject)).cloned());
        if forgotten.is_empty() {
            return;
        }
        for subject in &forgotten {
            match subject {
                Subject::Reference(reference) => {
                    self.references.remove_mut(reference);
                }
                Subject::Expression { expression, references } => {
                    if let Some(keyed) = self.expressions.get_mut(expression) {
                        keyed.remove_mut(references);
                        if keyed.is_empty() {
                            self.expressions.remove_mut(expression);
                        }
                    }
                }
            }
            isolate(&mut self.apart, subject);
        }
        self.relations = self
            .relations
            .iter()
            .filter_map(|(order, network)| {
                let mut network = network.clone();
                for subject in &forgotten {
                    isolate(&mut network, subject);
                }
                (!network.is_empty()).then_some((*order, network))
            })
            .collect();
    }

    /// Stores the orders either `first` or `second` allows `left` to stand
    /// in to `right`, when that is neither every order nor none.
    fn keep_union(
        &mut self,
        order: Order,
        left: &Subject<R>,
        right: &Subject<R>,
        first: &Self,
        second: &Self,
    ) {
        let union = first.relation(order, left, right).union(second.relation(order, left, right));
        if union != Relation::ANY && !union.is_empty() {
            self.store(order, left, right, union);
        }
    }

    /// Stores the relation of `left` to `right` under `order`.
    fn store(&mut self, order: Order, left: &Subject<R>, right: &Subject<R>, relation: Relation) {
        if let Some(network) = self.relations.get_mut(&order) {
            store(network, left, right, relation);
        } else {
            let mut network = Map::new_sync();
            store(&mut network, left, right, relation);
            self.relations.insert_mut(order, network);
        }
    }

    /// Keeps `left` and `right` apart in a join when `other`, the side that
    /// did not list the pair, never has both hold a value either.
    fn keep_apart_in_both(&mut self, left: &Subject<R>, right: &Subject<R>, other: &Self) {
        if left != right && other.never_both(left, right) && !self.null(left) && !self.null(right) {
            self.keep_apart(left, right);
        }
    }
}

/// Whether `subject` is a literal. Composition never relates two literals,
/// which stand in the order their values decide.
fn literal<R>(subject: &Subject<R>) -> bool {
    matches!(subject, Subject::Expression { expression: Expr::Value(_), .. })
}

/// The subjects `row` relates to, without the literals when `plain`, which
/// stops at the first literal since literals sort last.
fn neighbours_of<R: Ord>(
    row: Option<&Row<R>>,
    plain: bool,
) -> impl Iterator<Item = (&Subject<R>, &Relation)> {
    row.into_iter().flat_map(Map::iter).take_while(move |(subject, _)| !plain || !literal(subject))
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

/// Drops the relation of `left` to `right`, and its mirror.
fn forget<R: Ord + Clone>(network: &mut Network<R>, left: &Subject<R>, right: &Subject<R>) {
    for (from, to) in [(left, right), (right, left)] {
        let emptied = network.get_mut(from).is_some_and(|row| {
            row.remove_mut(to);
            row.is_empty()
        });
        if emptied {
            network.remove_mut(from);
        }
    }
}

/// Drops `subject` from `neighbours`, which lists each subject's neighbours
/// in both directions, with every entry naming it.
fn isolate<R: Ord + Clone, V: Clone>(
    neighbours: &mut Map<Subject<R>, Map<Subject<R>, V>>,
    subject: &Subject<R>,
) {
    let Some(row) = neighbours.get(subject).cloned() else {
        return;
    };
    neighbours.remove_mut(subject);
    for neighbour in row.keys() {
        let emptied = neighbours.get_mut(neighbour).is_some_and(|row| {
            row.remove_mut(subject);
            row.is_empty()
        });
        if emptied {
            neighbours.remove_mut(neighbour);
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

    /// The expression subject of the identifier `name` read as written,
    /// resolving no name.
    fn written(name: &str) -> Subject<String> {
        Subject::Expression {
            expression: Expr::Identifier(Ident::new(name)),
            references: Vec::new(),
        }
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
    fn an_unknown_subject_never_equal_to_itself_may_still_be_null() {
        let mut guarantees = Guarantees::<String>::new();
        guarantees.relate(Order::Number, &ref_subject("x"), &ref_subject("x"), Relation::LESS);
        assert!(!guarantees.is_impossible());
    }

    #[test]
    fn debug_output_lists_pairs_kept_apart() {
        let mut guarantees = integers(&["x", "y"], &["x", "y"]);
        guarantees.relate(Order::Number, &ref_subject("x"), &ref_subject("y"), Relation::NONE);
        let debug = format!("{guarantees:?}");
        assert!(
            debug.contains(
                r#"apart: {Reference("x"): {Reference("y"): ()}, Reference("y"): {Reference("x"): ()}}"#
            ),
            "{debug}"
        );
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

    /// A random run of guarantees over a few integers, as the closure
    /// oracle replays it.
    #[derive(Debug, Clone)]
    struct Script {
        non_null: Vec<bool>,
        null: Vec<bool>,
        constraints: Vec<(usize, usize, Relation)>,
    }

    impl Script {
        /// Draws a run over `names` and applies it to guarantees holding
        /// each name to any integer or `NULL`.
        fn draw(draws: &mut Draws, names: &[&str]) -> (Self, Guarantees<String>) {
            let count = names.len();
            let mut guarantees = integers(names, names);
            let mut script = Self {
                non_null: alloc::vec![false; count],
                null: alloc::vec![false; count],
                constraints: Vec::new(),
            };
            for _ in 0..=draws.below(24) {
                let left = draws.below(count);
                match draws.below(8) {
                    0 | 1 => {
                        script.non_null[left] = true;
                        guarantees.restrict(ref_subject(names[left]), integer(false));
                    }
                    2 => {
                        script.null[left] = true;
                        let null = integer(true).with_values(Values::none());
                        guarantees.restrict(ref_subject(names[left]), null);
                    }
                    _ => {
                        let right = draws.below(count);
                        let bits = draws.below(8);
                        let relation = Relation {
                            less: bits & 1 != 0,
                            equal: bits & 2 != 0,
                            greater: bits & 4 != 0,
                        };
                        script.constraints.push((left, right, relation));
                        let (left, right) = (ref_subject(names[left]), ref_subject(names[right]));
                        guarantees.relate(Order::Number, &left, &right, relation);
                    }
                }
            }
            (script, guarantees)
        }

        /// Both runs, one after the other.
        fn and(&self, other: &Self) -> Self {
            let either = |left: &[bool], right: &[bool]| {
                left.iter().zip(right).map(|(left, right)| *left || *right).collect()
            };
            Self {
                non_null: either(&self.non_null, &other.non_null),
                null: either(&self.null, &other.null),
                constraints: [&self.constraints[..], &other.constraints[..]].concat(),
            }
        }

        /// The relation of each subject to each other one under
        /// `Order::Number`, closed from scratch through every subject held
        /// to a value, `None` when no row satisfies the run.
        fn closure(&self) -> Option<Vec<Vec<Relation>>> {
            let count = self.non_null.len();
            if (0..count).any(|index| self.non_null[index] && self.null[index]) {
                return None;
            }
            let mut relations = alloc::vec![alloc::vec![Relation::ANY; count]; count];
            for (index, row) in relations.iter_mut().enumerate() {
                row[index] = Relation::EQUAL;
            }
            for &(left, right, relation) in &self.constraints {
                relations[left][right] = relations[left][right].intersection(relation);
                relations[right][left] = relations[right][left].intersection(relation.mirrored());
            }
            let mut changed = true;
            while changed {
                changed = false;
                for middle in (0..count).filter(|&middle| self.non_null[middle]) {
                    for left in 0..count {
                        for right in 0..count {
                            let first = relations[left][middle];
                            let second = relations[middle][right];
                            if left == right || first.is_empty() || second.is_empty() {
                                continue;
                            }
                            let composed = first.composed(second);
                            let narrowed = relations[left][right].intersection(composed);
                            if narrowed != relations[left][right] {
                                relations[left][right] = narrowed;
                                relations[right][left] = narrowed.mirrored();
                                changed = true;
                            }
                        }
                    }
                }
            }
            let mut null = self.null.clone();
            for left in 0..count {
                for right in 0..count {
                    if !relations[left][right].is_empty() {
                        continue;
                    }
                    match (self.non_null[left], self.non_null[right]) {
                        (true, true) => return None,
                        (true, false) => null[right] = true,
                        (false, true) => null[left] = true,
                        (false, false) => null[left] |= left == right,
                    }
                }
            }
            for (left, row) in relations.iter_mut().enumerate() {
                for (right, relation) in row.iter_mut().enumerate() {
                    if left != right && (null[left] || null[right]) {
                        *relation = Relation::NONE;
                    }
                }
            }
            Some(relations)
        }
    }

    /// Names for `count` subjects.
    fn names(count: usize) -> Vec<String> {
        (0..count).map(|index| alloc::format!("s{index}")).collect()
    }

    /// Asserts `guarantees` hold the relations `closure` of `script`
    /// computes, under every order since only `Order::Number` was related.
    fn assert_closure(guarantees: &Guarantees<String>, script: &Script, names: &[&str]) {
        let Some(relations) = script.closure() else {
            assert!(guarantees.is_impossible(), "{script:?}");
            return;
        };
        assert!(!guarantees.is_impossible(), "{script:?}");
        for (left, row) in relations.iter().enumerate() {
            for (right, expected) in row.iter().enumerate().filter(|(right, _)| *right != left) {
                let (left, right) = (&ref_subject(names[left]), &ref_subject(names[right]));
                let number = guarantees.relation(Order::Number, left, right);
                assert_eq!(number, *expected, "{left:?} to {right:?} in {script:?}");
                let other = if expected.is_empty() { Relation::NONE } else { Relation::ANY };
                let time = guarantees.relation(Order::Time, left, right);
                assert_eq!(time, other, "{left:?} to {right:?} in {script:?}");
            }
        }
    }

    #[test]
    fn relations_match_their_closure_whatever_the_order() {
        for seed in 0..4000_u64 {
            let mut draws = Draws(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let names = names(2 + draws.below(7));
            let names: Vec<_> = names.iter().map(String::as_str).collect();
            let (script, guarantees) = Script::draw(&mut draws, &names);
            assert_closure(&guarantees, &script, &names);
        }
    }

    #[test]
    fn meet_matches_the_closure_of_both_runs() {
        for seed in 0..4000_u64 {
            let mut draws = Draws(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let names = names(2 + draws.below(7));
            let names: Vec<_> = names.iter().map(String::as_str).collect();
            let (left_script, left) = Script::draw(&mut draws, &names);
            let (right_script, right) = Script::draw(&mut draws, &names);
            assert_closure(&left.meet(&right), &left_script.and(&right_script), &names);
        }
    }

    #[test]
    fn join_unites_the_relations_of_both_sides() {
        for seed in 0..4000_u64 {
            let mut draws = Draws(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let names = names(2 + draws.below(7));
            let names: Vec<_> = names.iter().map(String::as_str).collect();
            let (_, left) = Script::draw(&mut draws, &names);
            let (_, right) = Script::draw(&mut draws, &names);
            let joined = left.join(&right);
            for (left_name, right_name) in
                names.iter().flat_map(|left| names.iter().map(move |right| (left, right)))
            {
                let (left_subject, right_subject) =
                    (&ref_subject(left_name), &ref_subject(right_name));
                for order in [Order::Number, Order::Time] {
                    let relation = |guarantees: &Guarantees<String>| {
                        guarantees.relation(order, left_subject, right_subject)
                    };
                    let expected = match (left.is_impossible(), right.is_impossible()) {
                        (true, _) => relation(&right),
                        (false, true) => relation(&left),
                        (false, false) => relation(&left).union(relation(&right)),
                    };
                    assert_eq!(
                        relation(&joined),
                        expected,
                        "seed {seed}: {left_name} to {right_name}"
                    );
                }
            }
        }
    }

    #[test]
    fn retain_keeps_what_held_among_the_kept_subjects() {
        for seed in 0..4000_u64 {
            let mut draws = Draws(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let names = names(2 + draws.below(7));
            let names: Vec<_> = names.iter().map(String::as_str).collect();
            let (_, before) = Script::draw(&mut draws, &names);
            let kept: Vec<bool> = names.iter().map(|_| draws.below(3) != 0).collect();
            let mut retained = before.clone();
            let mut decided = Vec::new();
            retained.retain(|subject| {
                let index = names
                    .iter()
                    .position(|drawn| matches!(subject, Subject::Reference(name) if name == drawn))
                    .expect("only drawn names are subjects");
                assert!(!decided.contains(&index), "seed {seed}: {subject:?} decided twice");
                decided.push(index);
                kept[index]
            });
            if before.is_impossible() {
                assert!(retained.is_impossible(), "seed {seed}");
                continue;
            }
            // What held among the kept subjects, replayed as a run of its own.
            let mut script = Script {
                non_null: alloc::vec![false; names.len()],
                null: alloc::vec![false; names.len()],
                constraints: Vec::new(),
            };
            for (index, name) in names.iter().enumerate() {
                let evaluation = retained.reference(&(*name).to_string());
                if !kept[index] {
                    assert_eq!(evaluation, None, "seed {seed}: {name}");
                    continue;
                }
                assert_eq!(evaluation, before.reference(&(*name).to_string()), "seed {seed}");
                let evaluation = evaluation.expect("every drawn name is restricted");
                script.non_null[index] = !evaluation.may_be_null();
                script.null[index] = evaluation.values().is_empty();
                for (other, other_name) in
                    names.iter().enumerate().filter(|(other, _)| kept[*other])
                {
                    let (left, right) = (&ref_subject(name), &ref_subject(other_name));
                    for order in [Order::Number, Order::Time] {
                        let relation = retained.relation(order, left, right);
                        assert_eq!(relation, before.relation(order, left, right), "seed {seed}");
                    }
                    let relation = retained.relation(Order::Number, left, right);
                    script.constraints.push((index, other, relation));
                }
            }
            assert_closure(&retained, &script, &names);
            // Relating more stays closed through the kept subjects alone.
            let (more_script, more) = Script::draw(&mut draws, &names);
            assert_closure(&retained.meet(&more), &script.and(&more_script), &names);
            let mut every = before.clone();
            every.retain(|_| true);
            assert_eq!(every, before, "seed {seed}");
            let mut none = before.clone();
            none.retain(|_| false);
            assert_eq!(none, Guarantees::new(), "seed {seed}");
        }
    }

    #[test]
    fn retain_forgets_a_written_subject_and_keeps_what_it_implied() {
        let now = written("now");
        let (x, y, z) = (ref_subject("x"), ref_subject("y"), ref_subject("z"));
        let mut guarantees = integers(&["x", "y", "z"], &[]);
        guarantees.restrict(now.clone(), int4_known(3));
        guarantees.relate(Order::Number, &x, &y, Relation::GREATER);
        guarantees.relate(Order::Number, &z, &x, Relation::GREATER);
        guarantees.relate(Order::Number, &now, &x, Relation::LESS);
        guarantees.retain(|subject| *subject != x && *subject != now);
        assert_eq!(guarantees.reference(&"x".to_string()), None);
        assert_eq!(guarantees.get(&now), None);
        assert_eq!(guarantees.reference(&"y".to_string()), Some(&integer(false)));
        assert_eq!(guarantees.relation(Order::Number, &x, &y), Relation::ANY);
        assert_eq!(guarantees.relation(Order::Number, &now, &z), Relation::ANY);
        // `z > x > y` still held of `z` and `y` once `x` is forgotten.
        assert_eq!(guarantees.relation(Order::Number, &z, &y), Relation::GREATER);
        guarantees.restrict(x.clone(), int4_known(0));
        assert_eq!(guarantees.reference(&"x".to_string()), Some(&int4_known(0)));
        let mut impossible = Guarantees::<String>::impossible();
        impossible.retain(|_| false);
        assert!(impossible.is_impossible());
    }

    #[test]
    fn retain_forgets_subjects_only_a_relation_names_deciding_each_once() {
        let (y, u, w) = (ref_subject("y"), ref_subject("u"), ref_subject("w"));
        let mut guarantees = integers(&["y"], &[]);
        // Nothing is known of `u` and `w`, so `u` is only held apart from `y`.
        guarantees.relate(Order::Number, &y, &u, Relation::NONE);
        guarantees.relate(Order::Number, &y, &w, Relation::LESS);
        guarantees.relate(Order::Time, &y, &w, Relation::GREATER);
        let mut decided = Vec::new();
        guarantees.retain(|subject| {
            decided.push(subject.clone());
            *subject == y
        });
        decided.sort_unstable();
        assert_eq!(decided.iter().collect::<Vec<_>>(), [&u, &w, &y]);
        for order in [Order::Number, Order::Time] {
            assert_eq!(guarantees.relation(order, &y, &u), Relation::ANY);
            assert_eq!(guarantees.relation(order, &w, &y), Relation::ANY);
        }
        assert_eq!(guarantees, integers(&["y"], &[]));
    }

    #[test]
    fn join_keeps_a_relation_the_other_side_holds_null() {
        // `a < b OR b IS NULL`
        let mut left = integers(&["a", "b"], &["a", "b"]);
        left.relate(Order::Number, &ref_subject("a"), &ref_subject("b"), Relation::LESS);
        let mut right = integers(&["a", "b"], &["a", "b"]);
        right.restrict(ref_subject("b"), integer(true).with_values(Values::none()));
        let joined = left.join(&right);
        assert_eq!(
            joined.relation(Order::Number, &ref_subject("a"), &ref_subject("b")),
            Relation::LESS
        );
    }

    #[test]
    fn join_keeps_two_subjects_apart_each_side_holds_null() {
        // `x IS NULL OR y IS NULL`
        let null = integer(true).with_values(Values::none());
        let mut left = integers(&["x", "y"], &["x", "y"]);
        left.restrict(ref_subject("x"), null.clone());
        let mut right = integers(&["x", "y"], &["x", "y"]);
        right.restrict(ref_subject("y"), null);
        let mut joined = left.join(&right);
        let text = Order::Text(crate::scalar::Collation::C);
        for order in [Order::Number, Order::Time, text] {
            assert_eq!(
                joined.relation(order, &ref_subject("x"), &ref_subject("y")),
                Relation::NONE
            );
        }
        joined.restrict(ref_subject("x"), integer(false));
        assert!(joined.reference(&"y".to_string()).is_some_and(|y| y.values().is_empty()));
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
            written("f"),
            Evaluation::known(
                PgValue::Text(alloc::string::String::from("a")),
                ScalarType::Text(TextRules::DEFAULT),
            ),
        );
        assert_eq!(guarantees.reference(&"x".to_string()), Some(&int4_known(1)));
        assert_eq!(guarantees.reference(&"y".to_string()), None);
        assert_eq!(guarantees.get(&Subject::Reference("x".to_string())), Some(&int4_known(1)));
        assert_eq!(
            guarantees.expression(&Expr::Identifier(Ident::new("f")), &[]),
            Some(&Evaluation::known(
                PgValue::Text(alloc::string::String::from("a")),
                ScalarType::Text(TextRules::DEFAULT)
            ))
        );
        assert_eq!(guarantees.expression(&Expr::Identifier(Ident::new("g")), &[]), None);
        // The same expression whose names resolve otherwise is another subject
        let elsewhere = ["x".to_string()];
        assert_eq!(guarantees.expression(&Expr::Identifier(Ident::new("f")), &elsewhere), None);
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
        right.restrict(written("f"), int4_known(3));
        let met = left.meet(&right);
        assert_eq!(met.reference(&"x".to_string()), Some(&int4_known(1)));
        assert_eq!(met.reference(&"y".to_string()), Some(&int4_known(2)));
        assert_eq!(met.expression(&Expr::Identifier(Ident::new("f")), &[]), Some(&int4_known(3)));
        // An impossible side meets to impossible
        assert!(left.meet(&Guarantees::impossible()).is_impossible());
        assert!(Guarantees::<String>::impossible().meet(&left).is_impossible());
    }

    #[test]
    fn join_keeps_only_the_shared_subjects() {
        let mut left = Guarantees::<String>::new();
        left.restrict(ref_subject("x"), int4_known(1));
        left.restrict(ref_subject("left_only"), int4_known(4));
        left.restrict(written("f"), int4_known(5));
        let mut right = Guarantees::<String>::new();
        right.restrict(ref_subject("x"), int4_known(1));
        right.restrict(ref_subject("right_only"), int4_known(6));
        right.restrict(written("f"), int4_known(5));
        let joined = left.join(&right);
        assert_eq!(joined.reference(&"x".to_string()), Some(&int4_known(1)));
        assert_eq!(joined.reference(&"left_only".to_string()), None);
        assert_eq!(joined.reference(&"right_only".to_string()), None);
        assert_eq!(
            joined.expression(&Expr::Identifier(Ident::new("f")), &[]),
            Some(&int4_known(5))
        );
        // An impossible side joins to the other side
        let mut other = Guarantees::<String>::new();
        other.restrict(ref_subject("y"), int4_known(7));
        assert_eq!(Guarantees::<String>::impossible().join(&other), other);
        assert_eq!(left.join(&Guarantees::impossible()), left);
    }
}
