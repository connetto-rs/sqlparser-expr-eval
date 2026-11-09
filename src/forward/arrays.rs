//! Arrays, forward: the `ARRAY[...]` constructor and `||` over an array.

use alloc::{collections::BTreeSet, vec, vec::Vec};

use sql_values::postgres::{ArrayDim, PgArray, PgValue};
use sqlparser::ast::{Array, CastKind, DataType, Expr};

use super::{
    super::{
        context::Context,
        dialect::EvalDialect,
        domain::{Evaluation, Unlisted, Values},
        scalar::{ArrayElement, ScalarType, TextRules},
        support,
    },
    State, forward,
    functions::{coerce_to, joined_type},
};

/// How many arrays an operation over listed operands lists before its
/// result reads as unlisted.
const ARRAY_LISTING: usize = 64;

/// The forward evaluation of `ARRAY[...]`, never `NULL` itself, of the
/// elements read as their common type, or of the sub-arrays of a
/// multidimensional array.
pub(super) fn constructor<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    array: &Array,
) -> Result<Evaluation, C::Error> {
    let mut elements = Vec::with_capacity(array.elem.len());
    for element in &array.elem {
        elements.push(forward(state, element)?);
    }
    if elements.is_empty() {
        // PostgreSQL types an empty constructor only under a cast.
        return Ok(Evaluation::raising(ScalarType::Array(ArrayElement::Unknown)));
    }
    if elements.iter().any(|element| element.ty().is_array()) {
        return Ok(multidimensional(&elements));
    }
    let mut ty = ScalarType::Unknown;
    let mut error = false;
    for element in &elements {
        let (joined, refused) = joined_type(ty, element.ty());
        ty = joined;
        error |= refused;
    }
    // Untyped literals and `NULL`s alone make a `text` array.
    if elements.iter().all(Evaluation::is_untyped) {
        ty = ScalarType::Text(TextRules::DEFAULT);
    }
    let array_type = ScalarType::Array(ArrayElement::of(ty));
    let elements: Vec<Evaluation> =
        elements.into_iter().map(|element| coerce_to(element, ty)).collect();
    Ok(finish(&elements, array_type, error))
}

/// `ARRAY[...]::type[]`, which PostgreSQL reads with each element cast to
/// the element type and each sub-array to the array type, a bracket
/// sub-array cast the same way, so that even an empty constructor has a
/// type.
pub(super) fn cast_constructor<D: EvalDialect, C: Context>(
    state: &mut State<'_, D, C>,
    kind: &CastKind,
    array: &Array,
    array_data_type: &DataType,
    element: &DataType,
) -> Result<Evaluation, C::Error> {
    let mut elements = Vec::with_capacity(array.elem.len());
    for item in &array.elem {
        elements.push(match support::unnest(item) {
            Expr::Array(inner) => cast_constructor(state, kind, inner, array_data_type, element)?,
            item => {
                let source = forward(state, item)?;
                let target = if source.ty().is_array() { array_data_type } else { element };
                super::literals::cast_evaluation(state, kind, source, target)
            }
        });
    }
    let array_type = ScalarType::Array(ArrayElement::of(state.context.data_type(element)));
    Ok(finish(&elements, array_type, false))
}

/// The finished evaluation of the listed `elements` of `array_type`,
/// multidimensional when an element is an array, impossible when one is,
/// and carrying the elements' raise.
fn finish(elements: &[Evaluation], array_type: ScalarType, error: bool) -> Evaluation {
    if elements.iter().any(|element| element.ty().is_array()) {
        return multidimensional(elements);
    }
    if elements.iter().any(Evaluation::is_impossible) {
        return Evaluation::impossible(array_type);
    }
    Evaluation::new(
        array_type,
        false,
        error || elements.iter().any(Evaluation::may_raise),
        element_choices(elements),
    )
}

/// The arrays the elements make, one per choice of a value or a `NULL` for
/// each element, unlisted for an element of unlisted values or past
/// [`ARRAY_LISTING`] arrays.
fn element_choices(elements: &[Evaluation]) -> Values {
    let mut arrays: Vec<Vec<Option<PgValue>>> = vec![Vec::new()];
    for element in elements {
        let Values::Listed(listed) = element.values() else {
            return Values::Unlisted(Unlisted::any());
        };
        let mut choices: Vec<Option<PgValue>> = listed.iter().cloned().map(Some).collect();
        if element.may_be_null() {
            choices.push(None);
        }
        if arrays.len().saturating_mul(choices.len()) > ARRAY_LISTING {
            return Values::Unlisted(Unlisted::any());
        }
        arrays = arrays
            .iter()
            .flat_map(|prefix| {
                choices.iter().map(move |choice| {
                    let mut next = prefix.clone();
                    next.push(choice.clone());
                    next
                })
            })
            .collect();
    }
    Values::Listed(arrays.into_iter().map(|array| PgValue::Array(PgArray::new(array))).collect())
}

/// `ARRAY[...]` of sub-arrays, folded when every sub-array is one constant
/// and raising when their dimensions differ.
pub(super) fn multidimensional(elements: &[Evaluation]) -> Evaluation {
    let mut ty = ScalarType::Unknown;
    let mut error = false;
    for element in elements {
        let (joined, refused) = joined_type(ty, element.ty());
        ty = joined;
        // PostgreSQL matches no sub-array with a scalar element.
        error |= refused || !(element.ty().is_array() || element.ty() == ScalarType::Unknown);
    }
    if error || !ty.is_array() {
        return Evaluation::raising(ScalarType::Array(ArrayElement::of(ty)));
    }
    let error = elements.iter().any(Evaluation::may_raise);
    let constants: Option<Vec<&PgArray>> = elements
        .iter()
        .map(|element| {
            match element.constant() {
                Some(PgValue::Array(array)) if !element.may_be_null() => Some(array),
                _ => None,
            }
        })
        .collect();
    let Some(constants) = constants else {
        // A `NULL` or a differing sub-array makes PostgreSQL raise.
        return Evaluation::new(ty, false, true, Values::Unlisted(Unlisted::any()));
    };
    let inner = constants[0].dims();
    if constants.iter().any(|array| array.dims() != inner) {
        return Evaluation::raising(ty);
    }
    let outer = u32::try_from(constants.len()).ok().and_then(|length| ArrayDim::new(1, length));
    let mut dims = Vec::with_capacity(inner.len() + 1);
    dims.extend(outer);
    dims.extend_from_slice(inner);
    let flat = constants.iter().flat_map(|array| array.elements().iter().cloned()).collect();
    match PgArray::with_dims(dims, flat) {
        Some(array) if !inner.is_empty() => {
            Evaluation::new(
                ty,
                false,
                error,
                Values::Listed(BTreeSet::from([PgValue::Array(array)])),
            )
        }
        _ => Evaluation::new(ty, false, true, Values::Unlisted(Unlisted::any())),
    }
}

/// How `||` combines an array side with the other side.
#[derive(Clone, Copy)]
enum Concat {
    /// `array_cat`, two arrays.
    Cat,
    /// `array_append`, an element after the array.
    Append,
    /// `array_prepend`, an element before the array.
    Prepend,
}

/// `left || right` with an array side, as `array_cat`, `array_append` or
/// `array_prepend` reads it, `None` when neither side is an array.
pub(super) fn concat(left: &Evaluation, right: &Evaluation) -> Option<Evaluation> {
    let (array, other, left_array) = match (left.ty(), right.ty()) {
        (ScalarType::Array(_), _) => (left, right, true),
        (_, ScalarType::Array(_)) => (right, left, false),
        _ => return None,
    };
    let element = array.ty().element().unwrap_or(ScalarType::Unknown);
    let untyped_null = other.is_untyped_null();
    // An untyped literal or `NULL` beside an array reads as an array of its
    // element type.
    let (form, other) = match other.ty() {
        ScalarType::Array(_) => (Concat::Cat, other.clone()),
        ScalarType::UnknownLiteral => {
            (Concat::Cat, super::fold::reinterpret_unknown(other.clone(), array.ty()))
        }
        _ if untyped_null => (Concat::Cat, Evaluation::null(array.ty())),
        _ if left_array => (Concat::Append, other.clone()),
        _ => (Concat::Prepend, other.clone()),
    };
    let other_element = match form {
        Concat::Cat => other.ty().element().unwrap_or(ScalarType::Unknown),
        Concat::Append | Concat::Prepend => other.ty(),
    };
    let (joined, refused) = joined_type(element, other_element);
    let ty = ScalarType::Array(ArrayElement::of(joined));
    let open_types = element.is_unmodelled() || other_element.is_unmodelled();
    if refused && !open_types {
        // PostgreSQL has no `||` between the two types.
        return Some(Evaluation::raising(ty));
    }
    let error = array.may_raise() || other.may_raise() || open_types;
    let (left, right) = if left_array { (array, &other) } else { (&other, array) };
    // `array_cat` is `NULL` only for two `NULL`s, and the other two never.
    let null = matches!(form, Concat::Cat) && left.may_be_null() && right.may_be_null();
    if open_types {
        return Some(Evaluation::new(ty, null, error, Values::Unlisted(Unlisted::any())));
    }
    Some(match listed_pairs(left, right) {
        Some(pairs) => {
            let mut values = BTreeSet::new();
            let mut raises = false;
            let mut open = false;
            for (left, right) in pairs {
                match combine(form, left, right) {
                    Fold::Value(Some(array)) => {
                        values.insert(PgValue::Array(array));
                    }
                    Fold::Value(None) => {}
                    Fold::Raises => raises = true,
                    // A multidimensional array may refuse the other side.
                    Fold::Open => {
                        open = true;
                        raises = true;
                    }
                }
            }
            let values =
                if open { Values::Unlisted(Unlisted::any()) } else { Values::Listed(values) };
            Evaluation::new(ty, null, error || raises, values)
        }
        // A multidimensional array may refuse an element.
        None => Evaluation::new(ty, null, true, Values::Unlisted(Unlisted::any())),
    })
}

/// Every pair of a listed value or `NULL` of `left` with one of `right`,
/// `None` for an unlisted side or past [`ARRAY_LISTING`] pairs.
fn listed_pairs<'a>(
    left: &'a Evaluation,
    right: &'a Evaluation,
) -> Option<Vec<(Option<&'a PgValue>, Option<&'a PgValue>)>> {
    let choices = |side: &'a Evaluation| -> Option<Vec<Option<&'a PgValue>>> {
        let Values::Listed(listed) = side.values() else { return None };
        let mut choices: Vec<Option<&PgValue>> = listed.iter().map(Some).collect();
        if side.may_be_null() {
            choices.push(None);
        }
        Some(choices)
    };
    let (lefts, rights) = (choices(left)?, choices(right)?);
    if lefts.len().saturating_mul(rights.len()) > ARRAY_LISTING {
        return None;
    }
    Some(lefts.iter().flat_map(|l| rights.iter().map(move |r| (*l, *r))).collect())
}

/// The result of one `||` over constants.
enum Fold {
    /// The array, or `NULL`.
    Value(Option<PgArray>),
    /// PostgreSQL raises.
    Raises,
    /// The evaluator does not fold it.
    Open,
}

/// A constant or `NULL` operand of `||`.
enum Side<'a> {
    /// An array, or `NULL`.
    Array(Option<&'a PgArray>),
    /// A value that is not an array.
    Scalar,
}

impl<'a> Side<'a> {
    /// The side a constant or `NULL` operand makes.
    fn of(value: Option<&'a PgValue>) -> Self {
        match value {
            Some(PgValue::Array(array)) => Side::Array(Some(array)),
            None => Side::Array(None),
            Some(_) => Side::Scalar,
        }
    }
}

/// One `||` of `form` over a left and a right constant or `NULL`.
fn combine(form: Concat, left: Option<&PgValue>, right: Option<&PgValue>) -> Fold {
    match (form, Side::of(left), Side::of(right)) {
        (Concat::Cat, Side::Array(left), Side::Array(right)) => cat(left, right),
        (Concat::Append, Side::Array(array), _) => with_element(array, right.cloned(), false),
        (Concat::Prepend, _, Side::Array(array)) => with_element(array, left.cloned(), true),
        _ => Fold::Open,
    }
}

/// `array_cat` of two one-dimensional arrays or `NULL`s, keeping the left
/// array's lower bound, and open for a multidimensional array.
fn cat(left: Option<&PgArray>, right: Option<&PgArray>) -> Fold {
    let (left, right) = match (left, right) {
        (None, None) => return Fold::Value(None),
        (Some(array), None) | (None, Some(array)) => return Fold::Value(Some(array.clone())),
        (Some(left), Some(right)) => (left, right),
    };
    if left.dims().is_empty() {
        return Fold::Value(Some(right.clone()));
    }
    if right.dims().is_empty() {
        return Fold::Value(Some(left.clone()));
    }
    let ([left_dim], [_]) = (left.dims(), right.dims()) else { return Fold::Open };
    let elements = left.elements().iter().chain(right.elements()).cloned().collect::<Vec<_>>();
    resized(left_dim.lower(), elements)
}

/// `array_append`, or `array_prepend` when `before`, of `element` to a
/// one-dimensional array or `NULL`, keeping the array's lower bound.
fn with_element(array: Option<&PgArray>, element: Option<PgValue>, before: bool) -> Fold {
    let Some(array) = array.filter(|array| !array.dims().is_empty()) else {
        return Fold::Value(Some(PgArray::new(vec![element])));
    };
    let [dim] = array.dims() else {
        // PostgreSQL appends only to an empty or one-dimensional array.
        return Fold::Raises;
    };
    // `array_prepend` computes the bound before the first, which overflows
    // below `i32::MIN`.
    if before && dim.lower() == i32::MIN {
        return Fold::Raises;
    }
    let mut elements = Vec::with_capacity(array.elements().len() + 1);
    if before {
        elements.push(element.clone());
    }
    elements.extend(array.elements().iter().cloned());
    if !before {
        elements.push(element);
    }
    resized(dim.lower(), elements)
}

/// A one-dimensional array of `elements` from `lower`, raising past the
/// bounds PostgreSQL holds.
fn resized(lower: i32, elements: Vec<Option<PgValue>>) -> Fold {
    let dim = u32::try_from(elements.len()).ok().and_then(|length| ArrayDim::new(lower, length));
    match dim.and_then(|dim| PgArray::with_dims(vec![dim], elements)) {
        Some(array) => Fold::Value(Some(array)),
        None => Fold::Raises,
    }
}

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use sql_values::postgres::{IntegerType, PgValue};

    use crate::{
        domain::{Evaluation, Unlisted, Values},
        scalar::ScalarType,
        test_context::{ColumnContext, read},
    };

    fn printed(sql: &str) -> Option<alloc::string::String> {
        read(ColumnContext::new(), sql).constant().map(ToString::to_string)
    }

    fn truth(sql: &str) -> Option<PgValue> {
        read(ColumnContext::new(), sql).constant().cloned()
    }

    /// An array is a value of its own, `NULL` only as a whole.
    #[test]
    fn an_array_is_a_value_of_its_own() {
        assert_eq!(printed("ARRAY['a']").as_deref(), Some("{a}"));
        assert_eq!(printed("ARRAY[1, NULL]").as_deref(), Some("{1,NULL}"));
        assert_eq!(printed("ARRAY[[1, 2], [3, 4]]").as_deref(), Some("{{1,2},{3,4}}"));
        assert_eq!(truth("ARRAY[NULL::int] IS NULL"), Some(PgValue::Boolean(false)));
        assert_eq!(truth("(NULL::int[]) IS NULL"), Some(PgValue::Boolean(true)));
        assert!(read(ColumnContext::new(), "ARRAY[]").always_raises());
        assert_eq!(printed("ARRAY[]::int[]").as_deref(), Some("{}"));
        assert!(read(ColumnContext::new(), "ARRAY[ARRAY[1, 2], ARRAY[3]]").always_raises());
    }

    /// A cast reads text through `array_in` and casts an array element by
    /// element.
    #[test]
    fn an_array_cast_reads_text_and_casts_elements() {
        assert_eq!(printed("'{1,2}'::int[]").as_deref(), Some("{1,2}"));
        assert!(read(ColumnContext::new(), "'{a}'::int[]").always_raises());
        assert_eq!(printed("ARRAY['abc', 'de']::varchar(2)[]").as_deref(), Some("{ab,de}"));
        assert_eq!(printed("ARRAY[1, 2]::text").as_deref(), Some("{1,2}"));
        assert!(read(ColumnContext::new(), "1::int[]").always_raises());
    }

    /// `||` concatenates, appends and prepends as PostgreSQL does.
    #[test]
    fn concatenation_follows_the_array_functions() {
        assert_eq!(printed("'' || 7 || ARRAY['1']").as_deref(), Some("{7,1}"));
        assert_eq!(printed("ARRAY[1] || 2").as_deref(), Some("{1,2}"));
        assert_eq!(printed("ARRAY[1] || '{2,3}'").as_deref(), Some("{1,2,3}"));
        assert_eq!(printed("ARRAY[1] || NULL").as_deref(), Some("{1}"));
        assert_eq!(printed("NULL::int[] || 2").as_deref(), Some("{2}"));
        assert_eq!(printed("'[0:1]={1,2}'::int[] || ARRAY[3]").as_deref(), Some("[0:2]={1,2,3}"));
        assert!(read(ColumnContext::new(), "ARRAY[1] || 'x'").always_raises());
        assert!(read(ColumnContext::new(), "ARRAY[1] || ARRAY['a']").always_raises());
        assert!(read(ColumnContext::new(), "ARRAY[[1, 2]] || 3").always_raises());
    }

    /// A cast of `ARRAY[...]` reads each element as the cast's element type.
    #[test]
    fn a_cast_constructor_types_its_elements() {
        assert_eq!(printed("ARRAY[1, 'x']::varchar(1)[]").as_deref(), Some("{1,x}"));
        assert_eq!(printed("ARRAY[[1, 2], [3, 4]]::text[]").as_deref(), Some("{{1,2},{3,4}}"));
    }

    /// Arrays join under the common type of their elements, a text array may
    /// be a `varchar` array PostgreSQL never compares with a `text` array,
    /// and a multidimensional or unmodelled side may refuse `||`.
    #[test]
    fn array_types_join_and_stay_open_where_unmodelled() {
        assert_eq!(printed("coalesce(NULL::int2[], ARRAY[5])").as_deref(), Some("{5}"));
        assert_eq!(printed("greatest(ARRAY[1]::int2[], ARRAY[2])").as_deref(), Some("{2}"));
        assert!(read(ColumnContext::new(), "ARRAY['a']::varchar(1)[] = ARRAY['a']").may_raise());
        assert!(
            read(ColumnContext::new(), "nullif(ARRAY['a'], ARRAY['a']::varchar(1)[])").may_raise()
        );
        assert!(read(ColumnContext::new(), "'{{1,2},{3,4}}'::int[] || '{9}'").may_raise());
        assert!(!read(ColumnContext::new(), "'a'::char(2) || ARRAY['b']").always_raises());
    }

    /// Arrays compare with arrays of their element type, by value.
    #[test]
    fn arrays_compare_by_value() {
        assert_eq!(truth("ARRAY[1, 2] < ARRAY[1, 3]"), Some(PgValue::Boolean(true)));
        assert_eq!(truth("ARRAY[1, 2] = ARRAY[2, 1]"), Some(PgValue::Boolean(false)));
        assert_eq!(truth("'[0:0]={1}'::int[] = '{1}'::int[]"), Some(PgValue::Boolean(false)));
        assert!(read(ColumnContext::new(), "ARRAY[1] = ARRAY['1']").always_raises());
        assert!(read(ColumnContext::new(), "ARRAY[1] = ARRAY[1]::bigint[]").always_raises());
    }

    /// PostgreSQL refuses an array where it takes a scalar.
    #[test]
    fn an_array_in_a_scalar_position_raises() {
        for sql in [
            "CASE WHEN ARRAY[1] THEN 1 END",
            "NOT ARRAY[1]",
            "1 BETWEEN ARRAY[1] AND 2",
            "ARRAY[1] = 1",
            "ARRAY[1] + 1",
            "ARRAY[1] IN (1, 2)",
        ] {
            assert!(read(ColumnContext::new(), sql).always_raises(), "{sql}");
        }
    }

    /// `ANY` reads each array a row may hold, and an untyped literal as an
    /// array of the value's type.
    #[test]
    fn any_reads_whole_arrays() {
        assert_eq!(truth("5 = ANY ('{1,5}'::int[])"), Some(PgValue::Boolean(true)));
        assert_eq!(truth("5 = ANY ('{1,2}')"), Some(PgValue::Boolean(false)));
        assert!(read(ColumnContext::new(), "5 = ANY ('x')").always_raises());
        assert!(read(ColumnContext::new(), "5 = ANY (NULL::int[])").is_always_null());
        let column = Evaluation::new(
            ScalarType::Integer(IntegerType::Integer),
            true,
            false,
            Values::Unlisted(Unlisted::any()),
        );
        let context = ColumnContext::new().with_column("i", column);
        assert!(read(context, "i = ANY ('{1,2}'::text[])").may_raise());
    }

    /// `ANY` over a multidimensional constructor reads every element.
    #[test]
    fn any_reads_every_element_of_a_multidimensional_constructor() {
        assert_eq!(truth("5 = ANY (ARRAY[[1, 2], [3, 5]])"), Some(PgValue::Boolean(true)));
        assert_eq!(truth("4 = ANY (ARRAY[[1, 2], [3, 5]])"), Some(PgValue::Boolean(false)));
    }

    fn int_array() -> ScalarType {
        ScalarType::Array(crate::scalar::ArrayElement::Integer(IntegerType::Integer))
    }

    /// An impossible element makes an impossible array, and an `ANY` over an
    /// impossible array is impossible.
    #[test]
    fn an_impossible_element_or_array_is_impossible() {
        let x = || {
            ColumnContext::new()
                .with_column("x", Evaluation::impossible(ScalarType::Integer(IntegerType::Integer)))
                .with_column("a", Evaluation::impossible(int_array()))
        };
        assert!(read(x(), "ARRAY[x]").is_impossible());
        assert!(read(x(), "ARRAY[x]::int8[]").is_impossible());
        assert!(read(x(), "1 = ANY (a)").is_impossible());
    }

    /// Elements of many listed values, or operands of many arrays, leave the
    /// arrays unlisted past the listing bound.
    #[test]
    fn many_listed_choices_read_as_unlisted() {
        let values: alloc::collections::BTreeSet<PgValue> = (0..9).map(PgValue::Integer).collect();
        let int4 = ScalarType::Integer(IntegerType::Integer);
        let context = || {
            ColumnContext::new().with_column(
                "x",
                Evaluation::new(int4, false, false, Values::Listed(values.clone())),
            )
        };
        assert!(matches!(read(context(), "ARRAY[x, x]").values(), Values::Unlisted(_)));
        let concat = read(context(), "ARRAY[x] || ARRAY[x]");
        assert!(matches!(concat.values(), Values::Unlisted(_)) && concat.may_raise());
        let unlisted = ColumnContext::new().with_column(
            "a",
            Evaluation::new(int_array(), true, false, Values::Unlisted(Unlisted::any())),
        );
        assert!(matches!(read(unlisted, "a || ARRAY[1]").values(), Values::Unlisted(_)));
    }

    /// Sub-arrays fold when constant and alike, raise beside a scalar, and
    /// stay open when unlisted or empty.
    #[test]
    fn sub_arrays_fold_raise_or_stay_open() {
        assert!(read(ColumnContext::new(), "ARRAY[ARRAY[1], 2]").always_raises());
        let context = ColumnContext::new().with_column(
            "i",
            Evaluation::new(
                ScalarType::Integer(IntegerType::Integer),
                true,
                false,
                Values::Unlisted(Unlisted::any()),
            ),
        );
        let open = read(context, "ARRAY[ARRAY[i], ARRAY[1]]");
        assert!(open.may_raise() && matches!(open.values(), Values::Unlisted(_)));
        let empty = read(ColumnContext::new(), "ARRAY[ARRAY[]::int[], ARRAY[]::int[]]");
        assert!(matches!(empty.values(), Values::Unlisted(_)));
        assert_eq!(truth("5 = ANY (ARRAY[[1, NULL], [2, 5]])"), Some(PgValue::Boolean(true)));
    }

    /// `NULL` and empty arrays concatenate as `array_cat` does, and a bound
    /// past `i32` raises.
    #[test]
    fn null_empty_and_bound_concatenations() {
        assert!(read(ColumnContext::new(), "NULL::int[] || NULL::int[]").is_always_null());
        assert_eq!(printed("'{}'::int[] || ARRAY[1]").as_deref(), Some("{1}"));
        assert_eq!(printed("ARRAY[1] || '{}'::int[]").as_deref(), Some("{1}"));
        assert!(
            read(ColumnContext::new(), "'[2147483645:2147483646]={1,2}'::int[] || 3")
                .always_raises()
        );
        assert!(
            read(ColumnContext::new(), "0 || '[-2147483648:-2147483647]={1,2}'::int[]")
                .always_raises()
        );
        let (left, right) = (PgValue::Integer(1), None);
        assert!(matches!(
            super::combine(super::Concat::Cat, Some(&left), right),
            super::Fold::Open
        ));
    }

    /// Untyped literals on both sides of `ANY` read as `text`, and a text
    /// array orders only under a byte collation.
    #[test]
    fn untyped_any_and_text_array_order() {
        assert_eq!(truth("'a' = ANY ('{a}')"), Some(PgValue::Boolean(true)));
        let ordered = read(ColumnContext::new(), "ARRAY['a'] < ARRAY['b']");
        assert!(ordered.may_be_true() && ordered.may_be_false());
        let generic =
            crate::test_context::read_default(ColumnContext::new(), "ARRAY[1] = ARRAY[2]");
        assert!(!generic.may_raise());
    }

    /// An array casts to text through its output, which a date element
    /// leaves to `DateStyle`, and to no scalar type.
    #[test]
    fn an_array_casts_to_text_alone() {
        assert_eq!(printed("ARRAY[1, 2]::text").as_deref(), Some("{1,2}"));
        let dated = read(ColumnContext::new(), "ARRAY[DATE '2020-01-01']::text");
        assert!(dated.constant().is_none() && !dated.always_raises());
        assert!(read(ColumnContext::new(), "ARRAY[1]::int").always_raises());
    }

    /// Element modifiers apply to every element, a refused modifier raises,
    /// and an unmodelled element type stays open.
    #[test]
    fn element_modifiers_apply_to_every_element() {
        assert_eq!(printed("'{1.25}'::numeric(3,1)[]").as_deref(), Some("{1.3}"));
        assert!(read(ColumnContext::new(), "'{1}'::numeric(0)[]").always_raises());
        assert_eq!(
            printed("'{\"2020-01-01 10:00:00.6\"}'::timestamp(0)[]").as_deref(),
            Some("{\"2020-01-01 10:00:01\"}")
        );
        let float = read(ColumnContext::new(), "'{1.5}'::float8[]");
        assert!(matches!(float.values(), Values::Unlisted(_)));
        let resolved = read(ColumnContext::new(), "ARRAY[1.5]::float8[] = '{1}'");
        assert!(resolved.may_be_true() && resolved.may_be_false());
    }
}
