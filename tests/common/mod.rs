//! The context and helpers the integration tests and the benchmarks read
//! the public API through.

use sqlparser::{
    ast::{
        ArrayElemTypeDef, DataType, Expr, Function, ObjectName, ObjectNamePart, Query, SelectItem,
        SetExpr, Statement,
    },
    dialect::PostgreSqlDialect,
    parser::{Parser, ParserError},
};
use sqlparser_expr_eval::{
    Builtin, Callee, CalleeKind, Context, Evaluation, Guarantees, NullInput, NullResult,
    QueryEvaluation, Resolved, ScalarType, TextRules, Unlisted, Values, Volatility,
};

/// What a checking type's checks make of a value converted to it.
pub type Check = fn(Evaluation) -> Evaluation;

/// A context that resolves a fixed set of columns, callees, collations,
/// named types and domain checks and one nested query.
pub struct Ctx {
    pub columns: Vec<(String, Evaluation)>,
    pub callees: Vec<(String, Callee)>,
    pub collations: Vec<(String, TextRules)>,
    pub types: Vec<(String, ScalarType)>,
    /// The named types that check a converted value, each with what its
    /// checks make of one, read the same over an array of the type.
    pub checks: Vec<(String, Check)>,
    pub query: Option<QueryEvaluation>,
    /// The references the engine may plan as constants.
    pub plan_constants: Vec<String>,
}

impl Ctx {
    /// A context resolving `columns` and nothing else.
    pub fn with_columns(columns: Vec<(String, Evaluation)>) -> Self {
        Self {
            columns,
            callees: Vec::new(),
            collations: Vec::new(),
            types: Vec::new(),
            checks: Vec::new(),
            query: None,
            plan_constants: Vec::new(),
        }
    }

    /// The context with `callee` resolving calls to `name`.
    pub fn with_callee(mut self, name: &str, callee: Callee) -> Self {
        self.callees.push((name.to_string(), callee));
        self
    }

    /// The checks of `data_type`, or of its element type when it is an
    /// array, when it names a checking type.
    fn check(&self, data_type: &DataType) -> Option<Check> {
        let data_type = match data_type {
            DataType::Array(ArrayElemTypeDef::SquareBracket(element, _)) => element,
            data_type => data_type,
        };
        let DataType::Custom(name, _) = data_type else {
            return None;
        };
        let name = name.0.last().and_then(ObjectNamePart::as_ident)?;
        self.checks.iter().find(|(known, _)| name.value == *known).map(|(_, check)| *check)
    }
}

impl Context for Ctx {
    type Reference = String;
    type Error = ParserError;

    fn reference(
        &mut self,
        expression: &Expr,
    ) -> Result<Option<Resolved<Self::Reference>>, Self::Error> {
        let name = match expression {
            Expr::Identifier(ident) => Some(ident.value.clone()),
            Expr::CompoundIdentifier(parts) if parts.len() == 1 => Some(parts[0].value.clone()),
            _ => None,
        };
        let Some(name) = name else {
            return Ok(None);
        };
        Ok(self.columns.iter().find(|(key, _)| *key == name).map(|(_, evaluation)| {
            Resolved { reference: name.clone(), evaluation: evaluation.clone() }
        }))
    }

    fn callees(&mut self, call: &Function) -> Result<Vec<Callee>, Self::Error> {
        let Some(name) = call.name.0.last().and_then(ObjectNamePart::as_ident) else {
            return Ok(Vec::new());
        };
        let name = name.value.to_lowercase();
        Ok(self
            .callees
            .iter()
            .filter(|(callee_name, _)| *callee_name == name)
            .map(|(_, callee)| callee.clone())
            .collect())
    }

    fn query(
        &mut self,
        _query: &Query,
        _guarantees: &Guarantees<Self::Reference>,
    ) -> Result<QueryEvaluation, Self::Error> {
        Ok(self.query.clone().unwrap_or_else(QueryEvaluation::unknown))
    }

    fn collation(&mut self, name: &ObjectName) -> Result<TextRules, Self::Error> {
        let name = name.0.last().and_then(ObjectNamePart::as_ident);
        Ok(self
            .collations
            .iter()
            .find(|(known, _)| name.is_some_and(|name| name.value == *known))
            .map_or(TextRules::UNKNOWN, |(_, rules)| *rules))
    }

    fn data_type(&self, data_type: &DataType) -> ScalarType {
        let DataType::Custom(name, _) = data_type else {
            return ScalarType::of(data_type);
        };
        let name = name.0.last().and_then(ObjectNamePart::as_ident);
        self.types
            .iter()
            .find(|(known, _)| name.is_some_and(|name| name.value == *known))
            .map_or_else(|| ScalarType::of(data_type), |(_, ty)| ty.clone())
    }

    fn cast_checks(&mut self, data_type: &DataType) -> Result<bool, Self::Error> {
        Ok(self.check(data_type).is_some())
    }

    fn checked_cast(
        &mut self,
        data_type: &DataType,
        converted: Evaluation,
    ) -> Result<Evaluation, Self::Error> {
        Ok(match self.check(data_type) {
            Some(check) => check(converted),
            None => converted,
        })
    }

    fn may_plan_as_constant(&self, reference: &Self::Reference) -> bool {
        self.plan_constants.contains(reference)
    }
}

/// A non-raising column of `ty` with unlisted values, nullable when `null`.
pub fn column(ty: ScalarType, null: bool) -> Evaluation {
    Evaluation::new(ty, null, false, Values::Unlisted(Unlisted::any()))
}

/// The expression of `SELECT expression`, parsed as PostgreSQL.
pub fn parse(expression: &str) -> Expr {
    let statements =
        Parser::parse_sql(&PostgreSqlDialect {}, &format!("SELECT {expression}")).expect("parses");
    let Statement::Query(query) = &statements[0] else {
        panic!("the test statement is a query");
    };
    let SetExpr::Select(select) = query.body.as_ref() else {
        panic!("the test query is a select");
    };
    let [SelectItem::UnnamedExpr(expr)] = select.projection.as_slice() else {
        panic!("the test select projects one expression");
    };
    expr.clone()
}

/// A callee that folds the built-in `builtin`, returns `returns`, is strict,
/// returns `NULL` only on `NULL` input and raises where its fold does.
pub fn builtin_callee(builtin: Builtin, returns: ScalarType) -> Callee {
    Callee::new(
        Some(builtin),
        NullInput::Strict,
        NullResult::OnlyOnNullInput,
        Volatility::Immutable,
        CalleeKind::Normal,
        returns,
        false,
        true,
        false,
        0,
    )
}
