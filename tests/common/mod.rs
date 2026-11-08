//! The context and helpers both integration test files read the public API
//! through.

use sqlparser::{
    ast::{Expr, Function, ObjectNamePart, Query, SelectItem, SetExpr, Statement},
    dialect::PostgreSqlDialect,
    parser::{Parser, ParserError},
};
use sqlparser_expr_eval::{
    Builtin, Callee, CalleeKind, Context, Evaluation, Guarantees, NullInput, QueryEvaluation,
    Resolved, ScalarType, Unlisted, Values, Volatility,
};

/// A context that resolves a fixed set of columns and callees and one nested
/// query.
pub struct Ctx {
    pub columns: Vec<(String, Evaluation)>,
    pub callees: Vec<(String, Callee)>,
    pub query: Option<QueryEvaluation>,
}

impl Ctx {
    /// A context resolving `columns` and nothing else.
    pub fn with_columns(columns: Vec<(String, Evaluation)>) -> Self {
        Self { columns, callees: Vec::new(), query: None }
    }

    /// The context with `callee` resolving calls to `name`.
    pub fn with_callee(mut self, name: &str, callee: Callee) -> Self {
        self.callees.push((name.to_string(), callee));
        self
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
}

/// A non-raising column of `ty` with unlisted values, nullable when `null`.
pub fn column(ty: ScalarType, null: bool) -> Evaluation {
    Evaluation::new(ty, null, false, Values::Unlisted(Unlisted::any()))
}

/// The expression of `SELECT expression`, parsed as PostgreSQL.
#[expect(clippy::expect_used, clippy::panic, reason = "a malformed test expression fails the test")]
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

/// A callee that folds the built-in `builtin` and returns `returns`.
pub fn builtin_callee(builtin: Builtin, returns: ScalarType) -> Callee {
    Callee::new(
        Some(builtin),
        NullInput::Strict,
        Volatility::Immutable,
        CalleeKind::Normal,
        returns,
        false,
        0,
    )
}
