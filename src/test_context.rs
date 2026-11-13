//! A test context that resolves named columns and callees.

use alloc::{collections::BTreeMap, string::String, vec::Vec};

use sqlparser::{
    ast::{Expr, Function, Query, SelectItem, SetExpr, Statement},
    dialect::PostgreSqlDialect,
    parser::{Parser, ParserError},
};

use super::{
    context::{Builtin, Callee, Context, QueryEvaluation, Resolved, Rows, Volatility},
    domain::Evaluation,
    guarantees::Guarantees,
    scalar::ScalarType,
};

/// The columns and callees a test resolves.
pub struct ColumnContext {
    pub columns: BTreeMap<String, Evaluation>,
    pub callees: BTreeMap<String, Vec<Callee>>,
    pub query: Option<QueryEvaluation>,
    pub bodies: BTreeMap<usize, Evaluation>,
}

impl ColumnContext {
    /// A context with no columns and no callees.
    #[must_use]
    pub fn new() -> Self {
        Self {
            columns: BTreeMap::new(),
            callees: BTreeMap::new(),
            query: None,
            bodies: BTreeMap::new(),
        }
    }

    /// The context with the column `name` holding `evaluation`.
    #[must_use]
    pub fn with_column(mut self, name: &str, evaluation: Evaluation) -> Self {
        self.columns.insert(name.to_string(), evaluation);
        self
    }

    /// The context with the callee `name` among the ones `call` may reach.
    #[must_use]
    pub fn with_callee(mut self, name: &str, callee: Callee) -> Self {
        self.callees.entry(name.to_string()).or_default().push(callee);
        self
    }

    /// The context whose nested query projects `column` over `rows`.
    #[must_use]
    pub fn with_query(mut self, rows: Rows, column: Evaluation) -> Self {
        self.query = Some(QueryEvaluation { rows, columns: Some(vec![column]), error: false });
        self
    }

    /// The context whose callee with token `token` reads as `evaluation`
    /// from its body.
    #[must_use]
    pub fn with_body(mut self, token: usize, evaluation: Evaluation) -> Self {
        self.bodies.insert(token, evaluation);
        self
    }
}

impl Context for ColumnContext {
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
        Ok(self
            .columns
            .get(&name)
            .cloned()
            .map(|evaluation| Resolved { reference: name, evaluation }))
    }

    fn callees(&mut self, call: &Function) -> Result<Vec<Callee>, Self::Error> {
        let Some(name) = call
            .name
            .0
            .last()
            .and_then(sqlparser::ast::ObjectNamePart::as_ident)
            .map(|ident| ident.value.to_lowercase())
        else {
            return Ok(Vec::new());
        };
        Ok(self.callees.get(&name).cloned().unwrap_or_default())
    }

    fn query(
        &mut self,
        query: &Query,
        guarantees: &Guarantees<Self::Reference>,
    ) -> Result<QueryEvaluation, Self::Error> {
        let _ = (query, guarantees);
        Ok(self.query.clone().unwrap_or_else(QueryEvaluation::unknown))
    }

    fn evaluate_body(
        &mut self,
        callee: &Callee,
        arguments: &[Evaluation],
    ) -> Result<Option<Evaluation>, Self::Error> {
        let _ = arguments;
        Ok(self.bodies.get(&callee.token).cloned())
    }
}

/// The expression of `SELECT expression`.
pub fn parse(expression: &str) -> Expr {
    parse_statement(&format!("SELECT {expression}"))
}

/// The expression a statement projects, which it must spell as
/// `SELECT expression` of a single expression.
pub fn parse_statement(statement: &str) -> Expr {
    let statements =
        Parser::parse_sql(&PostgreSqlDialect {}, statement).expect("the test statement parses");
    let Statement::Query(query) = &statements[0] else { panic!("the test statement is a query") };
    let SetExpr::Select(select) = query.body.as_ref() else { panic!("the test query is a select") };
    let [SelectItem::UnnamedExpr(expr)] = select.projection.as_slice() else {
        panic!("the test select projects one expression");
    };
    expr.clone()
}

/// Reads `sql` under PostgreSQL with no settings.
pub fn read(mut context: ColumnContext, sql: &str) -> Evaluation {
    let options = crate::postgres::PostgreSqlOptions::default();
    crate::forward::evaluate(
        &parse(sql),
        &PostgreSqlDialect {},
        &options,
        &mut context,
        &Guarantees::new(),
    )
    .expect("the context resolves")
}

/// Reads `sql` under the conservative generic dialect, with no settings.
pub fn read_default(mut context: ColumnContext, sql: &str) -> Evaluation {
    let dialect = sqlparser::dialect::GenericDialect {};
    crate::forward::evaluate(&parse(sql), &dialect, &(), &mut context, &Guarantees::new())
        .expect("the context resolves")
}

/// A callee that returns `returns` and is strict.
pub fn strict_callee(returns: ScalarType) -> Callee {
    Callee {
        builtin: None,
        null_input: super::context::NullInput::Strict,
        volatility: Volatility::Immutable,
        kind: super::context::CalleeKind::Normal,
        returns,
        returns_set: false,
        token: 0,
    }
}

/// A callee that folds the built-in `builtin` and returns `returns`.
pub fn builtin_callee(builtin: Builtin, returns: ScalarType) -> Callee {
    Callee {
        builtin: Some(builtin),
        null_input: super::context::NullInput::Strict,
        volatility: Volatility::Immutable,
        kind: super::context::CalleeKind::Normal,
        returns,
        returns_set: false,
        token: 0,
    }
}

#[cfg(test)]
mod tests {
    use sqlparser::ast::{
        Expr, Function, FunctionArgumentList, FunctionArguments, NullTreatment, ObjectName,
        ObjectNamePart,
    };

    use super::*;
    use crate::domain::Values;

    fn expect_panic(statement: &str, message: &str) {
        let result = std::panic::catch_unwind(|| parse_statement(statement));
        let error = result.unwrap_err();
        let text = error
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| error.downcast_ref::<&str>().copied())
            .unwrap_or_default();
        assert!(text.contains(message), "the panic is {text}, want {message:?}");
    }

    #[test]
    fn parse_statement_refuses_other_statements() {
        expect_panic("CREATE TABLE t (x integer)", "the test statement is a query");
    }

    #[test]
    fn parse_statement_refuses_set_operations() {
        expect_panic("SELECT 1 UNION SELECT 2", "the test query is a select");
    }

    #[test]
    fn parse_statement_refuses_several_projections() {
        expect_panic("SELECT 1, 2", "the test select projects one expression");
    }

    #[test]
    fn a_single_part_compound_identifier_resolves() {
        let mut context = ColumnContext::new().with_column(
            "x",
            Evaluation::known(
                sql_values::postgres::PgValue::Integer(1),
                ScalarType::Integer(sql_values::postgres::IntegerType::Integer),
            ),
        );
        let evaluation = crate::forward::evaluate(
            &Expr::CompoundIdentifier(vec![sqlparser::ast::Ident::new("x")]),
            &sqlparser::dialect::GenericDialect,
            &(),
            &mut context,
            &Guarantees::new(),
        )
        .unwrap();
        assert_eq!(
            evaluation.values(),
            &Values::Listed(alloc::collections::BTreeSet::from([
                sql_values::postgres::PgValue::Integer(1)
            ]))
        );
        assert!(!evaluation.may_be_null());
    }

    #[test]
    fn a_two_part_compound_identifier_is_unresolved() {
        let evaluation = crate::forward::evaluate(
            &Expr::CompoundIdentifier(vec![
                sqlparser::ast::Ident::new("s"),
                sqlparser::ast::Ident::new("x"),
            ]),
            &sqlparser::dialect::GenericDialect,
            &(),
            &mut ColumnContext::new(),
            &Guarantees::new(),
        )
        .unwrap();
        assert!(matches!(evaluation.ty(), ScalarType::Unknown | ScalarType::UnknownLiteral));
    }

    #[test]
    fn a_call_named_by_an_object_name_is_unresolved() {
        let call = Expr::Function(Function {
            name: ObjectName(vec![ObjectNamePart::Function(
                sqlparser::ast::ObjectNamePartFunction {
                    name: sqlparser::ast::Ident::new("f"),
                    args: Vec::new(),
                },
            )]),
            uses_odbc_syntax: false,
            parameters: FunctionArguments::None,
            args: FunctionArguments::List(FunctionArgumentList {
                duplicate_treatment: None,
                args: Vec::new(),
                clauses: Vec::new(),
            }),
            within_group: Vec::new(),
            filter: None,
            null_treatment: Some(NullTreatment::IgnoreNulls),
            over: None,
        });
        let evaluation = crate::forward::evaluate(
            &call,
            &sqlparser::dialect::GenericDialect,
            &(),
            &mut ColumnContext::new(),
            &Guarantees::new(),
        )
        .unwrap();
        assert!(matches!(evaluation.ty(), ScalarType::Unknown | ScalarType::UnknownLiteral));
    }

    #[test]
    fn the_callee_helpers_build_strict_callees() {
        assert_eq!(strict_callee(ScalarType::Boolean).builtin, None);
        assert_eq!(
            builtin_callee(Builtin::Upper, ScalarType::Text(crate::TextRules::DEFAULT)).builtin,
            Some(Builtin::Upper)
        );
    }
}
