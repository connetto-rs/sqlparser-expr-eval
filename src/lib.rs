#![doc = include_str!("../README.md")]
// Tests link the standard library whatever the features, as their harness does.
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

#[macro_use]
extern crate alloc;

mod backward;
mod context;
mod dialect;
mod domain;
mod forward;
mod guarantees;
mod scalar;
mod support;
#[cfg(test)]
mod test_context;

pub use backward::implied;
pub use context::{
    Builtin, Callee, CalleeKind, Context, NullInput, QueryEvaluation, Resolved, Rows, Volatility,
};
pub use dialect::EvalDialect;
pub use domain::{Evaluation, Interval, Outcomes, Unlisted, Values};
pub use forward::{
    evaluate,
    fold::{Comparison, RowKinds},
};
pub use guarantees::{Guarantees, Subject};
pub use scalar::{ScalarType, TextRules};
pub use sql_values;
