#![doc = include_str!("../README.md")]
// Tests link the standard library whatever the features, as their harness does.
#![cfg_attr(not(any(feature = "std", test)), no_std)]
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

extern crate alloc;

mod scalar;

pub use scalar::{ScalarType, TextRules};
pub use sql_values;
