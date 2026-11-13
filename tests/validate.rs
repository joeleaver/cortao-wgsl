//! Every reference composition parses and validates as WGSL (naga), with no GPU needed.

use cortao_wgsl::{BASE, FILTER_PASS, MAIN_PASS, PREPASS, compose, reference as r};

fn validate(name: &str, parts: &[&str]) {
    let source = compose(parts);
    let module = naga::front::wgsl::parse_str(&source).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&source)));
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&source)));
}

#[test]
fn reference_passes_validate() {
    validate("prepass", &[BASE, PREPASS, r::PREPASS]);
    validate("prepass continue", &[BASE, PREPASS, r::PREPASS_CONTINUE]);
    validate("main pass, occlusion", &[BASE, MAIN_PASS, r::MAIN_PASS_COMMON, r::MAIN_PASS_OCCLUSION]);
    validate("main pass, bent normal", &[BASE, MAIN_PASS, r::MAIN_PASS_COMMON, r::MAIN_PASS_BENT_NORMAL]);
    validate("main pass, packed bent normal", &[BASE, MAIN_PASS, r::MAIN_PASS_COMMON, r::MAIN_PASS_PACKED_BENT_NORMAL]);
    validate("filter pass, occlusion", &[BASE, FILTER_PASS, r::FILTER_PASS_COMMON, r::FILTER_PASS_OCCLUSION]);
    validate("filter pass, bent normal", &[BASE, FILTER_PASS, r::FILTER_PASS_COMMON, r::FILTER_PASS_BENT_NORMAL]);
}
