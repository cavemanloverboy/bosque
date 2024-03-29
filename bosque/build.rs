use std::env;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    cbindgen::Builder::new()
        .with_crate(crate_dir.clone())
        .with_language(cbindgen::Language::C)
        .with_autogen_warning("/* These bindings are autogenerated via cbindgen */")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("../examples/c/bosque.h");

    cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_language(cbindgen::Language::Cxx)
        .with_autogen_warning("/* These bindings are autogenerated via cbindgen */")
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file("../examples/c/bosque.hpp");
}
