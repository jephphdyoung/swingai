//! Guards the rule from ADR 0001: `swingai-core` and `swingai-contracts` stay
//! platform-neutral, so the contract types can be developed and tested on the
//! Fedora box while the capture runtime targets Windows 11.
//!
//! That these crates *compile* on Linux is proven by this test file existing —
//! it does not build otherwise. What compiling cannot catch is a
//! `#[cfg(windows)]` arm sneaking in, which builds fine here and quietly makes
//! the two platforms behave differently. Hence the source scan.
//!
//! When genuinely platform-specific code arrives it goes in a capture crate
//! behind a trait, and this test keeps it from drifting back down into the
//! shared layer.

use std::path::{Path, PathBuf};

/// Markers that mean "this file behaves differently depending on the OS".
const PLATFORM_MARKERS: &[&str] = &[
    "cfg(windows)",
    "cfg(unix)",
    "cfg(target_os",
    "cfg(target_family",
    "std::os::windows",
    "std::os::unix",
    "windows_sys",
    "windows-sys",
    "winapi",
];

fn crate_src(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crates/")
        .join(name)
        .join("src")
}

fn rust_files(dir: &Path, into: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("reading {dir:?}: {e}")) {
        let path = entry.expect("directory entry").path();
        if path.is_dir() {
            rust_files(&path, into);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            into.push(path);
        }
    }
}

#[test]
fn shared_crates_contain_no_platform_specific_code() {
    let mut files = Vec::new();
    for crate_name in ["swingai-core", "swingai-contracts"] {
        rust_files(&crate_src(crate_name), &mut files);
    }
    assert!(!files.is_empty(), "found no sources to scan");

    let mut offenders = Vec::new();
    for file in &files {
        let source = std::fs::read_to_string(file).expect("source is readable");
        for (number, line) in source.lines().enumerate() {
            // The rule is about behaviour, not prose — doc comments name Windows
            // and Linux deliberately.
            if line.trim_start().starts_with("//") {
                continue;
            }
            for marker in PLATFORM_MARKERS {
                if line.contains(marker) {
                    offenders.push(format!("{}:{}: {marker}", file.display(), number + 1));
                }
            }
        }
    }

    assert!(
        offenders.is_empty(),
        "platform-specific code belongs in a capture crate behind a trait, not in the \
         shared layer (ADR 0001):\n  {}",
        offenders.join("\n  ")
    );
}

#[test]
fn shared_crates_stay_on_the_platform_neutral_dependency_allow_list() {
    for crate_name in ["swingai-core", "swingai-contracts"] {
        let manifest_path = crate_src(crate_name).parent().unwrap().join("Cargo.toml");
        let manifest = std::fs::read_to_string(&manifest_path).expect("manifest is readable");

        // Everything a platform-neutral crate may name. Anything else is either
        // a platform binding or a dependency worth arguing about in review.
        //
        // `time` is here for RFC 3339 parsing, with default features off — its
        // timezone-database and system-clock features are exactly the
        // platform-specific behaviour this test exists to keep out.
        let allowed = ["serde", "serde_json", "swingai-core", "time"];
        let dependency_lines = manifest
            .lines()
            .skip_while(|line| !line.starts_with("[dependencies]"))
            .skip(1)
            .take_while(|line| !line.starts_with('['))
            .filter(|line| line.contains('='));

        for line in dependency_lines {
            let name = line.split('=').next().unwrap().trim();
            assert!(
                allowed.contains(&name),
                "{crate_name} depends on {name:?}, which is not on the platform-neutral \
                 allow-list {allowed:?} — see ADR 0001 before adding it here"
            );
        }
    }
}

#[test]
fn the_date_time_dependency_does_not_pull_in_platform_behaviour() {
    let workspace_manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("native/")
        .join("Cargo.toml");
    let manifest = std::fs::read_to_string(&workspace_manifest).expect("manifest is readable");

    let time_line = manifest
        .lines()
        .find(|line| line.trim_start().starts_with("time = "))
        .expect("the workspace declares the `time` dependency");

    // Default features bring in local-offset handling and a system clock; with
    // them off, all that is left is parsing, which is pure computation.
    assert!(
        time_line.contains("default-features = false"),
        "`time` must not use default features: {time_line}"
    );
    for banned in ["local-offset", "tzdb", "wasm"] {
        assert!(
            !time_line.contains(banned),
            "`time` must not enable {banned:?}: {time_line}"
        );
    }
}
