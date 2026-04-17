use std::path::{Path, PathBuf};

use crate::build::CargoBuildProfile;
use crate::cargo::Cargo;
use crate::cli::{CliError, Result};
use crate::commands::generate::run_generate_java_with_output_from_source_dir;
use crate::config::{Config, Target};
use crate::pack::resolve_build_cargo_args;
use crate::reporter::Reporter;
use crate::target::JavaHostTarget;
use crate::toolchain::NativeHostToolchain;

use super::link::{build_jvm_native_library, compile_jni_library, resolve_jni_include_directories};
use super::outputs::{
    remove_stale_flat_jvm_outputs_if_current_host_unrequested,
    remove_stale_requested_jvm_shared_library_copies_after_success,
    remove_stale_structured_jvm_outputs,
};

#[derive(Debug, Clone)]
pub(crate) struct JvmCargoContext {
    pub(crate) host_target: JavaHostTarget,
    pub(crate) rust_target_triple: String,
    pub(crate) release: bool,
    pub(crate) build_profile: CargoBuildProfile,
    pub(crate) artifact_name: String,
    pub(crate) cargo_manifest_path: PathBuf,
    pub(crate) manifest_path: PathBuf,
    pub(crate) package_selector: Option<String>,
    pub(crate) target_directory: PathBuf,
    pub(crate) cargo_command_args: Vec<String>,
    pub(crate) toolchain_selector: Option<String>,
    pub(crate) crate_outputs: JvmCrateOutputs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct JvmCrateOutputs {
    pub(crate) builds_staticlib: bool,
    pub(crate) builds_cdylib: bool,
}

pub(crate) struct JvmPackagingTarget {
    pub(crate) cargo_context: JvmCargoContext,
    pub(crate) toolchain: NativeHostToolchain,
}

pub(crate) struct PreparedJavaPackaging {
    pub(crate) java_host_targets: Vec<JavaHostTarget>,
    pub(crate) packaging_targets: Vec<JvmPackagingTarget>,
}

impl JvmCargoContext {
    pub(crate) fn artifact_directory(&self) -> PathBuf {
        self.target_directory
            .join(&self.rust_target_triple)
            .join(self.build_profile.output_directory_name())
    }
}

pub(crate) fn check_java_packaging_prereqs(
    config: &Config,
    release: bool,
    cargo_args: &[String],
) -> Result<()> {
    prepare_java_packaging(config, release, cargo_args).map(|_| ())
}

pub(crate) fn pack_java(
    config: &Config,
    options: crate::commands::pack::PackJavaOptions,
    prepared: Option<PreparedJavaPackaging>,
    reporter: &Reporter,
) -> Result<()> {
    if !config.is_java_jvm_enabled() {
        return Err(CliError::CommandFailed {
            command: "targets.java.jvm.enabled = false".to_string(),
            status: None,
        });
    }

    reporter.section("☕", "Packing Java");

    ensure_java_no_build_supported(
        config,
        options.execution.no_build,
        options.experimental,
        "pack java",
    )?;

    let PreparedJavaPackaging {
        java_host_targets,
        packaging_targets,
    } = if let Some(prepared) = prepared {
        prepared
    } else {
        let step = reporter.step("Validating JVM toolchains");
        let prepared = prepare_java_packaging(
            config,
            options.execution.release,
            &options.execution.cargo_args,
        )?;
        step.finish_success();
        prepared
    };

    if options.execution.regenerate {
        let source_directory = selected_jvm_package_source_directory(&packaging_targets)?;
        let artifact_name = selected_jvm_package_artifact_name(&packaging_targets)?;
        let step = reporter.step("Generating C header");
        generate_java_header(config, &source_directory, artifact_name)?;
        step.finish_success();

        let step = reporter.step("Generating Java bindings");
        run_generate_java_with_output_from_source_dir(
            config,
            Some(config.java_jvm_output()),
            &source_directory,
            artifact_name,
        )?;
        step.finish_success();
    }

    let mut packaged_outputs = Vec::with_capacity(packaging_targets.len());
    for packaging_target in &packaging_targets {
        let host_target = packaging_target.cargo_context.host_target;
        let step = reporter.step(&format!(
            "Building Rust library for {}",
            host_target.canonical_name()
        ));
        let build_artifacts =
            build_jvm_native_library(packaging_target, options.execution.release, &step)?;
        step.finish_success();

        let step = reporter.step(&format!(
            "Compiling JNI library for {}",
            host_target.canonical_name()
        ));
        packaged_outputs.push(compile_jni_library(
            config,
            packaging_target,
            &build_artifacts,
            &step,
        )?);
        step.finish_success();
    }

    let artifact_name = selected_jvm_package_artifact_name(&packaging_targets)?;
    remove_stale_requested_jvm_shared_library_copies_after_success(
        &config.java_jvm_output(),
        &packaged_outputs,
        artifact_name,
    )?;
    remove_stale_structured_jvm_outputs(
        &config.java_jvm_output().join("native"),
        &java_host_targets,
    )?;
    remove_stale_flat_jvm_outputs_if_current_host_unrequested(
        &config.java_jvm_output(),
        JavaHostTarget::current(),
        &java_host_targets,
        artifact_name,
    )?;

    reporter.finish();
    Ok(())
}

pub(crate) fn prepare_java_packaging(
    config: &Config,
    release: bool,
    cargo_args: &[String],
) -> Result<PreparedJavaPackaging> {
    let build_cargo_args = resolve_build_cargo_args(config, cargo_args);
    ensure_java_pack_cargo_args_supported(&build_cargo_args)?;
    let build_profile = crate::build::resolve_build_profile(release, &build_cargo_args);
    let java_host_targets = resolve_java_host_targets_for_packaging(config)?;
    let packaging_targets = resolve_jvm_packaging_targets(
        config,
        &build_cargo_args,
        release,
        build_profile,
        &java_host_targets,
    )?;

    Ok(PreparedJavaPackaging {
        java_host_targets,
        packaging_targets,
    })
}

pub(crate) fn ensure_java_no_build_supported(
    config: &Config,
    no_build: bool,
    experimental: bool,
    command_name: &str,
) -> Result<()> {
    if no_build && config.should_process(Target::Java, experimental) {
        return Err(CliError::CommandFailed {
            command: format!(
                "{command_name} --no-build is unsupported in Phase 4 when JVM packaging is enabled; rerun without --no-build"
            ),
            status: None,
        });
    }

    Ok(())
}

pub(crate) fn ensure_java_pack_cargo_args_supported(cargo_args: &[String]) -> Result<()> {
    if let Some(target_selector) = Cargo::current(cargo_args)?.target_selector() {
        return Err(CliError::CommandFailed {
            command: format!(
                "pack java resolves desktop targets from targets.java.jvm.host_targets; remove cargo --target '{}'",
                target_selector
            ),
            status: None,
        });
    }

    Ok(())
}

pub(crate) fn selected_jvm_package_source_directory(
    packaging_targets: &[JvmPackagingTarget],
) -> Result<PathBuf> {
    packaging_targets
        .first()
        .and_then(|target| target.cargo_context.manifest_path.parent())
        .map(Path::to_path_buf)
        .ok_or_else(|| CliError::CommandFailed {
            command: "could not resolve selected Cargo package source directory for JVM generation"
                .to_string(),
            status: None,
        })
}

fn selected_jvm_package_artifact_name(packaging_targets: &[JvmPackagingTarget]) -> Result<&str> {
    packaging_targets
        .first()
        .map(|target| target.cargo_context.artifact_name.as_str())
        .ok_or_else(|| CliError::CommandFailed {
            command: "could not resolve selected Cargo package artifact name for JVM generation"
                .to_string(),
            status: None,
        })
}

pub(crate) fn generate_java_header(
    config: &Config,
    source_directory: &Path,
    crate_name: &str,
) -> Result<()> {
    use boltffi_bindgen::{CHeaderLowerer, ir, scan_crate_with_pointer_width};

    let output_directory = config.java_jvm_output().join("jni");
    let output_path = output_directory.join(format!("{crate_name}.h"));

    std::fs::create_dir_all(&output_directory).map_err(|source| {
        CliError::CreateDirectoryFailed {
            path: output_directory.clone(),
            source,
        }
    })?;
    let host_pointer_width_bits = match usize::BITS {
        32 => Some(32),
        64 => Some(64),
        _ => None,
    };
    let mut module =
        scan_crate_with_pointer_width(source_directory, crate_name, host_pointer_width_bits)
            .map_err(|error| CliError::CommandFailed {
                command: format!("scan_crate: {}", error),
                status: None,
            })?;

    let contract = ir::build_contract(&mut module);
    if let Err(errors) = ir::validate_contract(&contract) {
        let report = errors
            .iter()
            .map(|e| format!("  - {e:?}"))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(CliError::CommandFailed {
            command: format!(
                "validate_contract: {} error(s)\n{report}",
                errors.len()
            ),
            status: None,
        });
    }
    let abi = ir::Lowerer::new(&contract).to_abi_contract();
    let header_code = CHeaderLowerer::new(&contract, &abi).generate();
    std::fs::write(&output_path, header_code).map_err(|source| CliError::WriteFailed {
        path: output_path,
        source,
    })?;

    Ok(())
}

fn resolve_java_host_targets_for_packaging(config: &Config) -> Result<Vec<JavaHostTarget>> {
    config
        .java_jvm_host_targets()
        .map_err(|message| CliError::CommandFailed {
            command: message,
            status: None,
        })
}

fn resolve_jvm_packaging_targets(
    config: &Config,
    build_cargo_args: &[String],
    release: bool,
    build_profile: CargoBuildProfile,
    host_targets: &[JavaHostTarget],
) -> Result<Vec<JvmPackagingTarget>> {
    let current_host = JavaHostTarget::current().ok_or_else(|| CliError::CommandFailed {
        command:
            "JVM packaging is only supported on darwin-arm64, darwin-x86_64, linux-x86_64, linux-aarch64, and windows-x86_64 hosts".to_string(),
        status: None,
    })?;
    let cargo = Cargo::current(build_cargo_args)?;
    let metadata = cargo.metadata()?;
    let cargo_manifest_path = cargo.manifest_path()?;
    let package_selector =
        cargo.effective_package_selector(config, &metadata, &cargo_manifest_path);
    let package = metadata.find_package(&cargo_manifest_path, package_selector.as_deref())?;
    let artifact_name = package
        .resolve_library_artifact_name(&config.crate_artifact_name(), &cargo_manifest_path)?
        .to_string();
    let toolchain_selector = cargo.toolchain_selector().map(str::to_owned);
    let cargo_command_args = cargo.probe_command_arguments();
    let crate_outputs = JvmCrateOutputs::from_metadata(
        &metadata,
        &artifact_name,
        &cargo_manifest_path,
        package_selector.as_deref(),
    )?;

    host_targets
        .iter()
        .copied()
        .map(|host_target| {
            let toolchain = NativeHostToolchain::discover(
                toolchain_selector.as_deref(),
                &cargo_command_args,
                host_target,
                current_host,
            )?;
            let cargo_context = JvmCargoContext {
                host_target,
                rust_target_triple: toolchain.rust_target_triple().to_string(),
                release,
                build_profile: build_profile.clone(),
                artifact_name: artifact_name.clone(),
                cargo_manifest_path: cargo_manifest_path.clone(),
                manifest_path: package.manifest_path.clone(),
                package_selector: package_selector.clone(),
                target_directory: metadata.target_directory.clone(),
                cargo_command_args: cargo_command_args.clone(),
                toolchain_selector: toolchain_selector.clone(),
                crate_outputs,
            };
            let _ = resolve_jni_include_directories(&cargo_context)?;
            Ok(JvmPackagingTarget {
                cargo_context,
                toolchain,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{
        JvmCargoContext, JvmCrateOutputs, JvmPackagingTarget, ensure_java_no_build_supported,
        ensure_java_pack_cargo_args_supported, selected_jvm_package_source_directory,
    };
    use crate::build::CargoBuildProfile;
    use crate::cli::CliError;
    use crate::config::{CargoConfig, Config, PackageConfig, TargetsConfig};
    use crate::target::JavaHostTarget;
    use crate::toolchain::NativeHostToolchain;

    fn config(java_enabled: bool) -> Config {
        Config {
            experimental: Vec::new(),
            cargo: CargoConfig::default(),
            package: PackageConfig {
                name: "workspace-member".to_string(),
                crate_name: None,
                version: None,
                description: None,
                license: None,
                repository: None,
            },
            targets: TargetsConfig {
                java: crate::config::JavaConfig {
                    jvm: crate::config::JavaJvmConfig {
                        enabled: java_enabled,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                ..Default::default()
            },
        }
    }

    #[test]
    fn rejects_pack_all_no_build_when_java_is_enabled() {
        let error = ensure_java_no_build_supported(&config(true), true, false, "pack all")
            .expect_err("expected no-build rejection");

        assert!(matches!(
            error,
            CliError::CommandFailed { command, status: None }
                if command.contains("pack all --no-build is unsupported in Phase 4")
        ));
    }

    #[test]
    fn allows_pack_all_no_build_when_java_is_disabled() {
        ensure_java_no_build_supported(&config(false), true, false, "pack all")
            .expect("expected no-build to be allowed");
    }

    #[test]
    fn rejects_explicit_cargo_target_for_pack_java() {
        let error = ensure_java_pack_cargo_args_supported(&[
            "--target".to_string(),
            "x86_64-unknown-linux-gnu".to_string(),
        ])
        .expect_err("expected explicit target rejection");

        assert!(matches!(
            error,
            CliError::CommandFailed { command, status: None }
                if command.contains("remove cargo --target 'x86_64-unknown-linux-gnu'")
        ));
    }

    #[test]
    fn pack_java_no_longer_requires_experimental_gate() {
        ensure_java_no_build_supported(&config(true), false, false, "pack java")
            .expect("expected pack java to proceed without experimental gate");
    }

    #[test]
    fn resolves_selected_jvm_package_source_directory_from_selected_package_manifest() {
        let current_host = JavaHostTarget::current().expect("current host");
        let packaging_targets = vec![JvmPackagingTarget {
            cargo_context: JvmCargoContext {
                host_target: current_host,
                rust_target_triple: "x86_64-unknown-linux-gnu".to_string(),
                release: false,
                build_profile: CargoBuildProfile::Debug,
                artifact_name: "workspace_member".to_string(),
                cargo_manifest_path: PathBuf::from("/tmp/workspace/Cargo.toml"),
                manifest_path: PathBuf::from("/tmp/workspace/member/Cargo.toml"),
                package_selector: Some("workspace-member".to_string()),
                target_directory: PathBuf::from("/tmp/boltffi-target"),
                cargo_command_args: Vec::new(),
                toolchain_selector: None,
                crate_outputs: JvmCrateOutputs {
                    builds_staticlib: true,
                    builds_cdylib: true,
                },
            },
            toolchain: NativeHostToolchain::discover(None, &[], current_host, current_host)
                .expect("native host toolchain"),
        }];

        let source_directory =
            selected_jvm_package_source_directory(&packaging_targets).expect("source directory");

        assert_eq!(source_directory, PathBuf::from("/tmp/workspace/member"));
    }
}
