use boltffi_ffi_rules::naming;
use indexmap::IndexMap;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use syn::{
    Attribute, Fields, FnArg, ImplItem, Item, ItemEnum, ItemImpl, ItemStruct, ItemTrait, Type,
};
use walkdir::WalkDir;

use crate::model::{
    BuiltinId, CallbackTrait, Class, ClosureSignature as MClosureSignature, Constructor,
    ConstructorParam, CustomType, Enumeration, Function, Method, Module, Parameter, Primitive,
    Receiver, Record, RecordField, ReturnType, StreamMethod, StreamMode, TraitMethod,
    TraitMethodParam, Type as MType, Variant,
};

mod compiler_type_resolution;

pub enum PendingKind {
    Record,
    Enum,
    Class,
    Callback,
}

pub enum TypeShape {
    Pending(PendingKind),
    Record {
        fields: Vec<RecordField>,
        is_repr_c: bool,
        is_error: bool,
        constructors: Vec<Constructor>,
        methods: Vec<Method>,
    },
    Enum {
        variants: Vec<Variant>,
        is_error: bool,
        repr_type: Option<Primitive>,
        constructors: Vec<Constructor>,
        methods: Vec<Method>,
    },
    Class {
        constructors: Vec<Constructor>,
        methods: Vec<Method>,
        streams: Vec<StreamMethod>,
    },
    Custom {
        repr: MType,
    },
}

pub struct TypeMeta {
    pub doc: Option<String>,
    pub shape: TypeShape,
}

/// A user-defined generic struct (e.g. `Request<T>`) with `#[boltffi::data]`.
///
/// Stored verbatim in syn form so that, when the scanner encounters a
/// concrete use-site like `Request<MyEffect>`, it can substitute the type
/// parameters and synthesize a concrete `#[data]`-shaped record in the
/// IR without ever emitting the generic form itself.
#[derive(Clone)]
pub struct GenericStructTemplate {
    pub type_params: Vec<String>,
    pub fields: Vec<GenericTemplateField>,
    pub is_repr_c: bool,
    pub is_error: bool,
    pub doc: Option<String>,
}

#[derive(Clone)]
pub struct GenericTemplateField {
    pub name: String,
    pub ty: syn::Type,
    pub doc: Option<String>,
    pub default: Option<String>,
}

/// A use-site like `Request<MyEffect>` the scanner has seen but not yet
/// materialized into a concrete record. Keyed by mangled name in
/// `SourceScanner::pending_generic_instantiations`.
#[derive(Clone)]
pub struct PendingInstantiation {
    pub base: String,
    pub args: Vec<syn::Type>,
}

type GenericTemplates = HashMap<String, GenericStructTemplate>;
type PendingGenericInstantiations = RefCell<IndexMap<String, PendingInstantiation>>;

/// Read-only bundle of everything `rust_type_to_ffi_type` needs to do
/// its job. Kept as a struct so that future additions (new lookup
/// tables, diagnostic channels, etc.) do not force changes at every
/// call site.
struct ResolveCtx<'a> {
    registry: &'a TypeRegistry,
    alias_resolver: &'a AliasResolver,
    compiler_canonical_types: &'a HashMap<String, String>,
    self_type_name: Option<&'a str>,
    generic_templates: &'a GenericTemplates,
    pending_instantiations: &'a PendingGenericInstantiations,
}


#[derive(Default)]
pub struct TypeRegistry {
    types: IndexMap<String, TypeMeta>,
    custom_type_names_by_remote_key: HashMap<CustomTypeLookupKey, String>,
}

impl TypeRegistry {
    pub fn is_enum(&self, name: &str) -> bool {
        self.types.get(name).is_some_and(|meta| {
            matches!(
                meta.shape,
                TypeShape::Pending(PendingKind::Enum) | TypeShape::Enum { .. }
            )
        })
    }

    pub fn contains(&self, name: &str) -> bool {
        self.types.contains_key(name)
    }

    pub fn doc(&self, name: &str) -> Option<&str> {
        self.types.get(name).and_then(|meta| meta.doc.as_deref())
    }

    pub fn register(&mut self, name: String, meta: TypeMeta) {
        self.types.insert(name, meta);
    }

    pub fn fill(&mut self, name: &str, shape: TypeShape) {
        if let Some(meta) = self.types.get_mut(name) {
            meta.shape = shape;
        }
    }

    pub fn fill_record_fields(
        &mut self,
        name: &str,
        fields: Vec<RecordField>,
        is_repr_c: bool,
        is_error: bool,
    ) {
        let Some(meta) = self.types.get_mut(name) else {
            return;
        };
        match &mut meta.shape {
            TypeShape::Record {
                fields: existing_fields,
                is_repr_c: existing_repr_c,
                is_error: existing_is_error,
                ..
            } => {
                *existing_fields = fields;
                *existing_repr_c = is_repr_c;
                *existing_is_error = is_error;
            }
            _ => {
                meta.shape = TypeShape::Record {
                    fields,
                    is_repr_c,
                    is_error,
                    constructors: Vec::new(),
                    methods: Vec::new(),
                };
            }
        }
    }

    pub fn set_doc(&mut self, name: &str, doc: String) {
        if let Some(meta) = self.types.get_mut(name) {
            meta.doc = Some(doc);
        }
    }

    pub fn is_record(&self, name: &str) -> bool {
        self.types.get(name).is_some_and(|meta| {
            matches!(
                meta.shape,
                TypeShape::Pending(PendingKind::Record) | TypeShape::Record { .. }
            )
        })
    }

    pub fn merge_record_impl(
        &mut self,
        name: &str,
        constructors: Vec<Constructor>,
        methods: Vec<Method>,
    ) {
        let Some(meta) = self.types.get_mut(name) else {
            return;
        };
        match &mut meta.shape {
            TypeShape::Record {
                constructors: existing_ctors,
                methods: existing_methods,
                ..
            } => {
                existing_ctors.extend(constructors);
                existing_methods.extend(methods);
            }
            TypeShape::Pending(PendingKind::Record) => {
                meta.shape = TypeShape::Record {
                    fields: Vec::new(),
                    is_repr_c: true,
                    is_error: false,
                    constructors,
                    methods,
                };
            }
            _ => {}
        }
    }

    pub fn merge_enum_impl(
        &mut self,
        name: &str,
        constructors: Vec<Constructor>,
        methods: Vec<Method>,
    ) {
        let Some(meta) = self.types.get_mut(name) else {
            return;
        };
        match &mut meta.shape {
            TypeShape::Enum {
                constructors: existing_ctors,
                methods: existing_methods,
                ..
            } => {
                existing_ctors.extend(constructors);
                existing_methods.extend(methods);
            }
            TypeShape::Pending(PendingKind::Enum) => {
                meta.shape = TypeShape::Enum {
                    variants: Vec::new(),
                    is_error: false,
                    repr_type: None,
                    constructors,
                    methods,
                };
            }
            _ => {}
        }
    }

    pub fn drain(self) -> impl Iterator<Item = (String, TypeMeta)> {
        self.types.into_iter()
    }

    pub fn classify_named_type(&self, name: &str) -> Option<MType> {
        let meta = self.types.get(name)?;
        Some(match &meta.shape {
            TypeShape::Pending(PendingKind::Record) | TypeShape::Record { .. } => {
                MType::Record(name.to_string())
            }
            TypeShape::Pending(PendingKind::Enum) | TypeShape::Enum { .. } => {
                MType::Enum(name.to_string())
            }
            TypeShape::Pending(PendingKind::Class) | TypeShape::Class { .. } => {
                MType::Object(name.to_string())
            }
            TypeShape::Pending(PendingKind::Callback) => MType::BoxedTrait(name.to_string()),
            TypeShape::Custom { repr } => MType::Custom {
                name: name.to_string(),
                repr: Box::new(repr.clone()),
            },
        })
    }

    pub fn classify_type_spelling(&self, spelling: &str) -> Option<MType> {
        self.classify_named_type(spelling).or_else(|| {
            syn::parse_str::<Type>(spelling)
                .ok()
                .and_then(|rust_type| self.classify_custom_remote_type(&rust_type))
        })
    }

    pub fn classify_custom_remote_type(&self, rust_type: &Type) -> Option<MType> {
        let custom_type_name = CustomTypeLookupKey::lookup_keys_for_rust_type(rust_type)
            .into_iter()
            .find_map(|lookup_key| self.custom_type_names_by_remote_key.get(&lookup_key))?;
        self.classify_named_type(custom_type_name)
    }

    pub fn register_custom_remote_type(
        &mut self,
        custom_type_name: &str,
        rust_type: &Type,
    ) -> Result<(), String> {
        let normalized_lookup_key = CustomTypeLookupKey::Normalized(CustomTypeNormalizedSpelling(
            normalize_type(rust_type),
        ));

        if let Some(existing_custom_type_name) = self
            .custom_type_names_by_remote_key
            .get(&normalized_lookup_key)
            .filter(|existing_custom_type_name| {
                existing_custom_type_name.as_str() != custom_type_name
            })
        {
            return Err(format!(
                "custom_type!: remote type already registered by `{}`",
                existing_custom_type_name
            ));
        }

        self.custom_type_names_by_remote_key
            .insert(normalized_lookup_key, custom_type_name.to_string());
        self.custom_type_names_by_remote_key
            .entry(CustomTypeLookupKey::Shape(CustomTypeShapeKey(
                type_shape_key(rust_type),
            )))
            .or_insert_with(|| custom_type_name.to_string());
        Ok(())
    }

    pub fn register_callback(&mut self, name: String) {
        self.types.insert(
            name,
            TypeMeta {
                shape: TypeShape::Pending(PendingKind::Callback),
                doc: None,
            },
        );
    }

    pub fn has_callback(&self, name: &str) -> bool {
        self.types
            .get(name)
            .is_some_and(|m| matches!(m.shape, TypeShape::Pending(PendingKind::Callback)))
    }
}

#[derive(Debug, Clone, Default)]
struct AliasResolver {
    use_aliases: HashMap<String, Vec<String>>,
    type_aliases: HashMap<String, Vec<String>>,
}

impl AliasResolver {
    fn from_items(items: &[Item]) -> Self {
        let mut resolver = Self::default();

        items
            .iter()
            .filter_map(|item| match item {
                Item::Use(item_use) => Some(&item_use.tree),
                _ => None,
            })
            .for_each(|tree| resolver.collect_use_tree(Vec::new(), tree));

        items
            .iter()
            .filter_map(|item| match item {
                Item::Type(item_type) => Some(item_type),
                _ => None,
            })
            .filter_map(|item_type| {
                let target = match item_type.ty.as_ref() {
                    Type::Path(type_path) => Some(Self::segments_from_path(type_path)),
                    _ => None,
                }?;
                Some((item_type.ident.to_string(), target))
            })
            .for_each(|(alias, target)| {
                resolver.type_aliases.insert(alias, target);
            });

        resolver
    }

    fn with_global(mut self, global: &HashMap<String, Vec<String>>) -> Self {
        global
            .iter()
            .filter(|&(name, _target)| !self.type_aliases.contains_key(name))
            .map(|(name, target)| (name.clone(), target.clone()))
            .collect::<Vec<_>>()
            .into_iter()
            .for_each(|(name, target)| {
                self.type_aliases.insert(name, target);
            });
        self
    }

    fn resolve_type_spelling<'a>(&self, spelling: &'a str) -> Cow<'a, str> {
        let stripped = spelling.trim().trim_start_matches("::");
        let parts: Vec<String> = stripped
            .split("::")
            .filter(|p| !p.is_empty())
            .map(|p| p.to_string())
            .collect();

        let resolved = self.resolve_segments(parts);
        let resolved_spelling = resolved
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("::");

        if resolved_spelling != stripped {
            Cow::Owned(resolved_spelling)
        } else {
            Cow::Borrowed(stripped)
        }
    }

    fn resolve_segments(&self, segments: Vec<String>) -> Vec<String> {
        let expanded = self.expand_use_aliases(segments);

        expanded
            .last()
            .and_then(|last| self.type_aliases.get(last))
            .cloned()
            .unwrap_or(expanded)
    }

    fn expand_use_aliases(&self, segments: Vec<String>) -> Vec<String> {
        self.expand_use_aliases_with_visited(segments, &mut HashSet::new())
    }

    fn expand_use_aliases_with_visited(
        &self,
        segments: Vec<String>,
        visited: &mut HashSet<Vec<String>>,
    ) -> Vec<String> {
        if !visited.insert(segments.clone()) {
            return segments;
        }

        match self.next_use_alias_segments(&segments) {
            Some(next_segments) if visited.contains(&next_segments) => segments,
            Some(next_segments) => self.expand_use_aliases_with_visited(next_segments, visited),
            None => segments,
        }
    }

    fn next_use_alias_segments(&self, segments: &[String]) -> Option<Vec<String>> {
        let first = segments.first()?;
        let replacement = self.use_aliases.get(first)?;
        let next_segments = replacement
            .iter()
            .cloned()
            .chain(segments.iter().skip(1).cloned())
            .collect::<Vec<_>>();
        (next_segments != segments).then_some(next_segments)
    }

    fn segments_from_path(type_path: &syn::TypePath) -> Vec<String> {
        type_path
            .path
            .segments
            .iter()
            .map(|seg| seg.ident.to_string())
            .collect()
    }

    fn collect_use_tree(&mut self, prefix: Vec<String>, tree: &syn::UseTree) {
        match tree {
            syn::UseTree::Path(path) => {
                let mut next_prefix = prefix;
                next_prefix.push(path.ident.to_string());
                self.collect_use_tree(next_prefix, &path.tree);
            }
            syn::UseTree::Name(name) => {
                let mut target = prefix;
                target.push(name.ident.to_string());
                self.use_aliases.insert(name.ident.to_string(), target);
            }
            syn::UseTree::Rename(rename) => {
                let mut target = prefix;
                target.push(rename.ident.to_string());
                self.use_aliases.insert(rename.rename.to_string(), target);
            }
            syn::UseTree::Group(group) => group
                .items
                .iter()
                .for_each(|item| self.collect_use_tree(prefix.clone(), item)),
            syn::UseTree::Glob(_) => {}
        }
    }
}

pub struct SourceScanner {
    module_name: String,
    type_registry: TypeRegistry,
    functions: Vec<Function>,
    callback_traits: Vec<CallbackTrait>,
    alias_resolver: AliasResolver,
    global_aliases: HashMap<String, Vec<String>>,
    compiler_canonical_types: HashMap<String, String>,
    integer_constants: HashMap<String, i128>,
    source_root: Option<PathBuf>,
    target_pointer_width_bits: Option<u8>,
    scan_errors: RefCell<Vec<ScanDiagnostic>>,
    generic_templates: GenericTemplates,
    pending_generic_instantiations: PendingGenericInstantiations,
}

/// A type the scanner could not resolve, together with where it appeared.
///
/// Emitted by `SourceScanner` any time a method, function, field, or variant
/// would have been silently dropped because `rust_type_to_ffi_type` returned
/// `None`. Collected so the scanner can fail hard rather than emit phantom
/// bindings that look valid but omit the offending symbol.
#[derive(Debug, Clone)]
pub struct ScanDiagnostic {
    pub context: String,
    pub rust_type: String,
}

impl std::fmt::Display for ScanDiagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unresolved type `{}` at {} (not a primitive, String, Vec<u8>, user `#[boltffi::data]` type, or supported container; generic user types like `Request<T>` are not yet supported)",
            self.rust_type, self.context
        )
    }
}

fn rust_type_spelling(ty: &syn::Type) -> String {
    let s = quote::quote!(#ty).to_string();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn struct_has_type_params(item_struct: &ItemStruct) -> bool {
    item_struct.generics.type_params().next().is_some()
}

fn mangle_generic_instance(base: &str, args: &[syn::Type]) -> String {
    let mut out = String::from(base);
    for arg in args {
        out.push_str(&mangle_type_for_name(arg));
    }
    out
}

fn mangle_type_for_name(ty: &syn::Type) -> String {
    let spelling = quote::quote!(#ty).to_string();
    spelling
        .chars()
        .filter(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Replace references to generic type parameters inside `ty` with the
/// concrete types from `subs`.
///
/// Only walks the AST shapes the scanner itself handles
/// (`Type::Path`, `Type::Reference`, `Type::Slice`, `Type::Tuple`).
/// Other shapes are returned unchanged; `rust_type_to_ffi_type` will then
/// fail to resolve them and record a diagnostic, which is the right
/// behavior — users asking for unsupported generic substitutions need a
/// clear error, not silent breakage.
fn substitute_type_params(ty: &syn::Type, subs: &HashMap<String, syn::Type>) -> syn::Type {
    match ty {
        syn::Type::Path(type_path) => {
            if type_path.qself.is_none()
                && type_path.path.leading_colon.is_none()
                && type_path.path.segments.len() == 1
                && type_path.path.segments[0].arguments.is_empty()
                && let Some(replacement) =
                    subs.get(&type_path.path.segments[0].ident.to_string())
            {
                return replacement.clone();
            }
            let mut new_path = type_path.clone();
            for seg in new_path.path.segments.iter_mut() {
                if let syn::PathArguments::AngleBracketed(args) = &mut seg.arguments {
                    for arg in args.args.iter_mut() {
                        if let syn::GenericArgument::Type(inner) = arg {
                            *inner = substitute_type_params(inner, subs);
                        }
                    }
                }
            }
            syn::Type::Path(new_path)
        }
        syn::Type::Reference(type_ref) => {
            let mut new_ref = type_ref.clone();
            *new_ref.elem = substitute_type_params(&type_ref.elem, subs);
            syn::Type::Reference(new_ref)
        }
        syn::Type::Slice(slice) => {
            let mut new_slice = slice.clone();
            *new_slice.elem = substitute_type_params(&slice.elem, subs);
            syn::Type::Slice(new_slice)
        }
        syn::Type::Tuple(tuple) => {
            let mut new_tuple = tuple.clone();
            new_tuple.elems = tuple
                .elems
                .iter()
                .map(|t| substitute_type_params(t, subs))
                .collect();
            syn::Type::Tuple(new_tuple)
        }
        other => other.clone(),
    }
}

impl SourceScanner {
    pub fn new(module_name: impl Into<String>) -> Self {
        Self::new_with_pointer_width(module_name, parse_target_pointer_width())
    }

    pub fn new_with_pointer_width(
        module_name: impl Into<String>,
        target_pointer_width_bits: Option<u8>,
    ) -> Self {
        Self {
            module_name: module_name.into(),
            type_registry: TypeRegistry::default(),
            functions: Vec::new(),
            callback_traits: Vec::new(),
            alias_resolver: AliasResolver::default(),
            global_aliases: HashMap::new(),
            compiler_canonical_types: HashMap::new(),
            integer_constants: HashMap::new(),
            source_root: None,
            target_pointer_width_bits,
            scan_errors: RefCell::new(Vec::new()),
            generic_templates: HashMap::new(),
            pending_generic_instantiations: RefCell::new(IndexMap::new()),
        }
    }

    fn record_unresolved(&self, context: impl Into<String>, ty: &syn::Type) {
        self.scan_errors.borrow_mut().push(ScanDiagnostic {
            context: context.into(),
            rust_type: rust_type_spelling(ty),
        });
    }

    pub fn take_scan_errors(&self) -> Vec<ScanDiagnostic> {
        std::mem::take(&mut self.scan_errors.borrow_mut())
    }

    fn resolve_ctx<'a>(
        &'a self,
        alias_resolver: &'a AliasResolver,
        self_type_name: Option<&'a str>,
    ) -> ResolveCtx<'a> {
        ResolveCtx {
            registry: &self.type_registry,
            alias_resolver,
            compiler_canonical_types: &self.compiler_canonical_types,
            self_type_name,
            generic_templates: &self.generic_templates,
            pending_instantiations: &self.pending_generic_instantiations,
        }
    }

    /// Materialize every pending `Request<MyEffect>`-style use-site the
    /// scanner observed. Iterates to a fixed point because expanding one
    /// template's fields may itself reference another generic template.
    pub fn finalize_generic_instantiations(&mut self) {
        const MAX_ITERATIONS: usize = 32;
        for _ in 0..MAX_ITERATIONS {
            let pending: Vec<(String, PendingInstantiation)> = self
                .pending_generic_instantiations
                .borrow_mut()
                .iter()
                .filter(|(mangled, _)| !self.type_registry.contains(mangled))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();

            if pending.is_empty() {
                break;
            }

            for (mangled, pending) in pending {
                self.materialize_generic_instance(&mangled, &pending);
            }
        }
    }

    fn materialize_generic_instance(&mut self, mangled: &str, pending: &PendingInstantiation) {
        let Some(template) = self.generic_templates.get(&pending.base).cloned() else {
            // Base template disappeared — should not happen, but a clean
            // no-op lets the surrounding diagnostic machinery point at
            // the original use-site error instead.
            return;
        };
        if template.type_params.len() != pending.args.len() {
            self.scan_errors.borrow_mut().push(ScanDiagnostic {
                context: format!(
                    "generic instantiation `{}` (template `{}` expects {} type parameter(s), got {})",
                    mangled,
                    pending.base,
                    template.type_params.len(),
                    pending.args.len()
                ),
                rust_type: format!("{}<...>", pending.base),
            });
            return;
        }

        let subs: HashMap<String, syn::Type> = template
            .type_params
            .iter()
            .cloned()
            .zip(pending.args.iter().cloned())
            .collect();

        // Insert a Pending record entry first so that, if one field's
        // type recursively references the same instantiation (self-referential
        // generics), the lookup already sees an entry and does not loop.
        self.type_registry.register(
            mangled.to_string(),
            TypeMeta {
                doc: template.doc.clone(),
                shape: TypeShape::Pending(PendingKind::Record),
            },
        );

        let resolved_fields: Vec<RecordField> = template
            .fields
            .iter()
            .filter_map(|field| {
                let substituted = substitute_type_params(&field.ty, &subs);
                let field_type = match rust_type_to_ffi_type(
                    &substituted,
                    &self.resolve_ctx(&self.alias_resolver, None),
                ) {
                    Some(t) => t,
                    None => {
                        self.record_unresolved(
                            format!(
                                "generic instantiation `{}` field `{}`",
                                mangled, field.name
                            ),
                            &substituted,
                        );
                        return None;
                    }
                };
                let mut record_field = RecordField::new(&field.name, field_type);
                if let Some(doc) = &field.doc {
                    record_field = record_field.with_doc(doc.clone());
                }
                if let Some(default) = &field.default {
                    record_field = record_field.with_default(default.clone());
                }
                Some(record_field)
            })
            .collect();

        self.type_registry.fill_record_fields(
            mangled,
            resolved_fields,
            template.is_repr_c,
            template.is_error,
        );
    }

    fn register_generic_template(&mut self, item_struct: &ItemStruct) {
        let name = item_struct.ident.to_string();
        let type_params: Vec<String> = item_struct
            .generics
            .type_params()
            .map(|tp| tp.ident.to_string())
            .collect();
        if type_params.is_empty() {
            return;
        }

        let fields: Vec<GenericTemplateField> = match &item_struct.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .filter_map(|f| {
                    let field_name = f.ident.as_ref()?.to_string();
                    Some(GenericTemplateField {
                        name: field_name,
                        ty: f.ty.clone(),
                        doc: extract_doc_string(&f.attrs),
                        default: extract_default_value(&f.attrs),
                    })
                })
                .collect(),
            Fields::Unnamed(unnamed) => unnamed
                .unnamed
                .iter()
                .enumerate()
                .map(|(i, f)| GenericTemplateField {
                    name: format!("value_{i}"),
                    ty: f.ty.clone(),
                    doc: extract_doc_string(&f.attrs),
                    default: None,
                })
                .collect(),
            Fields::Unit => Vec::new(),
        };

        let has_data = has_attribute(&item_struct.attrs, "data");
        let is_error = has_attribute(&item_struct.attrs, "error");
        let has_any_repr = item_struct.attrs.iter().any(|a| a.path().is_ident("repr"));
        let is_repr_c = if (has_data || is_error) && !has_any_repr {
            true
        } else {
            has_repr_c(&item_struct.attrs)
        };

        self.generic_templates.insert(
            name,
            GenericStructTemplate {
                type_params,
                fields,
                is_repr_c,
                is_error,
                doc: extract_doc_string(&item_struct.attrs),
            },
        );
    }

    pub fn scan_directory(&mut self, crate_path: &Path, dir: &Path) -> Result<(), String> {
        let mut files: Vec<_> = WalkDir::new(dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
            .map(|e| e.path().to_path_buf())
            .collect();
        files.sort();

        self.source_root = Some(dir.to_path_buf());
        self.global_aliases = Self::collect_global_aliases(&files)?;
        self.integer_constants =
            collect_integer_constants_from_files(dir, &files, self.target_pointer_width_bits)?;
        let compiler_targets = Self::collect_compiler_type_targets(&files, &self.global_aliases)?;
        self.compiler_canonical_types =
            compiler_type_resolution::resolve(crate_path, &self.module_name, compiler_targets)?;
        files
            .iter()
            .try_for_each(|path| self.collect_type_names(path))?;
        files
            .iter()
            .try_for_each(|path| self.collect_custom_types(path))?;
        files.iter().try_for_each(|path| self.scan_file(path))?;
        Ok(())
    }

    fn collect_compiler_type_targets(
        files: &[std::path::PathBuf],
        global_aliases: &HashMap<String, Vec<String>>,
    ) -> Result<Vec<String>, String> {
        let mut targets = Vec::<String>::new();
        let mut seen = HashSet::<String>::new();

        files.iter().try_for_each(|path| {
            let content = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let syntax = syn::parse_file(&content)
                .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

            let alias_resolver =
                AliasResolver::from_items(&syntax.items).with_global(global_aliases);
            syntax.items.iter().for_each(|item| {
                Self::collect_item_type_targets(item, &alias_resolver, &mut targets, &mut seen)
            });

            Ok::<(), String>(())
        })?;

        Ok(targets)
    }

    fn collect_item_type_targets(
        item: &Item,
        alias_resolver: &AliasResolver,
        out: &mut Vec<String>,
        seen: &mut HashSet<String>,
    ) {
        match item {
            Item::Struct(item_struct) => {
                let is_record = has_attribute(&item_struct.attrs, "ffi_record")
                    || has_attribute(&item_struct.attrs, "data")
                    || has_attribute(&item_struct.attrs, "error")
                    || has_repr_c(&item_struct.attrs)
                    || (has_attribute(&item_struct.attrs, "derive")
                        && has_ffi_type_derive(&item_struct.attrs));
                if is_record {
                    item_struct.fields.iter().for_each(|field| {
                        Self::collect_type_targets(&field.ty, alias_resolver, out, seen)
                    });
                }
            }
            Item::Enum(item_enum) => {
                let is_error = has_attribute(&item_enum.attrs, "error");
                let is_data_enum = has_repr_int(&item_enum.attrs)
                    || has_attribute(&item_enum.attrs, "data")
                    || is_error;
                if is_data_enum {
                    item_enum
                        .variants
                        .iter()
                        .flat_map(|variant| variant.fields.iter())
                        .for_each(|field| {
                            Self::collect_type_targets(&field.ty, alias_resolver, out, seen)
                        });
                }
            }
            Item::Impl(item_impl) => {
                let is_exported = has_attribute(&item_impl.attrs, "ffi_class")
                    || has_attribute(&item_impl.attrs, "export")
                    || has_data_impl_attribute(&item_impl.attrs);
                if is_exported {
                    item_impl
                        .items
                        .iter()
                        .filter_map(|impl_item| match impl_item {
                            ImplItem::Fn(method) => Some(method),
                            _ => None,
                        })
                        .filter(|method| matches!(method.vis, syn::Visibility::Public(_)))
                        .filter(|method| !method.attrs.iter().any(|a| a.path().is_ident("skip")))
                        .for_each(|method| {
                            method
                                .sig
                                .inputs
                                .iter()
                                .filter_map(|arg| match arg {
                                    FnArg::Typed(pat_type) => Some(pat_type.ty.as_ref()),
                                    FnArg::Receiver(_) => None,
                                })
                                .for_each(|ty| {
                                    Self::collect_type_targets(ty, alias_resolver, out, seen)
                                });

                            match &method.sig.output {
                                syn::ReturnType::Default => {}
                                syn::ReturnType::Type(_, ty) => {
                                    Self::collect_type_targets(
                                        ty.as_ref(),
                                        alias_resolver,
                                        out,
                                        seen,
                                    );
                                }
                            }
                        });
                }
            }
            Item::Trait(item_trait) => {
                let is_exported = has_attribute(&item_trait.attrs, "ffi_trait")
                    || has_attribute(&item_trait.attrs, "export");
                if is_exported {
                    item_trait
                        .items
                        .iter()
                        .filter_map(|trait_item| match trait_item {
                            syn::TraitItem::Fn(method) => Some(method),
                            _ => None,
                        })
                        .for_each(|method| {
                            method
                                .sig
                                .inputs
                                .iter()
                                .filter_map(|arg| match arg {
                                    FnArg::Typed(pat_type) => Some(pat_type.ty.as_ref()),
                                    FnArg::Receiver(_) => None,
                                })
                                .for_each(|ty| {
                                    Self::collect_type_targets(ty, alias_resolver, out, seen)
                                });

                            match &method.sig.output {
                                syn::ReturnType::Default => {}
                                syn::ReturnType::Type(_, ty) => {
                                    Self::collect_type_targets(
                                        ty.as_ref(),
                                        alias_resolver,
                                        out,
                                        seen,
                                    );
                                }
                            }
                        });
                }
            }
            Item::Fn(item_fn) => {
                let is_exported = has_attribute(&item_fn.attrs, "ffi_export")
                    || has_attribute(&item_fn.attrs, "export");
                if is_exported {
                    item_fn
                        .sig
                        .inputs
                        .iter()
                        .filter_map(|arg| match arg {
                            FnArg::Typed(pat_type) => Some(pat_type.ty.as_ref()),
                            FnArg::Receiver(_) => None,
                        })
                        .for_each(|ty| Self::collect_type_targets(ty, alias_resolver, out, seen));

                    match &item_fn.sig.output {
                        syn::ReturnType::Default => {}
                        syn::ReturnType::Type(_, ty) => {
                            Self::collect_type_targets(ty.as_ref(), alias_resolver, out, seen);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn collect_type_targets(
        ty: &Type,
        alias_resolver: &AliasResolver,
        out: &mut Vec<String>,
        seen: &mut HashSet<String>,
    ) {
        match ty {
            Type::Path(type_path) => {
                let all_segments_plain = type_path
                    .path
                    .segments
                    .iter()
                    .all(|seg| matches!(seg.arguments, syn::PathArguments::None));

                type_path
                    .path
                    .segments
                    .iter()
                    .filter_map(|seg| match &seg.arguments {
                        syn::PathArguments::AngleBracketed(args) => Some(args),
                        _ => None,
                    })
                    .flat_map(|args| args.args.iter())
                    .filter_map(|arg| match arg {
                        syn::GenericArgument::Type(inner_ty) => Some(inner_ty),
                        _ => None,
                    })
                    .for_each(|inner_ty| {
                        Self::collect_type_targets(inner_ty, alias_resolver, out, seen)
                    });

                type_path
                    .path
                    .segments
                    .iter()
                    .filter_map(|seg| match &seg.arguments {
                        syn::PathArguments::Parenthesized(args) => Some(args),
                        _ => None,
                    })
                    .for_each(|args| {
                        args.inputs.iter().for_each(|inner_ty| {
                            Self::collect_type_targets(inner_ty, alias_resolver, out, seen)
                        });
                        match &args.output {
                            syn::ReturnType::Default => {}
                            syn::ReturnType::Type(_, out_ty) => {
                                Self::collect_type_targets(
                                    out_ty.as_ref(),
                                    alias_resolver,
                                    out,
                                    seen,
                                );
                            }
                        }
                    });

                if all_segments_plain {
                    let path_str = type_path
                        .path
                        .segments
                        .iter()
                        .map(|seg| seg.ident.to_string())
                        .collect::<Vec<_>>()
                        .join("::");

                    let resolved = alias_resolver.resolve_type_spelling(&path_str).into_owned();
                    if resolved.starts_with("crate::") && seen.insert(resolved.clone()) {
                        out.push(resolved);
                    }
                }
            }
            Type::Reference(type_ref) => {
                Self::collect_type_targets(type_ref.elem.as_ref(), alias_resolver, out, seen);
            }
            Type::Slice(slice) => {
                Self::collect_type_targets(slice.elem.as_ref(), alias_resolver, out, seen);
            }
            Type::Array(array) => {
                Self::collect_type_targets(array.elem.as_ref(), alias_resolver, out, seen);
            }
            Type::Tuple(tuple) => tuple.elems.iter().for_each(|inner_ty| {
                Self::collect_type_targets(inner_ty, alias_resolver, out, seen)
            }),
            Type::Group(group) => {
                Self::collect_type_targets(group.elem.as_ref(), alias_resolver, out, seen);
            }
            Type::Paren(paren) => {
                Self::collect_type_targets(paren.elem.as_ref(), alias_resolver, out, seen);
            }
            _ => {}
        }
    }

    fn collect_global_aliases(
        files: &[std::path::PathBuf],
    ) -> Result<HashMap<String, Vec<String>>, String> {
        let mut aliases = HashMap::<String, Vec<String>>::new();
        let mut conflicts = HashSet::<String>::new();

        files.iter().try_for_each(|path| {
            let content = fs::read_to_string(path)
                .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

            let syntax = syn::parse_file(&content)
                .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

            let local = AliasResolver::from_items(&syntax.items);
            syntax
                .items
                .iter()
                .filter_map(|item| match item {
                    Item::Type(item_type) => Some(item_type),
                    _ => None,
                })
                .filter_map(|item_type| {
                    let target_path = match item_type.ty.as_ref() {
                        Type::Path(type_path) => {
                            Some(AliasResolver::segments_from_path(type_path).join("::"))
                        }
                        _ => None,
                    }?;

                    let resolved = local.resolve_type_spelling(&target_path).into_owned();
                    let segments = resolved
                        .split("::")
                        .filter(|p| !p.is_empty())
                        .map(|p| p.to_string())
                        .collect::<Vec<_>>();
                    Some((item_type.ident.to_string(), segments))
                })
                .for_each(|(alias_name, target)| match aliases.get(&alias_name) {
                    None => {
                        aliases.insert(alias_name, target);
                    }
                    Some(existing) if *existing == target => {}
                    Some(_) => {
                        conflicts.insert(alias_name);
                    }
                });

            Ok::<(), String>(())
        })?;

        conflicts.iter().for_each(|name| {
            aliases.remove(name);
        });

        Ok(aliases)
    }

    fn collect_custom_types(&mut self, path: &Path) -> Result<(), String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let syntax = syn::parse_file(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

        let alias_resolver =
            AliasResolver::from_items(&syntax.items).with_global(&self.global_aliases);
        syntax
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Macro(item_macro)
                    if item_macro
                        .mac
                        .path
                        .segments
                        .last()
                        .is_some_and(|segment| segment.ident == "custom_type") =>
                {
                    Some(item_macro)
                }
                _ => None,
            })
            .try_for_each(|item_macro| {
                self.collect_custom_type_macro(item_macro, &alias_resolver)
            })?;
        syntax
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Impl(item_impl) if has_attribute(&item_impl.attrs, "custom_ffi") => {
                    Some(item_impl)
                }
                _ => None,
            })
            .try_for_each(|item_impl| self.collect_custom_type(item_impl, &alias_resolver))
    }

    fn collect_custom_type_macro(
        &mut self,
        item_macro: &syn::ItemMacro,
        alias_resolver: &AliasResolver,
    ) -> Result<(), String> {
        let spec: CustomTypeMacroSpec = syn::parse2(item_macro.mac.tokens.clone())
            .map_err(|e| format!("custom_type!: failed to parse: {e}"))?;

        let name = spec.name.to_string();
        if self.type_registry.contains(&name) {
            return Err(format!(
                "custom_type!: `{}` conflicts with an existing record/enum/class name",
                name
            ));
        }

        let repr_syn_type = &spec.repr;
        let repr = rust_type_to_ffi_type(
            repr_syn_type,
            &self.resolve_ctx(alias_resolver, None),
        )
        .ok_or_else(|| {
            format!(
                "custom_type!: `{}` has an unsupported repr type: {}",
                name,
                quote::quote!(#repr_syn_type)
            )
        })?;

        self.type_registry.register(
            name.clone(),
            TypeMeta {
                doc: extract_doc_string(&item_macro.attrs),
                shape: TypeShape::Custom { repr: repr.clone() },
            },
        );
        self.type_registry
            .register_custom_remote_type(&name, &spec.remote)?;
        Ok(())
    }

    fn collect_custom_type(
        &mut self,
        item_impl: &ItemImpl,
        alias_resolver: &AliasResolver,
    ) -> Result<(), String> {
        let Some(name) = impl_self_type_ident(item_impl) else {
            return Err("custom_ffi: unsupported self type".to_string());
        };

        if self.type_registry.contains(&name) {
            return Err(format!(
                "custom_ffi: `{}` conflicts with an existing record/enum/class name",
                name
            ));
        }

        let repr_syn_type = item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Type(assoc) if assoc.ident == "FfiRepr" => Some(&assoc.ty),
                _ => None,
            })
            .next()
            .ok_or_else(|| format!("custom_ffi: `{}` is missing `type FfiRepr = ...;`", name))?;

        let repr = rust_type_to_ffi_type(
            repr_syn_type,
            &self.resolve_ctx(alias_resolver, None),
        )
        .ok_or_else(|| {
            format!(
                "custom_ffi: `{}` has an unsupported FfiRepr type: {}",
                name,
                quote::quote!(#repr_syn_type)
            )
        })?;

        self.type_registry.register(
            name.clone(),
            TypeMeta {
                doc: extract_doc_string(&item_impl.attrs),
                shape: TypeShape::Custom { repr: repr.clone() },
            },
        );

        Ok(())
    }

    fn collect_type_names(&mut self, path: &Path) -> Result<(), String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let syntax = syn::parse_file(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

        for item in &syntax.items {
            match item {
                Item::Struct(item_struct)
                    if has_attribute(&item_struct.attrs, "ffi_record")
                        || has_attribute(&item_struct.attrs, "data")
                        || has_attribute(&item_struct.attrs, "error")
                        || has_repr_c(&item_struct.attrs)
                        || (has_attribute(&item_struct.attrs, "derive")
                            && has_ffi_type_derive(&item_struct.attrs)) =>
                {
                    if struct_has_type_params(item_struct) {
                        // Generic user struct — store as a monomorphization
                        // template and skip the normal Pending registration
                        // so downstream `process_record` does not try to
                        // resolve `T` as a concrete type.
                        self.register_generic_template(item_struct);
                        continue;
                    }
                    self.type_registry.register(
                        item_struct.ident.to_string(),
                        TypeMeta {
                            doc: extract_doc_string(&item_struct.attrs),
                            shape: TypeShape::Pending(PendingKind::Record),
                        },
                    );
                }
                Item::Enum(item_enum)
                    if has_repr_int(&item_enum.attrs)
                        || has_attribute(&item_enum.attrs, "data")
                        || has_attribute(&item_enum.attrs, "error") =>
                {
                    self.type_registry.register(
                        item_enum.ident.to_string(),
                        TypeMeta {
                            doc: extract_doc_string(&item_enum.attrs),
                            shape: TypeShape::Pending(PendingKind::Enum),
                        },
                    );
                }
                Item::Impl(item_impl) => {
                    if (has_attribute(&item_impl.attrs, "ffi_class")
                        || has_attribute(&item_impl.attrs, "export"))
                        && !has_data_impl_attribute(&item_impl.attrs)
                        && let Type::Path(type_path) = item_impl.self_ty.as_ref()
                        && let Some(seg) = type_path.path.segments.last()
                    {
                        self.type_registry.register(
                            seg.ident.to_string(),
                            TypeMeta {
                                doc: None,
                                shape: TypeShape::Pending(PendingKind::Class),
                            },
                        );
                    }
                }
                _ => {}
            }
        }

        Ok(())
    }

    pub fn scan_file(&mut self, path: &Path) -> Result<(), String> {
        let content = fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

        let syntax = syn::parse_file(&content)
            .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

        self.alias_resolver =
            AliasResolver::from_items(&syntax.items).with_global(&self.global_aliases);
        let file_module_path = self
            .source_root
            .as_ref()
            .map(|source_root| module_path_for_source_file(source_root, path))
            .transpose()?
            .unwrap_or_default();
        syntax
            .items
            .iter()
            .try_for_each(|item| self.process_item(item, &file_module_path))?;

        Ok(())
    }

    fn process_item(&mut self, item: &Item, file_module_path: &[String]) -> Result<(), String> {
        match item {
            Item::Struct(item_struct) => {
                if let Some(doc) = extract_doc_string(&item_struct.attrs) {
                    self.type_registry
                        .set_doc(&item_struct.ident.to_string(), doc);
                }
                if has_attribute(&item_struct.attrs, "ffi_record")
                    || has_attribute(&item_struct.attrs, "data")
                    || has_attribute(&item_struct.attrs, "error")
                    || has_repr_c(&item_struct.attrs)
                    || (has_attribute(&item_struct.attrs, "derive")
                        && has_ffi_type_derive(&item_struct.attrs))
                {
                    if struct_has_type_params(item_struct) {
                        // Generic templates were captured during
                        // `collect_type_names`; concrete monomorphizations
                        // are synthesized from use-sites later.
                    } else {
                        self.process_record(item_struct);
                    }
                }
            }
            Item::Impl(item_impl) => {
                if has_data_impl_attribute(&item_impl.attrs) {
                    self.process_value_type_impl(item_impl);
                } else if has_attribute(&item_impl.attrs, "ffi_class")
                    || has_attribute(&item_impl.attrs, "export")
                {
                    self.process_class(item_impl);
                }
            }
            Item::Trait(item_trait)
                if has_attribute(&item_trait.attrs, "ffi_trait")
                    || has_attribute(&item_trait.attrs, "export") =>
            {
                self.process_callback_trait(item_trait);
            }
            Item::Fn(item_fn)
                if has_attribute(&item_fn.attrs, "ffi_export")
                    || has_attribute(&item_fn.attrs, "export") =>
            {
                self.process_function(item_fn);
            }
            Item::Enum(item_enum) => {
                let is_error = has_attribute(&item_enum.attrs, "error");
                if has_repr_int(&item_enum.attrs)
                    || has_attribute(&item_enum.attrs, "data")
                    || is_error
                {
                    self.process_enum(item_enum, is_error, file_module_path)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn resolve_typed_params(
        &self,
        inputs: &syn::punctuated::Punctuated<FnArg, syn::token::Comma>,
        self_type: Option<&str>,
        context: &str,
    ) -> Option<Vec<(String, MType)>> {
        let typed: Vec<_> = inputs
            .iter()
            .filter_map(|arg| match arg {
                FnArg::Typed(pat_type) => Some(pat_type),
                _ => None,
            })
            .collect();

        let resolved: Vec<(String, MType)> = typed
            .iter()
            .filter_map(|pat_type| {
                let name = match &*pat_type.pat {
                    syn::Pat::Ident(ident) => ident.ident.to_string(),
                    _ => return None,
                };
                match rust_type_to_ffi_type(
                    &pat_type.ty,
                    &self.resolve_ctx(&self.alias_resolver, self_type),
                ) {
                    Some(ty) => Some((name, ty)),
                    None => {
                        self.record_unresolved(
                            format!("{context} parameter `{name}`"),
                            &pat_type.ty,
                        );
                        None
                    }
                }
            })
            .collect();

        (resolved.len() == typed.len()).then_some(resolved)
    }

    fn resolve_output(
        &self,
        output: &syn::ReturnType,
        self_type: Option<&str>,
        context: &str,
    ) -> Option<MType> {
        match output {
            syn::ReturnType::Default => None,
            syn::ReturnType::Type(_, ty) => {
                match rust_type_to_ffi_type(
                    ty,
                    &self.resolve_ctx(&self.alias_resolver, self_type),
                ) {
                    Some(mty) => Some(mty),
                    None => {
                        self.record_unresolved(format!("{context} return"), ty);
                        None
                    }
                }
            }
        }
    }

    fn extract_receiver(sig: &syn::Signature) -> Receiver {
        sig.inputs
            .first()
            .and_then(|arg| match arg {
                syn::FnArg::Receiver(r) => Some(if r.mutability.is_some() {
                    Receiver::RefMut
                } else {
                    Receiver::Ref
                }),
                _ => None,
            })
            .unwrap_or(Receiver::None)
    }

    fn process_record(&mut self, item_struct: &ItemStruct) {
        let name = item_struct.ident.to_string();
        let fields = match &item_struct.fields {
            Fields::Named(named) => named
                .named
                .iter()
                .filter_map(|f| {
                    let field_name = f.ident.as_ref()?.to_string();
                    let field_type = match rust_type_to_ffi_type(
                        &f.ty,
                        &self.resolve_ctx(&self.alias_resolver, None),
                    ) {
                        Some(t) => t,
                        None => {
                            self.record_unresolved(
                                format!("record `{}` field `{}`", name, field_name),
                                &f.ty,
                            );
                            return None;
                        }
                    };
                    let mut record_field = RecordField::new(&field_name, field_type);
                    if let Some(doc) = extract_doc_string(&f.attrs) {
                        record_field = record_field.with_doc(doc);
                    }
                    if let Some(default) = extract_default_value(&f.attrs) {
                        record_field = record_field.with_default(default);
                    }
                    Some(record_field)
                })
                .collect(),
            _ => Vec::new(),
        };

        let has_data = has_attribute(&item_struct.attrs, "data");
        let is_error = has_attribute(&item_struct.attrs, "error");
        let has_any_repr = item_struct.attrs.iter().any(|a| a.path().is_ident("repr"));
        let is_repr_c = if (has_data || is_error) && !has_any_repr {
            true
        } else {
            has_repr_c(&item_struct.attrs)
        };
        self.type_registry
            .fill_record_fields(&name, fields, is_repr_c, is_error);
    }

    fn process_enum(
        &mut self,
        item_enum: &ItemEnum,
        is_error: bool,
        file_module_path: &[String],
    ) -> Result<(), String> {
        let name = item_enum.ident.to_string();
        let repr_type = extract_repr_int(&item_enum.attrs);
        let mut next_discriminant: i128 = 0;

        let variants = item_enum
            .variants
            .iter()
            .map(|v| -> Result<Variant, String> {
                let variant_name = v.ident.to_string();
                let discriminant = match v.discriminant.as_ref() {
                    Some((_, expr)) => parse_discriminant_expr(
                        expr,
                        &self.integer_constants,
                        file_module_path,
                        self.target_pointer_width_bits,
                    )
                    .ok_or_else(|| {
                        format!(
                            "failed to evaluate discriminant for enum `{}` variant `{}`",
                            name, variant_name
                        )
                    })?,
                    None => next_discriminant,
                };
                next_discriminant = discriminant.checked_add(1).ok_or_else(|| {
                    format!(
                        "discriminant overflow for enum `{}` variant `{}`",
                        name, variant_name
                    )
                })?;

                let fields: Vec<RecordField> = match &v.fields {
                    Fields::Named(named) => named
                        .named
                        .iter()
                        .filter_map(|f| {
                            let field_name = f.ident.as_ref()?.to_string();
                            let field_type = match rust_type_to_ffi_type(
                                &f.ty,
                                &self.resolve_ctx(&self.alias_resolver, None),
                            ) {
                                Some(t) => t,
                                None => {
                                    self.record_unresolved(
                                        format!(
                                            "enum `{}` variant `{}` field `{}`",
                                            name, variant_name, field_name
                                        ),
                                        &f.ty,
                                    );
                                    return None;
                                }
                            };
                            let mut record_field = RecordField::new(&field_name, field_type);
                            if let Some(doc) = extract_doc_string(&f.attrs) {
                                record_field = record_field.with_doc(doc);
                            }
                            Some(record_field)
                        })
                        .collect(),
                    Fields::Unnamed(unnamed) => unnamed
                        .unnamed
                        .iter()
                        .enumerate()
                        .filter_map(|(i, f)| {
                            let field_type = match rust_type_to_ffi_type(
                                &f.ty,
                                &self.resolve_ctx(&self.alias_resolver, None),
                            ) {
                                Some(t) => t,
                                None => {
                                    self.record_unresolved(
                                        format!(
                                            "enum `{}` variant `{}` field `{}`",
                                            name, variant_name, i
                                        ),
                                        &f.ty,
                                    );
                                    return None;
                                }
                            };
                            Some(RecordField::new(format!("value_{i}"), field_type))
                        })
                        .collect(),
                    Fields::Unit => Vec::new(),
                };

                let variant = Variant::new(&variant_name)
                    .with_discriminant(discriminant)
                    .maybe_doc(extract_doc_string(&v.attrs));
                Ok(fields
                    .into_iter()
                    .fold(variant, |v, field| v.with_field(field)))
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.type_registry.fill(
            &name,
            TypeShape::Enum {
                variants,
                is_error,
                repr_type,
                constructors: Vec::new(),
                methods: Vec::new(),
            },
        );
        Ok(())
    }

    fn process_function(&mut self, item_fn: &syn::ItemFn) {
        let sig = &item_fn.sig;
        let ctx = format!("function `{}`", sig.ident);
        let Some(params) = self.resolve_typed_params(&sig.inputs, None, &ctx) else {
            return;
        };
        let output = self.resolve_output(&sig.output, None, &ctx);
        if matches!(sig.output, syn::ReturnType::Type(..)) && output.is_none() {
            return;
        }

        let function = params
            .into_iter()
            .fold(Function::new(sig.ident.to_string()), |f, (name, ty)| {
                f.with_param(Parameter::new(&name, ty))
            })
            .maybe_doc(extract_doc_string(&item_fn.attrs))
            .maybe_return(output.map(ReturnType::from_output))
            .maybe_async(sig.asyncness.is_some());

        self.functions.push(function);
    }

    fn process_callback_trait(&mut self, item_trait: &ItemTrait) {
        let name = item_trait.ident.to_string();

        let callback = item_trait
            .items
            .iter()
            .filter_map(|item| match item {
                syn::TraitItem::Fn(method) => self.build_trait_method(method, &name),
                _ => None,
            })
            .fold(CallbackTrait::new(&name), |ct, m| ct.with_method(m))
            .maybe_doc(extract_doc_string(&item_trait.attrs));

        self.type_registry.register_callback(name);
        self.callback_traits.push(callback);
    }

    fn build_trait_method(
        &self,
        method: &syn::TraitItemFn,
        trait_name: &str,
    ) -> Option<TraitMethod> {
        let sig = &method.sig;
        let ctx = format!("callback trait method `{}::{}`", trait_name, sig.ident);
        let params = self.resolve_typed_params(&sig.inputs, None, &ctx)?;
        let output = self.resolve_output(&sig.output, None, &ctx);

        Some(
            params
                .into_iter()
                .fold(TraitMethod::new(sig.ident.to_string()), |tm, (name, ty)| {
                    tm.with_param(TraitMethodParam::new(&name, ty))
                })
                .maybe_doc(extract_doc_string(&method.attrs))
                .maybe_return(output.map(ReturnType::from_output))
                .maybe_async(sig.asyncness.is_some()),
        )
    }

    fn process_class(&mut self, item_impl: &ItemImpl) {
        let Some(class_name) = impl_self_type_ident(item_impl) else {
            return;
        };

        let mut constructors = Vec::new();
        let mut methods = Vec::new();
        let mut streams = Vec::new();

        item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(method) => Some(method),
                _ => None,
            })
            .filter(|method| matches!(method.vis, syn::Visibility::Public(_)))
            .filter(|method| !has_attribute(&method.attrs, "skip"))
            .for_each(|method| {
                if has_attribute(&method.attrs, "ffi_stream") {
                    if let Some(stream) = self.build_stream(method) {
                        streams.push(stream);
                    }
                    return;
                }

                if self.is_constructor(method, &class_name) {
                    if let Some(ctor) = self.build_constructor(method, &class_name) {
                        constructors.push(ctor);
                    }
                    return;
                }

                if let Some(built_method) = self.build_method(method, &class_name) {
                    methods.push(built_method);
                }
            });

        self.type_registry.fill(
            &class_name,
            TypeShape::Class {
                constructors,
                methods,
                streams,
            },
        );
    }

    fn process_value_type_impl(&mut self, item_impl: &ItemImpl) {
        let Some(type_name) = impl_self_type_ident(item_impl) else {
            return;
        };

        let is_record = self.type_registry.is_record(&type_name);
        let is_enum = self.type_registry.is_enum(&type_name);

        if !is_record && !is_enum {
            return;
        }

        let mut constructors = Vec::new();
        let mut methods = Vec::new();

        item_impl
            .items
            .iter()
            .filter_map(|item| match item {
                ImplItem::Fn(method) => Some(method),
                _ => None,
            })
            .filter(|method| matches!(method.vis, syn::Visibility::Public(_)))
            .filter(|method| !has_attribute(&method.attrs, "skip"))
            .for_each(|method| {
                if self.is_constructor(method, &type_name) {
                    if let Some(ctor) = self.build_constructor(method, &type_name) {
                        constructors.push(ctor);
                    }
                    return;
                }

                if let Some(built_method) = self.build_method(method, &type_name) {
                    methods.push(built_method);
                }
            });

        if is_record {
            self.type_registry
                .merge_record_impl(&type_name, constructors, methods);
        } else {
            self.type_registry
                .merge_enum_impl(&type_name, constructors, methods);
        }
    }

    fn build_method(&self, method: &syn::ImplItemFn, self_type_name: &str) -> Option<Method> {
        let sig = &method.sig;
        let receiver = Self::extract_receiver(sig);
        let ctx = format!("method `{}::{}`", self_type_name, sig.ident);
        let params = self.resolve_typed_params(&sig.inputs, Some(self_type_name), &ctx)?;
        let output = self.resolve_output(&sig.output, Some(self_type_name), &ctx);

        Some(
            params
                .into_iter()
                .fold(
                    Method::new(sig.ident.to_string(), receiver),
                    |m, (name, ty)| m.with_param(Parameter::new(&name, ty)),
                )
                .maybe_doc(extract_doc_string(&method.attrs))
                .maybe_return(output.map(ReturnType::from_output))
                .maybe_async(sig.asyncness.is_some()),
        )
    }

    fn build_stream(&self, method: &syn::ImplItemFn) -> Option<StreamMethod> {
        let name = method.sig.ident.to_string();

        let (item_type, mode) = extract_stream_attr(
            &method.attrs,
            &self.type_registry,
            &self.alias_resolver,
            &self.compiler_canonical_types,
        )?;

        Some(
            StreamMethod::new(&name, item_type)
                .with_mode(mode)
                .maybe_doc(extract_doc_string(&method.attrs)),
        )
    }

    fn is_constructor(&self, method: &syn::ImplItemFn, class_name: &str) -> bool {
        let has_self_receiver = method
            .sig
            .inputs
            .iter()
            .any(|arg| matches!(arg, syn::FnArg::Receiver(_)));
        if has_self_receiver {
            return false;
        }

        match &method.sig.output {
            syn::ReturnType::Default => false,
            syn::ReturnType::Type(_, ty) => {
                return_type_is_self(ty.as_ref(), class_name)
                    || return_type_is_result_self(ty.as_ref(), class_name)
                    || return_type_is_option_self(ty.as_ref(), class_name)
            }
        }
    }

    fn build_constructor(
        &self,
        method: &syn::ImplItemFn,
        self_type_name: &str,
    ) -> Option<Constructor> {
        let sig = &method.sig;
        let is_fallible = match &sig.output {
            syn::ReturnType::Default => false,
            syn::ReturnType::Type(_, ty) => return_type_is_result_self(ty.as_ref(), self_type_name),
        };
        let is_optional = match &sig.output {
            syn::ReturnType::Default => false,
            syn::ReturnType::Type(_, ty) => return_type_is_option_self(ty.as_ref(), self_type_name),
        };
        let ctx = format!("constructor `{}::{}`", self_type_name, sig.ident);
        let params = self.resolve_typed_params(&sig.inputs, Some(self_type_name), &ctx)?;

        Some(
            params
                .into_iter()
                .fold(
                    Constructor::new()
                        .with_name(sig.ident.to_string())
                        .with_fallible(is_fallible)
                        .with_optional(is_optional),
                    |c, (name, ty)| c.with_param(ConstructorParam::new(&name, ty)),
                )
                .maybe_doc(extract_doc_string(&method.attrs)),
        )
    }

    pub fn into_module(self) -> Module {
        let mut module = Module::new(&self.module_name);

        for (name, entry) in self.type_registry.drain() {
            match entry.shape {
                TypeShape::Record {
                    fields,
                    is_repr_c,
                    is_error,
                    constructors,
                    methods,
                } => {
                    let record = fields
                        .into_iter()
                        .fold(Record::new(&name), |r, f| r.with_field(f))
                        .with_repr_c(is_repr_c)
                        .maybe_doc(entry.doc);
                    let record = constructors
                        .into_iter()
                        .fold(record, |r, ctor| r.with_constructor(ctor));
                    let record = methods.into_iter().fold(record, |r, m| r.with_method(m));
                    let record = if is_error { record.as_error() } else { record };
                    module = module.with_record(record);
                }
                TypeShape::Enum {
                    variants,
                    is_error,
                    repr_type,
                    constructors,
                    methods,
                } => {
                    let mut enumeration = variants
                        .into_iter()
                        .fold(Enumeration::new(&name), |e, v| e.with_variant(v))
                        .maybe_doc(entry.doc);
                    if is_error {
                        enumeration = enumeration.as_error();
                    }
                    enumeration.repr_type = repr_type;
                    let enumeration = constructors
                        .into_iter()
                        .fold(enumeration, |e, ctor| e.with_constructor(ctor));
                    let enumeration = methods
                        .into_iter()
                        .fold(enumeration, |e, m| e.with_method(m));
                    module = module.with_enum(enumeration);
                }
                TypeShape::Class {
                    constructors,
                    methods,
                    streams,
                } => {
                    let class = constructors
                        .into_iter()
                        .fold(Class::new(&name), |c, ctor| c.with_constructor(ctor));
                    let class = methods.into_iter().fold(class, |c, m| c.with_method(m));
                    let class = streams
                        .into_iter()
                        .fold(class, |c, s| c.with_stream(s))
                        .maybe_doc(entry.doc);
                    module = module.with_class(class);
                }
                TypeShape::Custom { repr } => {
                    module = module.with_custom_type(CustomType::new(name, repr));
                }
                TypeShape::Pending(_) => {}
            }
        }

        for function in self.functions {
            module = module.with_function(function);
        }
        for callback in self.callback_traits {
            module = module.with_callback_trait(callback);
        }

        module.collect_derived_types();
        module
    }
}

struct CustomTypeMacroSpec {
    name: syn::Ident,
    remote: syn::Type,
    repr: syn::Type,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CustomTypeNormalizedSpelling(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CustomTypeShapeKey(String);

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum CustomTypeLookupKey {
    Normalized(CustomTypeNormalizedSpelling),
    Shape(CustomTypeShapeKey),
}

impl CustomTypeLookupKey {
    fn lookup_keys_for_rust_type(rust_type: &Type) -> [Self; 2] {
        [
            Self::Normalized(CustomTypeNormalizedSpelling(normalize_type(rust_type))),
            Self::Shape(CustomTypeShapeKey(type_shape_key(rust_type))),
        ]
    }
}

impl syn::parse::Parse for CustomTypeMacroSpec {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let _visibility: syn::Visibility = input.parse()?;
        let name: syn::Ident = input.parse()?;
        input.parse::<syn::Token![,]>()?;

        let mut remote: Option<syn::Type> = None;
        let mut repr: Option<syn::Type> = None;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            input.parse::<syn::Token![=]>()?;
            match key.to_string().as_str() {
                "remote" => {
                    remote = Some(input.parse()?);
                }
                "repr" => {
                    repr = Some(input.parse()?);
                }
                "error" => {
                    let _: syn::Type = input.parse()?;
                }
                "into_ffi" | "try_from_ffi" => {
                    let _: syn::Expr = input.parse()?;
                }
                _ => {
                    let _: syn::Expr = input.parse()?;
                }
            }

            if input.peek(syn::Token![,]) {
                input.parse::<syn::Token![,]>()?;
            }
        }

        let remote = remote.ok_or_else(|| input.error("custom_type!: missing `remote = ...`"))?;
        let repr = repr.ok_or_else(|| input.error("custom_type!: missing `repr = ...`"))?;

        Ok(Self { name, remote, repr })
    }
}

fn normalize_type(rust_type: &Type) -> String {
    quote::quote!(#rust_type).to_string().replace(' ', "")
}

fn type_shape_key(rust_type: &Type) -> String {
    match rust_type {
        Type::Reference(reference) => type_shape_key(reference.elem.as_ref()),
        Type::Path(type_path) => type_path
            .path
            .segments
            .last()
            .map(shape_key_for_segment)
            .unwrap_or_else(|| normalize_type(rust_type)),
        _ => normalize_type(rust_type),
    }
}

fn shape_key_for_segment(path_segment: &syn::PathSegment) -> String {
    let segment_name = path_segment.ident.to_string();
    let generic_shape_keys = match &path_segment.arguments {
        syn::PathArguments::AngleBracketed(arguments) => arguments
            .args
            .iter()
            .filter_map(|argument| match argument {
                syn::GenericArgument::Type(rust_type) => Some(type_shape_key(rust_type)),
                _ => None,
            })
            .collect::<Vec<_>>(),
        _ => Vec::new(),
    };

    if generic_shape_keys.is_empty() {
        segment_name
    } else {
        format!("{}<{}>", segment_name, generic_shape_keys.join(","))
    }
}

fn has_attribute(attrs: &[Attribute], name: &str) -> bool {
    attrs.iter().any(|attr| {
        attr.path().is_ident(name)
            || attr
                .path()
                .segments
                .last()
                .is_some_and(|segment| segment.ident == name)
    })
}

fn has_data_impl_attribute(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        let is_data = attr.path().is_ident("data")
            || attr
                .path()
                .segments
                .last()
                .is_some_and(|segment| segment.ident == "data");
        if !is_data {
            return false;
        }
        matches!(&attr.meta, syn::Meta::List(list) if list.tokens.to_string().trim() == "impl")
    })
}

fn extract_default_value(attrs: &[Attribute]) -> Option<String> {
    attrs.iter().find_map(|attr| {
        let path = attr.path();
        let is_boltffi_default = path.segments.len() == 2
            && path.segments[0].ident == "boltffi"
            && path.segments[1].ident == "default";
        if !is_boltffi_default {
            return None;
        }
        let tokens = match &attr.meta {
            syn::Meta::List(list) => list.tokens.to_string(),
            _ => return None,
        };
        Some(tokens.trim().to_string())
    })
}

fn extract_doc_string(attrs: &[Attribute]) -> Option<String> {
    let lines: Vec<String> = attrs
        .iter()
        .filter(|attr| attr.path().is_ident("doc"))
        .filter_map(|attr| match &attr.meta {
            syn::Meta::NameValue(nv) => {
                if let syn::Expr::Lit(syn::ExprLit {
                    lit: syn::Lit::Str(s),
                    ..
                }) = &nv.value
                {
                    Some(s.value())
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    if lines.is_empty() {
        return None;
    }

    let joined = lines
        .iter()
        .map(|line| line.strip_prefix(' ').unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");

    let trimmed = joined.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn return_type_is_self(ty: &Type, class_name: &str) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    type_path
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Self" || segment.ident == class_name)
}

fn return_type_is_result_self(ty: &Type, class_name: &str) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(result_segment) = type_path
        .path
        .segments
        .last()
        .filter(|segment| segment.ident == "Result")
    else {
        return false;
    };
    let syn::PathArguments::AngleBracketed(args) = &result_segment.arguments else {
        return false;
    };
    let Some(syn::GenericArgument::Type(Type::Path(ok_ty))) = args.args.first() else {
        return false;
    };
    ok_ty
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Self" || segment.ident == class_name)
}

fn return_type_is_option_self(ty: &Type, class_name: &str) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };
    let Some(option_segment) = type_path
        .path
        .segments
        .last()
        .filter(|segment| segment.ident == "Option")
    else {
        return false;
    };
    let syn::PathArguments::AngleBracketed(args) = &option_segment.arguments else {
        return false;
    };
    let Some(syn::GenericArgument::Type(Type::Path(inner_ty))) = args.args.first() else {
        return false;
    };
    inner_ty
        .path
        .segments
        .last()
        .is_some_and(|segment| segment.ident == "Self" || segment.ident == class_name)
}

fn impl_self_type_ident(item_impl: &ItemImpl) -> Option<String> {
    match item_impl.self_ty.as_ref() {
        Type::Path(type_path) => type_path
            .path
            .segments
            .last()
            .map(|segment| segment.ident.to_string()),
        Type::Group(group) => match group.elem.as_ref() {
            Type::Path(type_path) => type_path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            _ => None,
        },
        _ => None,
    }
}

fn has_repr_c(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("repr") {
            return false;
        }
        let Ok(meta) = attr.meta.require_list() else {
            return false;
        };
        meta.tokens.to_string().contains('C')
    })
}

fn has_repr_int(attrs: &[Attribute]) -> bool {
    extract_repr_int(attrs).is_some()
}

fn extract_repr_int(attrs: &[Attribute]) -> Option<Primitive> {
    attrs.iter().find_map(|attr| {
        if !attr.path().is_ident("repr") {
            return None;
        }
        let idents = attr
            .parse_args_with(
                syn::punctuated::Punctuated::<syn::Ident, syn::Token![,]>::parse_terminated,
            )
            .ok()?;
        idents
            .iter()
            .find_map(|ident| ident.to_string().parse().ok())
    })
}

fn module_path_for_source_file(
    source_root: &Path,
    file_path: &Path,
) -> Result<Vec<String>, String> {
    let relative_path = file_path.strip_prefix(source_root).map_err(|error| {
        format!(
            "Failed to resolve module path for {}: {}",
            file_path.display(),
            error
        )
    })?;
    let relative_components = relative_path
        .components()
        .map(|component| component.as_os_str().to_string_lossy().to_string())
        .collect::<Vec<_>>();
    let (file_name, directory_components) = relative_components
        .split_last()
        .ok_or_else(|| format!("Failed to resolve module path for {}", file_path.display()))?;
    let mut module_path = directory_components.to_vec();
    let file_stem = Path::new(file_name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| format!("Failed to resolve module path for {}", file_path.display()))?;
    if !matches!(file_stem, "lib" | "main" | "mod") {
        module_path.push(file_stem.to_string());
    }
    Ok(module_path)
}

fn collect_integer_constant_candidates(
    items: &[Item],
    module_path: &[String],
    out: &mut Vec<(String, syn::Expr)>,
) {
    items.iter().for_each(|item| match item {
        Item::Const(item_const) => {
            let key = module_path
                .iter()
                .cloned()
                .chain(std::iter::once(item_const.ident.to_string()))
                .collect::<Vec<_>>()
                .join("::");
            out.push((key, (*item_const.expr).clone()));
        }
        Item::Mod(item_mod) => {
            if let Some((_, inner_items)) = &item_mod.content {
                let mut nested_path = module_path.to_vec();
                nested_path.push(item_mod.ident.to_string());
                collect_integer_constant_candidates(inner_items, &nested_path, out);
            }
        }
        _ => {}
    });
}

fn constant_parent_module_path(constant_key: &str) -> Vec<String> {
    let mut segments = constant_key
        .split("::")
        .map(|segment| segment.to_string())
        .collect::<Vec<_>>();
    segments.pop();
    segments
}

fn resolve_integer_constants(
    mut candidates: Vec<(String, syn::Expr)>,
    target_pointer_width_bits: Option<u8>,
) -> HashMap<String, i128> {
    let mut resolved = HashMap::<String, i128>::new();
    loop {
        let unresolved_before = candidates.len();
        candidates = candidates
            .into_iter()
            .filter_map(|(key, expr)| {
                if resolved.contains_key(&key) {
                    return None;
                }
                let current_module_path = constant_parent_module_path(&key);
                if let Some(value) = parse_discriminant_expr(
                    &expr,
                    &resolved,
                    current_module_path.as_slice(),
                    target_pointer_width_bits,
                ) {
                    resolved.insert(key, value);
                    None
                } else {
                    Some((key, expr))
                }
            })
            .collect();
        if candidates.is_empty() || candidates.len() == unresolved_before {
            break;
        }
    }
    resolved
}

fn collect_integer_constants(
    items: &[Item],
    module_path: &[String],
    target_pointer_width_bits: Option<u8>,
) -> HashMap<String, i128> {
    let mut candidates = Vec::<(String, syn::Expr)>::new();
    collect_integer_constant_candidates(items, module_path, &mut candidates);
    resolve_integer_constants(candidates, target_pointer_width_bits)
}

fn collect_integer_constants_from_files(
    source_root: &Path,
    files: &[PathBuf],
    target_pointer_width_bits: Option<u8>,
) -> Result<HashMap<String, i128>, String> {
    let mut candidates = Vec::<(String, syn::Expr)>::new();
    files.iter().try_for_each(|file_path| {
        let module_path = module_path_for_source_file(source_root, file_path)?;
        let content = fs::read_to_string(file_path)
            .map_err(|error| format!("Failed to read {}: {}", file_path.display(), error))?;
        let syntax = syn::parse_file(&content)
            .map_err(|error| format!("Failed to parse {}: {}", file_path.display(), error))?;
        collect_integer_constant_candidates(&syntax.items, module_path.as_slice(), &mut candidates);
        Ok::<(), String>(())
    })?;
    Ok(resolve_integer_constants(
        candidates,
        target_pointer_width_bits,
    ))
}

fn parse_discriminant_expr(
    expr: &syn::Expr,
    constants: &HashMap<String, i128>,
    current_module_path: &[String],
    target_pointer_width_bits: Option<u8>,
) -> Option<i128> {
    fn parse_integer_literal(literal: &syn::LitInt) -> Option<i128> {
        if let Ok(value) = literal.base10_parse::<i128>() {
            return Some(value);
        }
        literal
            .base10_parse::<u128>()
            .ok()
            .filter(|value| *value <= u64::MAX as u128)
            .map(|value| value as i128)
    }

    fn parse_negated_integer_literal(literal: &syn::LitInt) -> Option<i128> {
        let unsigned = literal.base10_parse::<u128>().ok()?;
        if unsigned == (i64::MAX as u128) + 1 {
            return Some(i64::MIN as i128);
        }
        i128::try_from(unsigned).ok()?.checked_neg()
    }

    fn cast_target_name(cast_type: &syn::Type) -> Option<String> {
        match cast_type {
            syn::Type::Path(type_path) if type_path.qself.is_none() => type_path
                .path
                .segments
                .last()
                .map(|segment| segment.ident.to_string()),
            syn::Type::Group(group) => cast_target_name(group.elem.as_ref()),
            syn::Type::Paren(paren) => cast_target_name(paren.elem.as_ref()),
            _ => None,
        }
    }

    fn apply_integer_cast(
        value: i128,
        cast_type: &syn::Type,
        target_pointer_width_bits: Option<u8>,
    ) -> Option<i128> {
        match cast_target_name(cast_type)?.as_str() {
            "i8" => Some((value as i8) as i128),
            "u8" => Some((value as u8) as i128),
            "i16" => Some((value as i16) as i128),
            "u16" => Some((value as u16) as i128),
            "i32" => Some((value as i32) as i128),
            "u32" => Some((value as u32) as i128),
            "i64" => Some((value as i64) as i128),
            "u64" => Some((value as u64) as i128),
            "isize" => match target_pointer_width_bits {
                Some(32) => Some((value as i32) as i128),
                Some(64) => Some((value as i64) as i128),
                _ => None,
            },
            "usize" => match target_pointer_width_bits {
                Some(32) => Some((value as u32) as i128),
                Some(64) => Some((value as u64) as i128),
                _ => None,
            },
            _ => None,
        }
    }

    fn path_key(path: &[String]) -> Option<String> {
        (!path.is_empty()).then(|| path.join("::"))
    }

    fn path_segments(path: &syn::Path) -> Vec<String> {
        path.segments
            .iter()
            .map(|segment| segment.ident.to_string())
            .collect::<Vec<_>>()
    }

    fn const_lookup_keys(path: &syn::Path, current_module_path: &[String]) -> Vec<String> {
        let segments = path_segments(path);
        if segments.is_empty() {
            return Vec::new();
        }
        if path.leading_colon.is_some() {
            return path_key(segments.as_slice()).into_iter().collect();
        }

        match segments.first().map(String::as_str) {
            Some("crate") => {
                let absolute_path = segments.into_iter().skip(1).collect::<Vec<_>>();
                path_key(absolute_path.as_slice()).into_iter().collect()
            }
            Some("self") => {
                let resolved_path = current_module_path
                    .iter()
                    .cloned()
                    .chain(segments.into_iter().skip(1))
                    .collect::<Vec<_>>();
                path_key(resolved_path.as_slice()).into_iter().collect()
            }
            Some("super") => {
                let super_depth = segments
                    .iter()
                    .take_while(|segment| segment.as_str() == "super")
                    .count();
                if super_depth > current_module_path.len() {
                    return Vec::new();
                }
                let mut resolved_path = current_module_path
                    .iter()
                    .take(current_module_path.len() - super_depth)
                    .cloned()
                    .collect::<Vec<_>>();
                resolved_path.extend(segments.iter().skip(super_depth).cloned());
                let mut lookup_keys = path_key(resolved_path.as_slice())
                    .into_iter()
                    .collect::<Vec<_>>();
                let fallback_path = segments.into_iter().skip(super_depth).collect::<Vec<_>>();
                if let Some(fallback_key) = path_key(fallback_path.as_slice())
                    && lookup_keys.first() != Some(&fallback_key)
                {
                    lookup_keys.push(fallback_key);
                }
                lookup_keys
            }
            _ => {
                let relative_path = current_module_path
                    .iter()
                    .cloned()
                    .chain(segments.iter().cloned())
                    .collect::<Vec<_>>();
                let mut lookup_keys = path_key(relative_path.as_slice())
                    .into_iter()
                    .collect::<Vec<_>>();
                if let Some(absolute_key) = path_key(segments.as_slice())
                    && lookup_keys.first() != Some(&absolute_key)
                {
                    lookup_keys.push(absolute_key);
                }
                lookup_keys
            }
        }
    }

    match expr {
        syn::Expr::Lit(lit) => {
            if let syn::Lit::Int(int_lit) = &lit.lit {
                parse_integer_literal(int_lit)
            } else {
                None
            }
        }
        syn::Expr::Unary(unary) => {
            if matches!(unary.op, syn::UnOp::Neg(_))
                && let syn::Expr::Lit(lit) = unary.expr.as_ref()
                && let syn::Lit::Int(int_lit) = &lit.lit
            {
                return parse_negated_integer_literal(int_lit);
            }
            parse_discriminant_expr(
                &unary.expr,
                constants,
                current_module_path,
                target_pointer_width_bits,
            )
            .and_then(|value| match unary.op {
                syn::UnOp::Neg(_) => value.checked_neg(),
                syn::UnOp::Not(_) => Some(!value),
                _ => None,
            })
        }
        syn::Expr::Binary(binary) => {
            let left = parse_discriminant_expr(
                &binary.left,
                constants,
                current_module_path,
                target_pointer_width_bits,
            )?;
            let right = parse_discriminant_expr(
                &binary.right,
                constants,
                current_module_path,
                target_pointer_width_bits,
            )?;
            match binary.op {
                syn::BinOp::Add(_) => left.checked_add(right),
                syn::BinOp::Sub(_) => left.checked_sub(right),
                syn::BinOp::Mul(_) => left.checked_mul(right),
                syn::BinOp::Div(_) => (right != 0).then(|| left / right),
                syn::BinOp::Rem(_) => (right != 0).then(|| left % right),
                syn::BinOp::BitXor(_) => Some(left ^ right),
                syn::BinOp::BitAnd(_) => Some(left & right),
                syn::BinOp::BitOr(_) => Some(left | right),
                syn::BinOp::Shl(_) => u32::try_from(right)
                    .ok()
                    .and_then(|shift| left.checked_shl(shift)),
                syn::BinOp::Shr(_) => u32::try_from(right)
                    .ok()
                    .and_then(|shift| left.checked_shr(shift)),
                _ => None,
            }
        }
        syn::Expr::Paren(paren) => parse_discriminant_expr(
            &paren.expr,
            constants,
            current_module_path,
            target_pointer_width_bits,
        ),
        syn::Expr::Group(group) => parse_discriminant_expr(
            &group.expr,
            constants,
            current_module_path,
            target_pointer_width_bits,
        ),
        syn::Expr::Path(path) => const_lookup_keys(&path.path, current_module_path)
            .into_iter()
            .find_map(|key| constants.get(&key).copied()),
        syn::Expr::Cast(cast) => parse_discriminant_expr(
            &cast.expr,
            constants,
            current_module_path,
            target_pointer_width_bits,
        )
        .and_then(|value| apply_integer_cast(value, cast.ty.as_ref(), target_pointer_width_bits)),
        _ => None,
    }
}

fn parse_target_pointer_width() -> Option<u8> {
    std::env::var("BOLTFFI_TARGET_POINTER_WIDTH")
        .ok()
        .or_else(|| std::env::var("CARGO_CFG_TARGET_POINTER_WIDTH").ok())
        .and_then(|value| value.parse::<u8>().ok())
        .filter(|width| matches!(width, 32 | 64))
}

fn has_ffi_type_derive(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("derive") {
            return false;
        }
        let Ok(meta) = attr.meta.require_list() else {
            return false;
        };
        meta.tokens.to_string().contains("FfiType")
    })
}

fn extract_stream_attr(
    attrs: &[Attribute],
    registry: &TypeRegistry,
    alias_resolver: &AliasResolver,
    compiler_canonical_types: &HashMap<String, String>,
) -> Option<(MType, StreamMode)> {
    for attr in attrs {
        if !attr.path().is_ident("ffi_stream") {
            continue;
        }

        let Ok(meta) = attr.meta.require_list() else {
            continue;
        };

        let tokens = meta.tokens.to_string();
        let item_type =
            extract_item_type(&tokens, registry, alias_resolver, compiler_canonical_types)?;
        let mode = extract_stream_mode(&tokens);

        return Some((item_type, mode));
    }
    None
}

fn extract_item_type(
    tokens: &str,
    registry: &TypeRegistry,
    alias_resolver: &AliasResolver,
    compiler_canonical_types: &HashMap<String, String>,
) -> Option<MType> {
    let item_start = tokens.find("item")? + 4;
    let rest = &tokens[item_start..];
    let eq_pos = rest.find('=')?;
    let after_eq = rest[eq_pos + 1..].trim();

    let type_end = after_eq
        .find(',')
        .unwrap_or(after_eq.find(')').unwrap_or(after_eq.len()));
    let type_str = after_eq[..type_end].trim();

    string_to_ffi_type(type_str, registry, alias_resolver, compiler_canonical_types)
}

fn extract_stream_mode(tokens: &str) -> StreamMode {
    if tokens.contains("mode") {
        if tokens.contains("\"batch\"") {
            StreamMode::Batch
        } else if tokens.contains("\"callback\"") {
            StreamMode::Callback
        } else {
            StreamMode::Async
        }
    } else {
        StreamMode::Async
    }
}

fn rust_type_to_ffi_type(ty: &Type, ctx: &ResolveCtx<'_>) -> Option<MType> {
    if let Some(custom_type) = ctx.registry.classify_custom_remote_type(ty) {
        return Some(custom_type);
    }

    match ty {
        Type::Path(type_path) => {
            let last_segment = type_path.path.segments.last()?;
            let ident = last_segment.ident.to_string();

            if ident == "Self" {
                return ctx.self_type_name.and_then(|name| {
                    ctx.registry
                        .classify_named_type(name)
                        .or_else(|| Some(MType::Object(name.to_string())))
                });
            }

            if ident == "Box"
                && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
                && let Some(syn::GenericArgument::Type(Type::TraitObject(trait_obj))) =
                    args.args.first()
                && let Some(syn::TypeParamBound::Trait(trait_bound)) = trait_obj.bounds.first()
                && let Some(seg) = trait_bound.path.segments.last()
            {
                return Some(MType::BoxedTrait(seg.ident.to_string()));
            }

            if ident == "Arc"
                && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
                && let Some(syn::GenericArgument::Type(Type::TraitObject(trait_obj))) =
                    args.args.first()
                && let Some(syn::TypeParamBound::Trait(trait_bound)) = trait_obj.bounds.first()
                && let Some(seg) = trait_bound.path.segments.last()
            {
                return Some(MType::BoxedTrait(seg.ident.to_string()));
            }

            if (ident == "Arc" || ident == "Box")
                && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
                && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first()
            {
                return rust_type_to_ffi_type(inner_ty, ctx);
            }

            if ident == "Vec" {
                if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
                    && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first()
                {
                    let inner = rust_type_to_ffi_type(inner_ty, ctx)?;
                    return Some(MType::Vec(Box::new(inner)));
                }
                return None;
            }

            if ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
                    && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first()
                {
                    let inner = rust_type_to_ffi_type(inner_ty, ctx)?;
                    return Some(MType::Option(Box::new(inner)));
                }
                return None;
            }

            if ident == "Result" {
                if let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments {
                    let mut args_iter = args.args.iter();
                    if let Some(syn::GenericArgument::Type(ok_ty)) = args_iter.next() {
                        let ok = rust_type_to_ffi_type(ok_ty, ctx)?;
                        let err = args_iter
                            .next()
                            .and_then(|arg| {
                                if let syn::GenericArgument::Type(err_ty) = arg {
                                    rust_type_to_ffi_type(err_ty, ctx)
                                } else {
                                    None
                                }
                            })
                            .unwrap_or(MType::String);
                        return Some(MType::Result {
                            ok: Box::new(ok),
                            err: Box::new(err),
                        });
                    }
                }
                return None;
            }

            // User-defined generic struct use-site:
            // `Request<MyEffect>` where `Request<T>` was recorded as a
            // generic template during `collect_type_names`. The concrete
            // record with substituted fields is materialized later in a
            // finalization pass; here we just note the pending instance
            // and return a `Record` reference to the mangled name.
            if ctx.generic_templates.contains_key(&ident)
                && let syn::PathArguments::AngleBracketed(args) = &last_segment.arguments
            {
                let concrete_args: Vec<syn::Type> = args
                    .args
                    .iter()
                    .filter_map(|arg| match arg {
                        syn::GenericArgument::Type(t) => Some(t.clone()),
                        _ => None,
                    })
                    .collect();
                if !concrete_args.is_empty() {
                    let mangled = mangle_generic_instance(&ident, &concrete_args);
                    ctx.pending_instantiations
                        .borrow_mut()
                        .entry(mangled.clone())
                        .or_insert(PendingInstantiation {
                            base: ident.clone(),
                            args: concrete_args,
                        });
                    return Some(MType::Record(mangled));
                }
            }

            let path_str = type_path
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect::<Vec<_>>()
                .join("::");

            string_to_ffi_type(
                &path_str,
                ctx.registry,
                ctx.alias_resolver,
                ctx.compiler_canonical_types,
            )
        }
        Type::Reference(type_ref) => {
            if let Type::Path(inner) = &*type_ref.elem {
                let ident = inner.path.segments.last()?.ident.to_string();
                if ident == "str" {
                    return Some(MType::String);
                }
            }
            if let Type::Slice(slice) = &*type_ref.elem {
                let inner = rust_type_to_ffi_type(&slice.elem, ctx)?;
                return if type_ref.mutability.is_some() {
                    Some(MType::MutSlice(Box::new(inner)))
                } else {
                    Some(MType::Slice(Box::new(inner)))
                };
            }
            rust_type_to_ffi_type(&type_ref.elem, ctx)
        }
        Type::Slice(slice) => {
            let inner = rust_type_to_ffi_type(&slice.elem, ctx)?;
            Some(MType::Slice(Box::new(inner)))
        }
        Type::ImplTrait(impl_trait) => {
            for bound in &impl_trait.bounds {
                if let syn::TypeParamBound::Trait(trait_bound) = bound {
                    let trait_name = trait_bound
                        .path
                        .segments
                        .last()
                        .map(|s| s.ident.to_string())?;

                    if (trait_name == "FnMut" || trait_name == "Fn" || trait_name == "FnOnce")
                        && let syn::PathArguments::Parenthesized(args) =
                            &trait_bound.path.segments.last()?.arguments
                    {
                        let params: Vec<MType> = args
                            .inputs
                            .iter()
                            .filter_map(|t| rust_type_to_ffi_type(t, ctx))
                            .collect();

                        let returns = match &args.output {
                            syn::ReturnType::Default => MType::Void,
                            syn::ReturnType::Type(_, ty) => {
                                rust_type_to_ffi_type(ty, ctx).unwrap_or(MType::Void)
                            }
                        };

                        return Some(MType::Closure(MClosureSignature::new(params, returns)));
                    }

                    if ctx.registry.has_callback(&trait_name) {
                        return Some(MType::BoxedTrait(trait_name));
                    }
                }
            }
            None
        }
        Type::TraitObject(trait_obj) => {
            if let Some(syn::TypeParamBound::Trait(trait_bound)) = trait_obj.bounds.first()
                && let Some(seg) = trait_bound.path.segments.last()
            {
                return Some(MType::BoxedTrait(seg.ident.to_string()));
            }
            None
        }
        _ => None,
    }
}

fn string_to_ffi_type(
    s: &str,
    registry: &TypeRegistry,
    alias_resolver: &AliasResolver,
    compiler_canonical_types: &HashMap<String, String>,
) -> Option<MType> {
    let trimmed = s.trim();
    match trimmed {
        "i8" => Some(MType::Primitive(Primitive::I8)),
        "i16" => Some(MType::Primitive(Primitive::I16)),
        "i32" => Some(MType::Primitive(Primitive::I32)),
        "i64" => Some(MType::Primitive(Primitive::I64)),
        "u8" => Some(MType::Primitive(Primitive::U8)),
        "u16" => Some(MType::Primitive(Primitive::U16)),
        "u32" => Some(MType::Primitive(Primitive::U32)),
        "u64" => Some(MType::Primitive(Primitive::U64)),
        "f32" => Some(MType::Primitive(Primitive::F32)),
        "f64" => Some(MType::Primitive(Primitive::F64)),
        "bool" => Some(MType::Primitive(Primitive::Bool)),
        "usize" => Some(MType::Primitive(Primitive::USize)),
        "isize" => Some(MType::Primitive(Primitive::ISize)),
        "String" | "str" | "std::string::String" | "alloc::string::String" => Some(MType::String),
        s if s.starts_with("Vec<") => {
            let inner = &s[4..s.len() - 1];
            Some(MType::Vec(Box::new(string_to_ffi_type(
                inner,
                registry,
                alias_resolver,
                compiler_canonical_types,
            )?)))
        }
        s if s.starts_with("Option<") => {
            let inner = &s[7..s.len() - 1];
            Some(MType::Option(Box::new(string_to_ffi_type(
                inner,
                registry,
                alias_resolver,
                compiler_canonical_types,
            )?)))
        }
        s if s.starts_with("Result<") => {
            let inner = &s[7..s.len() - 1];
            let parts: Vec<&str> = inner.splitn(2, ',').map(|p| p.trim()).collect();
            let ok = string_to_ffi_type(
                parts.first()?,
                registry,
                alias_resolver,
                compiler_canonical_types,
            )?;
            let err = parts
                .get(1)
                .and_then(|e| {
                    string_to_ffi_type(e, registry, alias_resolver, compiler_canonical_types)
                })
                .unwrap_or(MType::String);
            Some(MType::Result {
                ok: Box::new(ok),
                err: Box::new(err),
            })
        }
        s => {
            if let Some(ty) = registry.classify_type_spelling(s) {
                return Some(ty);
            }

            let resolved = alias_resolver.resolve_type_spelling(s);
            let canonical = compiler_canonical_types
                .get(resolved.as_ref())
                .map(String::as_str)
                .unwrap_or(resolved.as_ref());

            BuiltinId::from_rust_path(canonical)
                .map(MType::Builtin)
                .or_else(|| registry.classify_type_spelling(canonical))
                .or_else(|| {
                    canonical
                        .rsplit("::")
                        .next()
                        .and_then(|name| registry.classify_type_spelling(name))
                })
        }
    }
}

fn validate_no_symbol_collisions(module: &Module) -> Result<(), String> {
    let mut symbols: HashMap<String, String> = HashMap::new();

    let mut check = |symbol: String, origin: String| -> Result<(), String> {
        if let Some(existing) = symbols.get(&symbol) {
            return Err(format!(
                "FFI symbol collision: '{}' is produced by both {} and {}. Rename one to avoid the conflict.",
                symbol, existing, origin
            ));
        }
        symbols.insert(symbol, origin);
        Ok(())
    };

    for func in &module.functions {
        let symbol = naming::function_ffi_name(&func.name).to_string();
        check(symbol, format!("fn {}()", func.name))?;
    }

    for record in &module.records {
        for ctor in &record.constructors {
            let symbol = if ctor.name == "new" {
                naming::class_ffi_new(&record.name).to_string()
            } else {
                naming::method_ffi_name(&record.name, &ctor.name).to_string()
            };
            check(symbol, format!("{}::{}()", record.name, ctor.name))?;
        }
        for method in &record.methods {
            let symbol = naming::method_ffi_name(&record.name, &method.name).to_string();
            check(symbol, format!("{}::{}()", record.name, method.name))?;
        }
    }

    for enumeration in &module.enums {
        for ctor in &enumeration.constructors {
            let symbol = if ctor.name == "new" {
                naming::class_ffi_new(&enumeration.name).to_string()
            } else {
                naming::method_ffi_name(&enumeration.name, &ctor.name).to_string()
            };
            check(symbol, format!("{}::{}()", enumeration.name, ctor.name))?;
        }
        for method in &enumeration.methods {
            let symbol = naming::method_ffi_name(&enumeration.name, &method.name).to_string();
            check(symbol, format!("{}::{}()", enumeration.name, method.name))?;
        }
    }

    for class in &module.classes {
        for ctor in &class.constructors {
            let symbol = if ctor.name == "new" {
                naming::class_ffi_new(&class.name).to_string()
            } else {
                naming::method_ffi_name(&class.name, &ctor.name).to_string()
            };
            check(symbol, format!("{}::{}()", class.name, ctor.name))?;
        }
        for method in &class.methods {
            let symbol = naming::method_ffi_name(&class.name, &method.name).to_string();
            check(symbol, format!("{}::{}()", class.name, method.name))?;
        }
    }

    Ok(())
}

pub fn scan_crate(crate_path: &Path, module_name: &str) -> Result<Module, String> {
    scan_crate_with_pointer_width(crate_path, module_name, None)
}

pub fn scan_crate_with_pointer_width(
    crate_path: &Path,
    module_name: &str,
    target_pointer_width_bits: Option<u8>,
) -> Result<Module, String> {
    let src_path = crate_path.join("src");
    let mut scanner = SourceScanner::new_with_pointer_width(
        module_name,
        target_pointer_width_bits.or_else(parse_target_pointer_width),
    );
    scanner.scan_directory(crate_path, &src_path)?;
    scanner.finalize_generic_instantiations();
    let diagnostics = scanner.take_scan_errors();
    if !diagnostics.is_empty() {
        let report = diagnostics
            .iter()
            .map(|d| format!("  - {d}"))
            .collect::<Vec<_>>()
            .join("\n");
        return Err(format!(
            "boltffi: {} unresolved type(s) encountered during scan — these would have been silently dropped from the generated bindings:\n{}",
            diagnostics.len(),
            report
        ));
    }
    let module = scanner.into_module();
    validate_no_symbol_collisions(&module)?;
    Ok(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn extract_doc_from_single_line() {
        let source: syn::File = syn::parse_quote! {
            /// A point in 2D space.
            struct Point;
        };
        let attrs = match &source.items[0] {
            Item::Struct(s) => &s.attrs,
            _ => panic!("expected struct"),
        };
        assert_eq!(
            extract_doc_string(attrs),
            Some("A point in 2D space.".to_string())
        );
    }

    #[test]
    fn extract_doc_from_multiple_lines() {
        let source: syn::File = syn::parse_quote! {
            /// First line.
            /// Second line.
            /// Third line.
            struct Widget;
        };
        let attrs = match &source.items[0] {
            Item::Struct(s) => &s.attrs,
            _ => panic!("expected struct"),
        };
        assert_eq!(
            extract_doc_string(attrs),
            Some("First line.\nSecond line.\nThird line.".to_string())
        );
    }

    #[test]
    fn extract_doc_returns_none_for_undocumented() {
        let source: syn::File = syn::parse_quote! {
            struct Bare;
        };
        let attrs = match &source.items[0] {
            Item::Struct(s) => &s.attrs,
            _ => panic!("expected struct"),
        };
        assert_eq!(extract_doc_string(attrs), None);
    }

    #[test]
    fn extract_doc_trims_empty_lines() {
        let source: syn::File = syn::parse_quote! {
            ///
            /// Actual content.
            ///
            struct Padded;
        };
        let attrs = match &source.items[0] {
            Item::Struct(s) => &s.attrs,
            _ => panic!("expected struct"),
        };
        let doc = extract_doc_string(attrs).unwrap();
        assert_eq!(doc, "Actual content.");
    }

    #[test]
    fn alias_resolution_stops_at_cycle_without_falling_back() {
        let resolver = AliasResolver {
            use_aliases: HashMap::from([
                ("Foo".to_string(), vec!["Bar".to_string()]),
                ("Bar".to_string(), vec!["Foo".to_string()]),
            ]),
            type_aliases: HashMap::new(),
        };

        assert_eq!(
            resolver.resolve_segments(vec!["Foo".to_string()]),
            vec!["Bar".to_string()]
        );
    }

    fn pending(kind: PendingKind) -> TypeMeta {
        TypeMeta {
            doc: None,
            shape: TypeShape::Pending(kind),
        }
    }

    fn pending_with_doc(kind: PendingKind, doc: &str) -> TypeMeta {
        TypeMeta {
            doc: Some(doc.to_string()),
            shape: TypeShape::Pending(kind),
        }
    }

    #[test]
    fn type_registry_single_map_classify() {
        let mut reg = TypeRegistry::default();
        reg.register("Point".into(), pending(PendingKind::Record));
        reg.register("Color".into(), pending(PendingKind::Enum));
        reg.register("Sensor".into(), pending(PendingKind::Class));
        reg.register(
            "UtcDateTime".into(),
            TypeMeta {
                doc: None,
                shape: TypeShape::Custom {
                    repr: MType::Primitive(Primitive::I64),
                },
            },
        );

        assert!(matches!(
            reg.classify_named_type("Point"),
            Some(MType::Record(_))
        ));
        assert!(matches!(
            reg.classify_named_type("Color"),
            Some(MType::Enum(_))
        ));
        assert!(matches!(
            reg.classify_named_type("Sensor"),
            Some(MType::Object(_))
        ));
        assert!(matches!(
            reg.classify_named_type("UtcDateTime"),
            Some(MType::Custom { .. })
        ));
        assert!(reg.classify_named_type("Unknown").is_none());
    }

    #[test]
    fn type_registry_is_enum() {
        let mut reg = TypeRegistry::default();
        reg.register("Status".into(), pending(PendingKind::Enum));
        reg.register("Point".into(), pending(PendingKind::Record));

        assert!(reg.is_enum("Status"));
        assert!(!reg.is_enum("Point"));
        assert!(!reg.is_enum("Missing"));
    }

    #[test]
    fn type_registry_contains() {
        let mut reg = TypeRegistry::default();
        reg.register("Point".into(), pending(PendingKind::Record));

        assert!(reg.contains("Point"));
        assert!(!reg.contains("Nope"));
    }

    #[test]
    fn type_registry_doc_at_registration() {
        let mut reg = TypeRegistry::default();
        reg.register(
            "Sensor".into(),
            pending_with_doc(PendingKind::Class, "A hardware sensor."),
        );

        assert_eq!(reg.doc("Sensor"), Some("A hardware sensor."));
    }

    #[test]
    fn type_registry_set_doc_after_registration() {
        let mut reg = TypeRegistry::default();
        reg.register("Sensor".into(), pending(PendingKind::Class));
        reg.set_doc("Sensor", "A hardware sensor.".into());

        assert_eq!(reg.doc("Sensor"), Some("A hardware sensor."));
    }

    #[test]
    fn type_registry_set_doc_ignores_unregistered() {
        let mut reg = TypeRegistry::default();
        reg.set_doc("Ghost", "spooky".into());

        assert!(reg.doc("Ghost").is_none());
    }

    #[test]
    fn type_registry_custom_type_classifies_correctly() {
        let mut reg = TypeRegistry::default();
        reg.register(
            "Timestamp".into(),
            TypeMeta {
                doc: None,
                shape: TypeShape::Custom {
                    repr: MType::Primitive(Primitive::I64),
                },
            },
        );

        assert!(matches!(
            reg.classify_named_type("Timestamp"),
            Some(MType::Custom { .. })
        ));
        assert!(!reg.is_enum("Timestamp"));
    }

    #[test]
    fn type_registry_custom_type_classifies_remote_generic_type() {
        let mut reg = TypeRegistry::default();
        reg.register(
            "UtcDateTime".into(),
            TypeMeta {
                doc: None,
                shape: TypeShape::Custom {
                    repr: MType::Primitive(Primitive::I64),
                },
            },
        );
        reg.register_custom_remote_type("UtcDateTime", &syn::parse_quote!(DateTime<Utc>))
            .unwrap();

        assert!(matches!(
            reg.classify_type_spelling("DateTime<Utc>"),
            Some(MType::Custom { .. })
        ));
        assert!(matches!(
            reg.classify_type_spelling("chrono::DateTime<chrono::Utc>"),
            Some(MType::Custom { .. })
        ));
    }

    #[test]
    fn scan_demo_crate_includes_datetime_custom_type_exports() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let demo_crate_path = repo_root.join("examples").join("demo");
        let module = scan_crate(&demo_crate_path, "demo").unwrap();

        assert!(
            module
                .functions
                .iter()
                .any(|function| function.name == "echo_datetime"),
            "expected echo_datetime to be exported"
        );
        assert!(
            module
                .functions
                .iter()
                .any(|function| function.name == "datetime_to_millis"),
            "expected datetime_to_millis to be exported"
        );
    }

    #[test]
    fn scan_demo_crate_preserves_callback_return_type() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .to_path_buf();
        let demo_crate_path = repo_root.join("examples").join("demo");
        let module = scan_crate(&demo_crate_path, "demo").unwrap();

        let function = module
            .functions
            .iter()
            .find(|function| function.name == "make_incrementing_callback")
            .expect("expected make_incrementing_callback to be exported");

        assert!(matches!(
            function.returns,
            crate::model::ReturnType::Value(crate::model::Type::BoxedTrait(ref name))
                if name == "ValueCallback"
        ));
    }

    #[test]
    fn type_registry_fill_replaces_pending() {
        let mut reg = TypeRegistry::default();
        reg.register("Point".into(), pending(PendingKind::Record));
        reg.fill(
            "Point",
            TypeShape::Record {
                fields: vec![RecordField::new("x", MType::Primitive(Primitive::F64))],
                is_repr_c: true,
                is_error: false,
                constructors: Vec::new(),
                methods: Vec::new(),
            },
        );

        assert!(matches!(
            reg.classify_named_type("Point"),
            Some(MType::Record(_))
        ));
        assert!(matches!(
            reg.types.get("Point").unwrap().shape,
            TypeShape::Record { ref fields, .. } if fields.len() == 1
        ));
    }

    #[test]
    fn type_registry_filled_enum_still_is_enum() {
        let mut reg = TypeRegistry::default();
        reg.register("Color".into(), pending(PendingKind::Enum));
        reg.fill(
            "Color",
            TypeShape::Enum {
                variants: vec![],
                is_error: false,
                repr_type: None,
                constructors: vec![],
                methods: vec![],
            },
        );

        assert!(reg.is_enum("Color"));
    }

    #[test]
    fn merge_record_impl_upgrades_pending_to_filled_shape() {
        let mut reg = TypeRegistry::default();
        reg.register("Point".into(), pending(PendingKind::Record));

        let ctor = Constructor::new().with_name("origin");
        let method = Method::new("magnitude", Receiver::Ref);
        reg.merge_record_impl("Point", vec![ctor], vec![method]);

        match &reg.types.get("Point").unwrap().shape {
            TypeShape::Record {
                fields,
                constructors,
                methods,
                ..
            } => {
                assert!(fields.is_empty());
                assert_eq!(constructors.len(), 1);
                assert_eq!(constructors[0].name, "origin");
                assert_eq!(methods.len(), 1);
                assert_eq!(methods[0].name, "magnitude");
            }
            other => panic!("expected Record, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn fill_record_fields_preserves_existing_methods() {
        let mut reg = TypeRegistry::default();
        reg.register("Point".into(), pending(PendingKind::Record));

        let ctor = Constructor::new().with_name("origin");
        let method = Method::new("magnitude", Receiver::Ref);
        reg.merge_record_impl("Point", vec![ctor], vec![method]);

        reg.fill_record_fields(
            "Point",
            vec![RecordField::new("x", MType::Primitive(Primitive::F64))],
            true,
            false,
        );

        match &reg.types.get("Point").unwrap().shape {
            TypeShape::Record {
                fields,
                is_repr_c,
                constructors,
                methods,
                ..
            } => {
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].name, "x");
                assert!(*is_repr_c);
                assert_eq!(constructors.len(), 1);
                assert_eq!(methods.len(), 1);
            }
            other => panic!("expected Record, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn fill_record_fields_creates_fresh_shape_from_pending() {
        let mut reg = TypeRegistry::default();
        reg.register("Point".into(), pending(PendingKind::Record));

        reg.fill_record_fields(
            "Point",
            vec![RecordField::new("x", MType::Primitive(Primitive::F64))],
            false,
            false,
        );

        match &reg.types.get("Point").unwrap().shape {
            TypeShape::Record {
                fields,
                is_repr_c,
                constructors,
                methods,
                ..
            } => {
                assert_eq!(fields.len(), 1);
                assert!(!*is_repr_c);
                assert!(constructors.is_empty());
                assert!(methods.is_empty());
            }
            other => panic!("expected Record, got {:?}", std::mem::discriminant(other)),
        }
    }

    #[test]
    fn extract_doc_from_struct_field() {
        let source: syn::File = syn::parse_quote! {
            struct Location {
                /// Unique identifier for this location.
                id: i64,
                lat: f64,
            }
        };
        let fields = match &source.items[0] {
            Item::Struct(s) => match &s.fields {
                Fields::Named(named) => &named.named,
                _ => panic!("expected named fields"),
            },
            _ => panic!("expected struct"),
        };
        assert_eq!(
            extract_doc_string(&fields[0].attrs),
            Some("Unique identifier for this location.".to_string())
        );
        assert_eq!(extract_doc_string(&fields[1].attrs), None);
    }

    #[test]
    fn extract_doc_from_enum_variant() {
        let source: syn::File = syn::parse_quote! {
            enum Direction {
                /// Pointing north.
                North,
                South,
            }
        };
        let variants = match &source.items[0] {
            Item::Enum(e) => &e.variants,
            _ => panic!("expected enum"),
        };
        assert_eq!(
            extract_doc_string(&variants[0].attrs),
            Some("Pointing north.".to_string())
        );
        assert_eq!(extract_doc_string(&variants[1].attrs), None);
    }

    #[test]
    fn extract_doc_from_enum_variant_field() {
        let source: syn::File = syn::parse_quote! {
            enum Status {
                Failed {
                    /// Error code describing the failure.
                    error_code: i32,
                    retry_count: i32,
                },
            }
        };
        let fields = match &source.items[0] {
            Item::Enum(e) => match &e.variants[0].fields {
                Fields::Named(named) => &named.named,
                _ => panic!("expected named fields"),
            },
            _ => panic!("expected enum"),
        };
        assert_eq!(
            extract_doc_string(&fields[0].attrs),
            Some("Error code describing the failure.".to_string())
        );
        assert_eq!(extract_doc_string(&fields[1].attrs), None);
    }

    #[test]
    fn parse_discriminant_expr_handles_const_refs_and_bit_ops() {
        let source: syn::File = syn::parse_quote! {
            const FLAG: i64 = 1 << 4;
            const MASK: i64 = FLAG | 0b11;
        };
        let constants = collect_integer_constants(&source.items, &[], None);
        let expr: syn::Expr = syn::parse_quote!(MASK + 1);

        assert_eq!(
            parse_discriminant_expr(&expr, &constants, &[], None),
            Some(20)
        );
    }

    #[test]
    fn discriminant_progression_tracks_explicit_expr_values() {
        let source: syn::File = syn::parse_quote! {
            const START: i64 = 10;
            enum Mode {
                A = START,
                B,
                C = 1 << 3,
                D,
            }
        };
        let constants = collect_integer_constants(&source.items, &[], None);
        let item_enum = match &source.items[1] {
            Item::Enum(item_enum) => item_enum,
            _ => panic!("expected enum"),
        };

        let discriminants = item_enum
            .variants
            .iter()
            .scan(0_i128, |next_discriminant, variant| {
                let value = match variant.discriminant.as_ref() {
                    Some((_, expr)) => parse_discriminant_expr(expr, &constants, &[], None),
                    None => Some(*next_discriminant),
                }?;
                *next_discriminant = value + 1;
                Some(value)
            })
            .collect::<Vec<_>>();

        assert_eq!(discriminants, vec![10, 11, 8, 9]);
    }

    #[test]
    fn collect_integer_constants_resolves_forward_references() {
        let source: syn::File = syn::parse_quote! {
            const A: i64 = B;
            const B: i64 = 3;
        };
        let constants = collect_integer_constants(&source.items, &[], None);

        assert_eq!(constants.get("A").copied(), Some(3));
        assert_eq!(constants.get("B").copied(), Some(3));
    }

    #[test]
    fn parse_discriminant_expr_prefers_full_qualified_const_path() {
        let constants = HashMap::from([
            ("BAR".to_string(), 1_i128),
            ("foo::BAR".to_string(), 9_i128),
        ]);

        let qualified: syn::Expr = syn::parse_quote!(foo::BAR);
        let unqualified: syn::Expr = syn::parse_quote!(BAR);
        let crate_qualified: syn::Expr = syn::parse_quote!(crate::foo::BAR);

        assert_eq!(
            parse_discriminant_expr(&qualified, &constants, &[], None),
            Some(9)
        );
        assert_eq!(
            parse_discriminant_expr(&unqualified, &constants, &[], None),
            Some(1)
        );
        assert_eq!(
            parse_discriminant_expr(&crate_qualified, &constants, &[], None),
            Some(9)
        );
    }

    #[test]
    fn parse_discriminant_expr_resolves_super_path_from_current_module() {
        let constants = HashMap::from([
            ("foo::FLAG".to_string(), 9_i128),
            ("foo::bar::FLAG".to_string(), 1_i128),
        ]);
        let expression: syn::Expr = syn::parse_quote!(super::FLAG);
        let current_module_path = vec!["foo".to_string(), "bar".to_string()];

        assert_eq!(
            parse_discriminant_expr(
                &expression,
                &constants,
                current_module_path.as_slice(),
                None
            ),
            Some(9)
        );
    }

    #[test]
    fn parse_discriminant_expr_applies_integer_cast_semantics() {
        let expression: syn::Expr = syn::parse_quote!((-1i16 as u8));

        assert_eq!(
            parse_discriminant_expr(&expression, &HashMap::new(), &[], None),
            Some(255)
        );
    }

    #[test]
    fn parse_discriminant_expr_accepts_u64_max_literal() {
        let expression: syn::Expr = syn::parse_quote!(18446744073709551615);

        assert_eq!(
            parse_discriminant_expr(&expression, &HashMap::new(), &[], None),
            Some(u64::MAX as i128)
        );
    }

    #[test]
    fn parse_discriminant_expr_accepts_i64_min_literal_form() {
        let expression: syn::Expr = syn::parse_quote!(-9223372036854775808);

        assert_eq!(
            parse_discriminant_expr(&expression, &HashMap::new(), &[], None),
            Some(i64::MIN as i128)
        );
    }

    #[test]
    fn parse_discriminant_expr_truncates_i64_cast() {
        let expression: syn::Expr = syn::parse_quote!(18446744073709551615 as i64);

        assert_eq!(
            parse_discriminant_expr(&expression, &HashMap::new(), &[], None),
            Some(-1)
        );
    }

    #[test]
    fn parse_discriminant_expr_requires_pointer_width_for_usize_casts() {
        let expression: syn::Expr = syn::parse_quote!(7 as usize);

        assert_eq!(
            parse_discriminant_expr(&expression, &HashMap::new(), &[], None),
            None
        );
    }

    #[test]
    fn parse_discriminant_expr_applies_target_pointer_width_to_usize_casts() {
        let expression: syn::Expr = syn::parse_quote!(4294967297 as usize);

        assert_eq!(
            parse_discriminant_expr(&expression, &HashMap::new(), &[], Some(32)),
            Some(1)
        );
        assert_eq!(
            parse_discriminant_expr(&expression, &HashMap::new(), &[], Some(64)),
            Some(4294967297)
        );
    }

    #[test]
    fn collect_integer_constants_from_files_collects_module_scoped_constants() {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_root = std::env::temp_dir().join(format!(
            "boltffi_scan_constants_{}_{}",
            std::process::id(),
            unique_suffix
        ));
        let source_root = temp_root.join("src");

        fs::create_dir_all(&source_root).expect("create source root");
        fs::write(source_root.join("lib.rs"), "pub mod constants;").expect("write lib.rs");
        fs::write(source_root.join("constants.rs"), "pub const TAG: i64 = 7;")
            .expect("write constants.rs");

        let files = vec![source_root.join("lib.rs"), source_root.join("constants.rs")];
        let constants = collect_integer_constants_from_files(&source_root, files.as_slice(), None)
            .expect("collect constants");
        let expression: syn::Expr = syn::parse_quote!(crate::constants::TAG);

        assert_eq!(constants.get("constants::TAG").copied(), Some(7));
        assert_eq!(
            parse_discriminant_expr(&expression, &constants, &[], None),
            Some(7)
        );

        fs::remove_dir_all(temp_root).expect("cleanup temp root");
    }

    fn scan_temp_crate(source: &str) -> Module {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_root = std::env::temp_dir().join(format!(
            "boltffi_scan_record_methods_{}_{}",
            std::process::id(),
            unique_suffix
        ));
        let src_dir = temp_root.join("src");
        fs::create_dir_all(&src_dir).expect("create src dir");
        fs::write(src_dir.join("lib.rs"), source).expect("write lib.rs");

        let module =
            scan_crate_with_pointer_width(&temp_root, "testlib", None).expect("scan failed");
        fs::remove_dir_all(&temp_root).expect("cleanup");
        module
    }

    fn scan_temp_crate_multi(files: &[(&str, &str)]) -> Module {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_root = std::env::temp_dir().join(format!(
            "boltffi_scan_record_methods_{}_{}",
            std::process::id(),
            unique_suffix
        ));
        let src_dir = temp_root.join("src");
        fs::create_dir_all(&src_dir).expect("create src dir");

        files.iter().for_each(|(name, content)| {
            fs::write(src_dir.join(name), content).expect("write file");
        });

        let module =
            scan_crate_with_pointer_width(&temp_root, "testlib", None).expect("scan failed");
        fs::remove_dir_all(&temp_root).expect("cleanup");
        module
    }

    #[test]
    fn error_structs_are_scanned_as_records() {
        let source = r#"
            use boltffi::*;

            #[error]
            pub struct AppError {
                pub code: i32,
                pub message: String,
            }

            #[export]
            pub fn may_fail(valid: bool) -> Result<String, AppError> {
                if valid {
                    Ok("ok".to_string())
                } else {
                    Err(AppError {
                        code: 400,
                        message: "bad".to_string(),
                    })
                }
            }
        "#;

        let module = scan_temp_crate(source);
        let error_record = module
            .records
            .iter()
            .find(|record| record.name == "AppError")
            .expect("AppError should be scanned as a record");

        assert!(error_record.is_error);
        assert_eq!(error_record.fields.len(), 2);
        assert_eq!(module.functions.len(), 1);
    }

    #[test]
    fn record_impl_scanned() {
        let source = r#"
            #[boltffi::data]
            pub struct Point {
                pub x: f64,
                pub y: f64,
            }

            #[boltffi::data(impl)]
            impl Point {
                pub fn origin() -> Self {
                    Self { x: 0.0, y: 0.0 }
                }

                pub fn magnitude(&self) -> f64 {
                    (self.x * self.x + self.y * self.y).sqrt()
                }
            }
        "#;

        let module = scan_temp_crate(source);

        let record = module.find_record("Point").expect("Point not found");
        assert_eq!(record.fields.len(), 2);
        assert_eq!(record.constructors.len(), 1);
        assert_eq!(record.constructors[0].name, "origin");
        assert_eq!(record.methods.len(), 1);
        assert_eq!(record.methods[0].name, "magnitude");
    }

    #[test]
    fn record_impl_cross_file_impl_before_struct() {
        let module = scan_temp_crate_multi(&[
            ("lib.rs", "pub mod record_impl;\npub mod record_struct;\n"),
            (
                "record_impl.rs",
                r#"
                        use super::record_struct::Point;

                        #[boltffi::data(impl)]
                        impl Point {
                            pub fn origin() -> Self {
                                Self { x: 0.0, y: 0.0 }
                            }

                            pub fn magnitude(&self) -> f64 {
                                (self.x * self.x + self.y * self.y).sqrt()
                            }
                        }
                    "#,
            ),
            (
                "record_struct.rs",
                r#"
                        #[boltffi::data]
                        pub struct Point {
                            pub x: f64,
                            pub y: f64,
                        }
                    "#,
            ),
        ]);

        let record = module.find_record("Point").expect("Point not found");
        assert_eq!(record.fields.len(), 2);
        assert_eq!(record.constructors.len(), 1);
        assert_eq!(record.constructors[0].name, "origin");
        assert_eq!(record.methods.len(), 1);
        assert_eq!(record.methods[0].name, "magnitude");
    }

    #[test]
    fn symbol_collision_between_method_and_function_is_detected() {
        let mut module = Module::new("test");
        module.functions.push(Function::new("point_distance"));
        module
            .records
            .push(Record::new("Point").with_method(Method::new("distance", Receiver::Ref)));

        let result = validate_no_symbol_collisions(&module);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("FFI symbol collision"),
            "error should mention collision: {}",
            err
        );
        assert!(
            err.contains("Point::distance()"),
            "error should mention the method: {}",
            err
        );
        assert!(
            err.contains("fn point_distance()"),
            "error should mention the function: {}",
            err
        );
    }

    #[test]
    fn no_collision_when_names_differ() {
        let mut module = Module::new("test");
        module.functions.push(Function::new("echo_point"));
        module
            .records
            .push(Record::new("Point").with_method(Method::new("distance", Receiver::Ref)));

        assert!(validate_no_symbol_collisions(&module).is_ok());
    }

    fn scan_temp_crate_result(source: &str) -> Result<Module, String> {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_root = std::env::temp_dir().join(format!(
            "boltffi_scan_diag_{}_{}",
            std::process::id(),
            unique_suffix
        ));
        let src_dir = temp_root.join("src");
        fs::create_dir_all(&src_dir).expect("create src dir");
        fs::write(src_dir.join("lib.rs"), source).expect("write lib.rs");
        let result = scan_crate_with_pointer_width(&temp_root, "testlib", None);
        fs::remove_dir_all(&temp_root).ok();
        result
    }

    #[test]
    fn unresolved_generic_user_type_in_return_is_hard_error() {
        // `Request<MyEffect>` is a user-defined generic wrapper the scanner
        // does not understand today (only Vec/Option/Result/Arc/Box handled).
        // This previously silently dropped the whole function from the
        // generated binding. The scanner must now fail with a diagnostic
        // naming the offending type and context.
        let source = r#"
            use boltffi::*;

            #[data]
            pub enum MyEffect { A, B }

            pub struct Request<T> { pub id: u32, pub effect: T }

            #[export]
            pub fn dispatch() -> Vec<Request<MyEffect>> {
                Vec::new()
            }
        "#;

        let err = scan_temp_crate_result(source).expect_err("scan should fail");
        assert!(
            err.contains("unresolved type"),
            "error should call out unresolved type: {err}"
        );
        assert!(
            err.contains("Request"),
            "error should name the offending type: {err}"
        );
        assert!(
            err.contains("dispatch"),
            "error should name the containing function: {err}"
        );
    }

    #[test]
    fn generic_data_struct_use_site_produces_monomorphized_record() {
        // `#[boltffi::data] struct Request<T>` plus `fn dispatch() -> Vec<Request<MyEffect>>`
        // should yield a concrete `RequestMyEffect` record in the module,
        // with fields substituted to their concrete types. The generic
        // template itself must not appear in the module.
        let source = r#"
            use boltffi::*;

            #[data]
            pub enum MyEffect { A, B }

            #[data]
            pub struct Request<T> {
                pub id: u32,
                pub effect: T,
            }

            #[export]
            pub fn dispatch() -> Vec<Request<MyEffect>> {
                Vec::new()
            }
        "#;

        let module = scan_temp_crate(source);

        assert!(
            module.find_record("RequestMyEffect").is_some(),
            "monomorphized record missing; records: {:?}",
            module.records.iter().map(|r| &r.name).collect::<Vec<_>>()
        );
        let record = module.find_record("RequestMyEffect").unwrap();
        let field_names: Vec<&str> = record.fields.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(field_names, vec!["id", "effect"]);

        assert!(
            module.find_record("Request").is_none(),
            "generic template `Request<T>` must not appear in the module"
        );

        let func = module
            .functions
            .iter()
            .find(|f| f.name == "dispatch")
            .expect("dispatch function scanned");
        let ret_fmt = format!("{:?}", &func.returns);
        assert!(
            ret_fmt.contains("RequestMyEffect"),
            "return type should reference the monomorphized record, got {ret_fmt}"
        );
    }

    #[test]
    fn generic_data_struct_works_as_parameter_and_in_containers() {
        // Covers two things:
        //  - the monomorphized record flows through a parameter position,
        //    not just a return,
        //  - `Option<Request<T>>` wraps the monomorphized record correctly.
        let source = r#"
            use boltffi::*;

            #[data]
            pub enum MyEffect { A, B }

            #[data]
            pub struct Request<T> {
                pub id: u32,
                pub effect: T,
            }

            #[export]
            pub fn accept(req: Option<Request<MyEffect>>) -> u32 {
                req.map(|r| r.id).unwrap_or(0)
            }
        "#;

        let module = scan_temp_crate(source);

        assert!(
            module.find_record("RequestMyEffect").is_some(),
            "monomorphized record missing"
        );
        let func = module
            .functions
            .iter()
            .find(|f| f.name == "accept")
            .expect("accept function scanned");
        let param = func.inputs.first().expect("accept has a parameter");
        let param_fmt = format!("{:?}", param);
        assert!(
            param_fmt.contains("RequestMyEffect"),
            "parameter should reference the monomorphized record, got {param_fmt}"
        );
    }

    #[test]
    fn generic_data_struct_instantiates_distinct_records_per_type_arg() {
        // Two different concrete arguments should produce two distinct
        // monomorphized records that do not alias.
        let source = r#"
            use boltffi::*;

            #[data]
            pub enum EventA { X, Y }

            #[data]
            pub enum EventB { P, Q }

            #[data]
            pub struct Request<T> {
                pub id: u32,
                pub effect: T,
            }

            #[export]
            pub fn a() -> Request<EventA> { Request { id: 0, effect: EventA::X } }

            #[export]
            pub fn b() -> Request<EventB> { Request { id: 0, effect: EventB::P } }
        "#;

        let module = scan_temp_crate(source);

        assert!(
            module.find_record("RequestEventA").is_some(),
            "RequestEventA missing"
        );
        assert!(
            module.find_record("RequestEventB").is_some(),
            "RequestEventB missing"
        );
        assert!(
            module.find_record("Request").is_none(),
            "generic template `Request<T>` must not appear directly"
        );
    }

    #[test]
    fn unresolved_user_type_in_param_is_hard_error() {
        // A `#[export]` impl method taking a type the scanner cannot resolve
        // must fail instead of silently disappearing from the binding.
        let source = r#"
            use boltffi::*;

            pub struct OpaqueThing;

            pub struct Core;

            #[export]
            impl Core {
                pub fn new() -> Self { Core }

                pub fn consume(&self, thing: OpaqueThing) {
                    let _ = thing;
                }
            }
        "#;

        let err = scan_temp_crate_result(source).expect_err("scan should fail");
        assert!(
            err.contains("unresolved type"),
            "error should call out unresolved type: {err}"
        );
        assert!(
            err.contains("OpaqueThing"),
            "error should name the offending type: {err}"
        );
        assert!(
            err.contains("consume"),
            "error should name the containing method: {err}"
        );
    }
}
