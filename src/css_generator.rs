use crate::config::Config;
use crate::theme_reader::CosmicTheme;
use std::fs;
use std::path::PathBuf;

pub struct ApplyResult {
    pub native_gtk4: Result<PathBuf, String>,
    pub native_gtk3: Result<PathBuf, String>,
    pub flatpak_count: usize,
    pub flatpak_errors: Vec<String>,
}

/// Read the COSMIC-generated CSS file (dark.css or light.css) which already
/// contains all @define-color rules with the correct palette shades.
fn read_cosmic_base_css(is_dark: bool) -> Option<String> {
    let config_dir = dirs::config_dir()?;
    let theme_file = if is_dark { "dark.css" } else { "light.css" };
    let path = config_dir.join("gtk-4.0/cosmic").join(theme_file);
    fs::read_to_string(&path)
        .map_err(|e| log::warn!("Failed to read COSMIC CSS {}: {}", path.display(), e))
        .ok()
}

fn button_removal_css_gtk4(theme: &CosmicTheme) -> String {
    let hover_bg = &theme.background.component.hover;
    let pressed_bg = &theme.background.component.pressed;
    format!(
        "\n/* Window control button styling (COSMIC-style) */\n\
         windowcontrols button {{\n\
         \x20   background: none;\n\
         \x20   background-color: transparent;\n\
         \x20   box-shadow: none;\n\
         \x20   border: none;\n\
         \x20   border-radius: 50%;\n\
         }}\n\
         windowcontrols button:hover {{\n\
         \x20   background: none;\n\
         \x20   background-color: {};\n\
         \x20   box-shadow: none;\n\
         \x20   border: none;\n\
         }}\n\
         windowcontrols button:active {{\n\
         \x20   background: none;\n\
         \x20   background-color: {};\n\
         \x20   box-shadow: none;\n\
         \x20   border: none;\n\
         }}\n\
         windowcontrols button image {{\n\
         \x20   background: none;\n\
         \x20   background-color: transparent;\n\
         }}\n",
        hover_bg.to_css(),
        pressed_bg.to_css()
    )
}

fn button_removal_css_gtk3(theme: &CosmicTheme) -> String {
    let hover_bg = &theme.background.component.hover;
    let pressed_bg = &theme.background.component.pressed;
    format!(
        "\n/* GTK3: Window control button styling (COSMIC-style) */\n\
         headerbar button.titlebutton,\n\
         .titlebar button.titlebutton {{\n\
         \x20   background: none;\n\
         \x20   background-color: transparent;\n\
         \x20   background-image: none;\n\
         \x20   box-shadow: none;\n\
         \x20   border: none;\n\
         }}\n\
         headerbar button.titlebutton:hover,\n\
         .titlebar button.titlebutton:hover {{\n\
         \x20   background: none;\n\
         \x20   background-color: {};\n\
         \x20   background-image: none;\n\
         \x20   box-shadow: none;\n\
         \x20   border: none;\n\
         }}\n\
         headerbar button.titlebutton:active,\n\
         .titlebar button.titlebutton:active {{\n\
         \x20   background: none;\n\
         \x20   background-color: {};\n\
         \x20   background-image: none;\n\
         \x20   box-shadow: none;\n\
         \x20   border: none;\n\
         }}\n",
        hover_bg.to_css(),
        pressed_bg.to_css()
    )
}

fn accent_window_css_gtk4(theme: &CosmicTheme) -> String {
    let accent = &theme.accent.base;
    let hover = &theme.accent.hover;
    format!(
        "\n/* Accent-colored window control icons when focused (COSMIC-style) */\n\
         windowcontrols button image {{\n\
         \x20   color: {};\n\
         \x20   -gtk-icon-style: symbolic;\n\
         }}\n\
         windowcontrols button:hover image {{\n\
         \x20   color: {};\n\
         }}\n\
         windowcontrols button:backdrop image {{\n\
         \x20   color: @headerbar_fg_color;\n\
         }}\n\
         headerbar:backdrop windowcontrols button image {{\n\
         \x20   color: @headerbar_fg_color;\n\
         }}\n",
        accent.to_css(),
        hover.to_css()
    )
}

fn accent_window_css_gtk3(theme: &CosmicTheme) -> String {
    let accent = &theme.accent.base;
    format!(
        "\n/* GTK3: Accent-colored titlebutton icons when focused */\n\
         headerbar button.titlebutton {{\n\
         \x20   color: {};\n\
         }}\n\
         headerbar:backdrop button.titlebutton {{\n\
         \x20   color: @headerbar_fg_color;\n\
         }}\n\
         .titlebar button.titlebutton {{\n\
         \x20   color: {};\n\
         }}\n\
         .titlebar:backdrop button.titlebutton {{\n\
         \x20   color: @headerbar_fg_color;\n\
         }}\n",
        accent.to_css(),
        accent.to_css()
    )
}

fn font_css(theme: &CosmicTheme) -> String {
    match &theme.interface_font {
        Some(font) => format!(
            "\n/* COSMIC interface font */\n\
             * {{\n\
             \x20   font-family: \"{}\", sans-serif;\n\
             }}\n",
            font.family
        ),
        None => String::new(),
    }
}

fn generate_gtk4_css(theme: &CosmicTheme, config: &Config) -> String {
    // Start with COSMIC's generated CSS (all color definitions + palette)
    let mut css = read_cosmic_base_css(theme.is_dark)
        .unwrap_or_else(|| generate_color_definitions(theme));

    if config.remove_button_backgrounds {
        css.push_str(&button_removal_css_gtk4(theme));
    }

    if config.apply_accent_headerbar {
        css.push_str(&accent_window_css_gtk4(theme));
    }

    if config.sync_font {
        css.push_str(&font_css(theme));
    }

    css
}

fn generate_gtk3_css(theme: &CosmicTheme, config: &Config) -> String {
    let mut css = read_cosmic_base_css(theme.is_dark)
        .unwrap_or_else(|| generate_color_definitions(theme));

    if config.remove_button_backgrounds {
        css.push_str(&button_removal_css_gtk4(theme));
        css.push_str(&button_removal_css_gtk3(theme));
    }

    if config.apply_accent_headerbar {
        css.push_str(&accent_window_css_gtk4(theme));
        css.push_str(&accent_window_css_gtk3(theme));
    }

    if config.sync_font {
        css.push_str(&font_css(theme));
    }

    css
}

/// Fallback: generate color definitions from the parsed theme if the COSMIC
/// CSS file isn't available.
fn generate_color_definitions(theme: &CosmicTheme) -> String {
    let mut css = String::from("/* Generated by cosmic-ext-gtk-theme (fallback) */\n");

    let colors = [
        ("window_bg_color", &theme.background.base),
        ("window_fg_color", &theme.background.on),
        ("view_bg_color", &theme.primary.base),
        ("view_fg_color", &theme.primary.on),
        ("headerbar_bg_color", &theme.background.base),
        ("headerbar_fg_color", &theme.background.on),
        ("headerbar_border_color_color", &theme.background.divider),
        ("headerbar_backdrop_color", &theme.background.base),
        ("sidebar_bg_color", &theme.primary.base),
        ("sidebar_fg_color", &theme.primary.on),
        ("sidebar_shade_color", &theme.primary.small_widget),
        ("sidebar_backdrop_color", &theme.primary.component.hover),
        ("secondary_sidebar_bg_color", &theme.secondary.base),
        ("secondary_sidebar_fg_color", &theme.secondary.on),
        ("secondary_sidebar_shade_color", &theme.secondary.small_widget),
        ("secondary_sidebar_backdrop_color", &theme.secondary.component.hover),
        ("card_bg_color", &theme.background.component.base),
        ("card_fg_color", &theme.background.component.on),
        ("thumbnail_bg_color", &theme.background.component.base),
        ("thumbnail_fg_color", &theme.background.component.on),
        ("dialog_bg_color", &theme.primary.base),
        ("dialog_fg_color", &theme.primary.on),
        ("popover_bg_color", &theme.background.component.base),
        ("popover_fg_color", &theme.background.component.on),
        ("accent_color", &theme.accent.base),
        ("accent_bg_color", &theme.accent.base),
        ("accent_fg_color", &theme.accent.on),
        ("destructive_color", &theme.destructive.base),
        ("destructive_bg_color", &theme.destructive.base),
        ("destructive_fg_color", &theme.destructive.on),
        ("warning_color", &theme.warning.base),
        ("warning_bg_color", &theme.warning.base),
        ("warning_fg_color", &theme.warning.on),
        ("success_color", &theme.success.base),
        ("success_bg_color", &theme.success.base),
        ("success_fg_color", &theme.success.on),
        ("error_color", &theme.destructive.base),
        ("error_bg_color", &theme.destructive.base),
        ("error_fg_color", &theme.destructive.on),
    ];

    for (name, color) in &colors {
        css.push_str(&format!("@define-color {} {};\n", name, color.to_css()));
    }

    css
}

fn write_css(path: &std::path::Path, content: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("Failed to create dir: {}", e))?;
    }

    // If it's a symlink, remove it first
    if path.is_symlink() {
        fs::remove_file(path).map_err(|e| format!("Failed to remove symlink: {}", e))?;
    }

    fs::write(path, content).map_err(|e| format!("Failed to write {}: {}", path.display(), e))
}

fn find_flatpak_app_dirs() -> Vec<PathBuf> {
    let flatpak_base = dirs::home_dir()
        .unwrap_or_default()
        .join(".var/app");

    if !flatpak_base.is_dir() {
        return Vec::new();
    }

    let mut dirs = Vec::new();
    if let Ok(entries) = fs::read_dir(&flatpak_base) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            }
        }
    }
    dirs
}

/// Grant Flatpak apps read-only access to system fonts via global override.
fn apply_flatpak_font_override() -> Result<(), String> {
    use std::process::Command;

    // Add font directories to global Flatpak override
    let overrides = [
        "/usr/share/fonts:ro",
        "xdg-data/fonts:ro",
        "~/.fonts:ro",
    ];

    for fs_override in &overrides {
        let output = Command::new("flatpak")
            .args(["override", "--user", &format!("--filesystem={}", fs_override)])
            .output()
            .map_err(|e| format!("Failed to run flatpak override: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            log::warn!("flatpak override for {} failed: {}", fs_override, stderr);
        }
    }

    Ok(())
}

pub fn apply_theme(config: &Config) -> ApplyResult {
    let theme = CosmicTheme::load();

    let (gtk4_css, gtk3_css) = match &theme {
        Some(t) => (generate_gtk4_css(t, config), generate_gtk3_css(t, config)),
        None => {
            return ApplyResult {
                native_gtk4: Err("Failed to read COSMIC theme".into()),
                native_gtk3: Err("Failed to read COSMIC theme".into()),
                flatpak_count: 0,
                flatpak_errors: vec!["Failed to read COSMIC theme".into()],
            };
        }
    };

    let config_dir = dirs::config_dir().unwrap_or_else(|| PathBuf::from("~/.config"));

    // Apply to native GTK4
    let gtk4_path = config_dir.join("gtk-4.0/gtk.css");
    let native_gtk4 = write_css(&gtk4_path, &gtk4_css).map(|()| gtk4_path);

    // Apply to native GTK3
    let gtk3_path = config_dir.join("gtk-3.0/gtk.css");
    let native_gtk3 = write_css(&gtk3_path, &gtk3_css).map(|()| gtk3_path);

    // Apply to Flatpak apps
    let mut flatpak_count = 0;
    let mut flatpak_errors = Vec::new();

    if config.apply_to_flatpaks {
        for app_dir in find_flatpak_app_dirs() {
            let gtk4_dir = app_dir.join("config/gtk-4.0");
            let gtk3_dir = app_dir.join("config/gtk-3.0");

            if gtk4_dir.is_dir() {
                let path = gtk4_dir.join("gtk.css");
                match write_css(&path, &gtk4_css) {
                    Ok(()) => flatpak_count += 1,
                    Err(e) => flatpak_errors.push(format!("{}: {}", app_dir.display(), e)),
                }
            }

            if gtk3_dir.is_dir() {
                let path = gtk3_dir.join("gtk.css");
                match write_css(&path, &gtk3_css) {
                    Ok(()) => flatpak_count += 1,
                    Err(e) => flatpak_errors.push(format!("{}: {}", app_dir.display(), e)),
                }
            }
        }

        // Fix Flatpak font access
        if config.fix_flatpak_fonts {
            if let Err(e) = apply_flatpak_font_override() {
                flatpak_errors.push(format!("Font override: {}", e));
            }
        }
    }

    ApplyResult {
        native_gtk4,
        native_gtk3,
        flatpak_count,
        flatpak_errors,
    }
}

pub fn restore_cosmic_symlinks() -> Result<String, String> {
    let config_dir = dirs::config_dir().unwrap_or_else(|| PathBuf::from("~/.config"));
    let cosmic_css_dir = config_dir.join("gtk-4.0/cosmic");

    let is_dark: bool = {
        let mode_path = config_dir.join("cosmic/com.system76.CosmicTheme.Mode/v1/is_dark");
        std::fs::read_to_string(&mode_path)
            .ok()
            .and_then(|s| ron::from_str(&s).ok())
            .unwrap_or(true)
    };

    let theme_css = if is_dark { "dark.css" } else { "light.css" };
    let target = cosmic_css_dir.join(theme_css);

    if !target.exists() {
        return Err(format!("COSMIC CSS not found at {}", target.display()));
    }

    let mut restored = Vec::new();

    for gtk_ver in &["gtk-4.0", "gtk-3.0"] {
        let css_path = config_dir.join(gtk_ver).join("gtk.css");
        if css_path.exists() || css_path.is_symlink() {
            let _ = fs::remove_file(&css_path);
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(&target, &css_path)
                .map_err(|e| format!("Failed to create symlink: {}", e))?;
            restored.push(format!("{}", css_path.display()));
        }
    }

    // Remove Flatpak CSS files
    for app_dir in find_flatpak_app_dirs() {
        for gtk_ver in &["gtk-4.0", "gtk-3.0"] {
            let css_path = app_dir.join("config").join(gtk_ver).join("gtk.css");
            if css_path.exists() {
                let _ = fs::remove_file(&css_path);
                restored.push(format!("{}", css_path.display()));
            }
        }
    }

    Ok(format!("Restored {} files", restored.len()))
}
