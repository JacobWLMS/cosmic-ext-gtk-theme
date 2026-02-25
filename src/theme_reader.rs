use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize)]
pub struct Rgba {
    pub red: f32,
    pub green: f32,
    pub blue: f32,
    pub alpha: f32,
}

impl Rgba {
    pub fn to_css(&self) -> String {
        let r = (self.red * 255.0).round().clamp(0.0, 255.0) as u8;
        let g = (self.green * 255.0).round().clamp(0.0, 255.0) as u8;
        let b = (self.blue * 255.0).round().clamp(0.0, 255.0) as u8;
        if (self.alpha - 1.0).abs() < 0.001 {
            format!("rgba({}, {}, {}, 1.00)", r, g, b)
        } else {
            format!("rgba({}, {}, {}, {:.2})", r, g, b, self.alpha)
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Component {
    pub base: Rgba,
    pub hover: Rgba,
    pub pressed: Rgba,
    pub selected: Rgba,
    pub selected_text: Rgba,
    pub focus: Rgba,
    pub divider: Rgba,
    pub on: Rgba,
    pub disabled: Rgba,
    pub on_disabled: Rgba,
    pub border: Rgba,
    pub disabled_border: Rgba,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Container {
    pub base: Rgba,
    pub component: Component,
    pub divider: Rgba,
    pub on: Rgba,
    pub small_widget: Rgba,
}

#[derive(Debug, Clone, Deserialize)]
pub enum FontWeight {
    Thin,
    ExtraLight,
    Light,
    Normal,
    Medium,
    Semibold,
    Bold,
    ExtraBold,
    Black,
}

#[derive(Debug, Clone, Deserialize)]
pub enum FontStretch {
    UltraCondensed,
    ExtraCondensed,
    Condensed,
    SemiCondensed,
    Normal,
    SemiExpanded,
    Expanded,
    ExtraExpanded,
    UltraExpanded,
}

#[derive(Debug, Clone, Deserialize)]
pub enum FontStyle {
    Normal,
    Italic,
    Oblique,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FontConfig {
    pub family: String,
    #[allow(dead_code)]
    pub weight: FontWeight,
    #[allow(dead_code)]
    pub stretch: FontStretch,
    #[allow(dead_code)]
    pub style: FontStyle,
}

pub struct CosmicTheme {
    pub is_dark: bool,
    pub accent: Component,
    pub destructive: Component,
    pub warning: Component,
    pub success: Component,
    pub background: Container,
    pub primary: Container,
    pub secondary: Container,
    pub interface_font: Option<FontConfig>,
}

fn cosmic_theme_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("cosmic")
}

fn read_ron_file<T: for<'de> Deserialize<'de>>(path: &std::path::Path) -> Option<T> {
    let contents = fs::read_to_string(path)
        .map_err(|e| log::warn!("Failed to read {}: {}", path.display(), e))
        .ok()?;
    ron::from_str(&contents)
        .map_err(|e| log::warn!("Failed to parse {}: {}", path.display(), e))
        .ok()
}

impl CosmicTheme {
    pub fn load() -> Option<Self> {
        let cosmic_dir = cosmic_theme_dir();
        let mode_dir = cosmic_dir.join("com.system76.CosmicTheme.Mode/v1");
        let is_dark: bool = read_ron_file(&mode_dir.join("is_dark")).unwrap_or(true);

        let theme_name = if is_dark {
            "com.system76.CosmicTheme.Dark"
        } else {
            "com.system76.CosmicTheme.Light"
        };
        let theme_dir = cosmic_dir.join(theme_name).join("v1");

        let accent: Component = read_ron_file(&theme_dir.join("accent"))?;
        let destructive: Component = read_ron_file(&theme_dir.join("destructive"))?;
        let warning: Component = read_ron_file(&theme_dir.join("warning"))?;
        let success: Component = read_ron_file(&theme_dir.join("success"))?;
        let background: Container = read_ron_file(&theme_dir.join("background"))?;
        let primary: Container = read_ron_file(&theme_dir.join("primary"))?;
        let secondary: Container = read_ron_file(&theme_dir.join("secondary"))?;

        // Font config is in a different location
        let tk_dir = cosmic_dir.join("com.system76.CosmicTk/v1");
        let interface_font: Option<FontConfig> = read_ron_file(&tk_dir.join("interface_font"));

        Some(CosmicTheme {
            is_dark,
            accent,
            destructive,
            warning,
            success,
            background,
            primary,
            secondary,
            interface_font,
        })
    }

    #[allow(dead_code)]
    pub fn theme_config_dir(&self) -> PathBuf {
        let theme_name = if self.is_dark {
            "com.system76.CosmicTheme.Dark"
        } else {
            "com.system76.CosmicTheme.Light"
        };
        cosmic_theme_dir().join(theme_name).join("v1")
    }
}
