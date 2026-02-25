use cosmic::cosmic_config::{self, cosmic_config_derive::CosmicConfigEntry, CosmicConfigEntry};
use serde::{Deserialize, Serialize};

use crate::app::App;

pub const CONFIG_VERSION: u64 = 1;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, CosmicConfigEntry)]
#[version = 1]
pub struct Config {
    pub remove_button_backgrounds: bool,
    pub apply_to_flatpaks: bool,
    pub sync_enabled: bool,
    pub apply_accent_headerbar: bool,
    pub sync_font: bool,
    pub fix_flatpak_fonts: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            remove_button_backgrounds: true,
            apply_to_flatpaks: true,
            sync_enabled: true,
            apply_accent_headerbar: true,
            sync_font: true,
            fix_flatpak_fonts: true,
        }
    }
}

impl Config {
    pub fn load() -> (Option<cosmic_config::Config>, Self) {
        match cosmic_config::Config::new(App::APP_ID, CONFIG_VERSION) {
            Ok(config_handler) => {
                let config = Config::get_entry(&config_handler).unwrap_or_else(|(errs, config)| {
                    log::warn!("errors loading config: {:?}", errs);
                    config
                });
                (Some(config_handler), config)
            }
            Err(err) => {
                log::error!("failed to create config handler: {}", err);
                (None, Config::default())
            }
        }
    }
}
