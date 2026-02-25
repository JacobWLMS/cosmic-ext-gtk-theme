use cosmic::{
    Application, Element,
    app::{Core, Task},
    cosmic_theme, executor,
    iced::{
        Alignment, Length, Subscription,
        futures::SinkExt,
        stream,
    },
    theme,
    widget,
};
use std::path::PathBuf;

use crate::config::Config;
use crate::css_generator;

#[derive(Clone, Debug)]
pub enum Message {
    ToggleSync(bool),
    ToggleButtonBackgrounds(bool),
    ToggleFlatpaks(bool),
    ToggleAccentHeaderbar(bool),
    ToggleSyncFont(bool),
    ToggleFlatpakFonts(bool),
    ApplyNow,
    ApplyResult(String),
    Restore,
    RestoreResult(String),
    ThemeChanged,
}

pub struct App {
    core: Core,
    config: Config,
    config_handler: Option<cosmic::cosmic_config::Config>,
    status_text: String,
    applying: bool,
}

impl App {
    pub const APP_ID: &'static str = "dev.jacobwlms.CosmicExtGtkTheme";

    fn save_config(&self) {
        if let Some(ref handler) = self.config_handler {
            if let Err(e) =
                cosmic::cosmic_config::CosmicConfigEntry::write_entry(&self.config, handler)
            {
                log::error!("Failed to save config: {}", e);
            }
        }
    }

    fn do_apply(&mut self) -> Task<Message> {
        self.applying = true;
        self.status_text = "Applying theme...".into();
        let config = self.config.clone();

        cosmic::task::future(async move {
            let result = css_generator::apply_theme(&config);
            let mut status = String::new();

            match &result.native_gtk4 {
                Ok(_) => status.push_str("GTK4: OK\n"),
                Err(e) => status.push_str(&format!("GTK4: {}\n", e)),
            }
            match &result.native_gtk3 {
                Ok(_) => status.push_str("GTK3: OK\n"),
                Err(e) => status.push_str(&format!("GTK3: {}\n", e)),
            }

            if config.apply_to_flatpaks {
                status.push_str(&format!("Flatpak: {} apps updated", result.flatpak_count));
                if !result.flatpak_errors.is_empty() {
                    status.push_str(&format!(
                        " ({} errors)",
                        result.flatpak_errors.len()
                    ));
                }
            }

            Message::ApplyResult(status)
        })
    }
}

impl Application for App {
    type Executor = executor::multi::Executor;
    type Flags = ();
    type Message = Message;
    const APP_ID: &'static str = "dev.jacobwlms.CosmicExtGtkTheme";

    fn core(&self) -> &Core {
        &self.core
    }

    fn core_mut(&mut self) -> &mut Core {
        &mut self.core
    }

    fn init(core: Core, _flags: Self::Flags) -> (Self, Task<Message>) {
        let (config_handler, config) = Config::load();

        let mut app = App {
            core,
            config,
            config_handler,
            status_text: "Ready".into(),
            applying: false,
        };

        let task = if app.config.sync_enabled {
            app.do_apply()
        } else {
            Task::none()
        };

        (app, task)
    }

    fn header_start(&self) -> Vec<Element<'_, Message>> {
        vec![]
    }

    fn view(&self) -> Element<'_, Message> {
        let cosmic_theme::Spacing {
            space_xs,
            space_m,
            space_l,
            ..
        } = theme::spacing();

        let heading = widget::text::title3("GTK Theme Sync");

        let description = widget::text::body(
            "Syncs COSMIC desktop theming to GTK applications for a consistent look. \
             Automatically re-applies when the COSMIC theme changes.",
        );

        // General settings
        let sync_toggle = widget::settings::item(
            "Enable theme sync",
            widget::toggler(self.config.sync_enabled)
                .on_toggle(Message::ToggleSync),
        );

        let general_section = widget::settings::section()
            .title("General")
            .add(sync_toggle);

        // Appearance settings
        let button_toggle = widget::settings::item(
            "Remove window button backgrounds",
            widget::toggler(self.config.remove_button_backgrounds)
                .on_toggle(Message::ToggleButtonBackgrounds),
        );

        let accent_toggle = widget::settings::item(
            "Accent-colored window controls on focus",
            widget::toggler(self.config.apply_accent_headerbar)
                .on_toggle(Message::ToggleAccentHeaderbar),
        );

        let font_toggle = widget::settings::item(
            "Sync COSMIC font to GTK apps",
            widget::toggler(self.config.sync_font)
                .on_toggle(Message::ToggleSyncFont),
        );

        let appearance_section = widget::settings::section()
            .title("Appearance")
            .add(button_toggle)
            .add(accent_toggle)
            .add(font_toggle);

        // Flatpak settings
        let flatpak_toggle = widget::settings::item(
            "Apply theme to Flatpak apps",
            widget::toggler(self.config.apply_to_flatpaks)
                .on_toggle(Message::ToggleFlatpaks),
        );

        let flatpak_font_toggle = widget::settings::item(
            "Grant Flatpak apps access to system fonts",
            widget::toggler(self.config.fix_flatpak_fonts)
                .on_toggle(Message::ToggleFlatpakFonts),
        );

        let flatpak_section = widget::settings::section()
            .title("Flatpak")
            .add(flatpak_toggle)
            .add(flatpak_font_toggle);

        // Action buttons
        let apply_btn = widget::button::suggested("Apply Now")
            .on_press_maybe(if self.applying {
                None
            } else {
                Some(Message::ApplyNow)
            });

        let restore_btn = widget::button::destructive("Restore Defaults")
            .on_press_maybe(if self.applying {
                None
            } else {
                Some(Message::Restore)
            });

        let buttons = widget::row()
            .spacing(space_xs)
            .push(apply_btn)
            .push(restore_btn);

        // Status
        let status_section = widget::settings::section()
            .title("Status")
            .add(widget::settings::item(
                "Last result",
                widget::text::body(&self.status_text),
            ));

        let content = widget::column()
            .spacing(space_m)
            .padding(space_l)
            .max_width(600)
            .push(heading)
            .push(description)
            .push(general_section)
            .push(appearance_section)
            .push(flatpak_section)
            .push(buttons)
            .push(status_section);

        widget::container(
            widget::scrollable(content).width(Length::Fill),
        )
        .width(Length::Fill)
        .height(Length::Fill)
        .align_x(Alignment::Center)
        .into()
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ToggleSync(enabled) => {
                self.config.sync_enabled = enabled;
                self.save_config();
                if enabled {
                    return self.do_apply();
                }
            }
            Message::ToggleButtonBackgrounds(enabled) => {
                self.config.remove_button_backgrounds = enabled;
                self.save_config();
                if self.config.sync_enabled {
                    return self.do_apply();
                }
            }
            Message::ToggleFlatpaks(enabled) => {
                self.config.apply_to_flatpaks = enabled;
                self.save_config();
                if self.config.sync_enabled {
                    return self.do_apply();
                }
            }
            Message::ToggleAccentHeaderbar(enabled) => {
                self.config.apply_accent_headerbar = enabled;
                self.save_config();
                if self.config.sync_enabled {
                    return self.do_apply();
                }
            }
            Message::ToggleSyncFont(enabled) => {
                self.config.sync_font = enabled;
                self.save_config();
                if self.config.sync_enabled {
                    return self.do_apply();
                }
            }
            Message::ToggleFlatpakFonts(enabled) => {
                self.config.fix_flatpak_fonts = enabled;
                self.save_config();
                if self.config.sync_enabled {
                    return self.do_apply();
                }
            }
            Message::ApplyNow => {
                return self.do_apply();
            }
            Message::ApplyResult(status) => {
                self.status_text = status;
                self.applying = false;
            }
            Message::Restore => {
                self.applying = true;
                self.status_text = "Restoring...".into();
                return cosmic::task::future(async {
                    match css_generator::restore_cosmic_symlinks() {
                        Ok(msg) => Message::RestoreResult(format!("Restored: {}", msg)),
                        Err(e) => Message::RestoreResult(format!("Error: {}", e)),
                    }
                });
            }
            Message::RestoreResult(status) => {
                self.status_text = status;
                self.applying = false;
            }
            Message::ThemeChanged => {
                if self.config.sync_enabled {
                    return self.do_apply();
                }
            }
        }
        Task::none()
    }

    fn subscription(&self) -> Subscription<Message> {
        if !self.config.sync_enabled {
            return Subscription::none();
        }

        let config_dir = dirs::config_dir().unwrap_or_else(|| PathBuf::from("~/.config"));
        let theme_dir = config_dir.join("cosmic");

        Subscription::run_with_id(
            "theme-watcher",
            stream::channel(1, move |mut sender| {
                let theme_dir = theme_dir.clone();
                async move {
                    use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
                    use tokio::sync::mpsc;

                    let (tx, mut rx) = mpsc::channel(10);

                    let mut watcher = RecommendedWatcher::new(
                        move |res: Result<notify::Event, notify::Error>| {
                            if let Ok(event) = res {
                                if event.kind.is_modify() || event.kind.is_create() {
                                    let _ = tx.blocking_send(());
                                }
                            }
                        },
                        Config::default(),
                    )
                    .expect("Failed to create file watcher");

                    let _ = watcher.watch(&theme_dir, RecursiveMode::Recursive);

                    loop {
                        if rx.recv().await.is_some() {
                            // Debounce
                            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
                            while rx.try_recv().is_ok() {}

                            let _ = sender.send(Message::ThemeChanged).await;
                        }
                    }
                }
            }),
        )
    }
}
