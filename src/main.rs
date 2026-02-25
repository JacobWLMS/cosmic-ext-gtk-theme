mod app;
mod config;
mod css_generator;
mod theme_reader;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    cosmic::app::run::<app::App>(cosmic::app::Settings::default(), ())?;
    Ok(())
}
