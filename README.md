# COSMIC GTK Theme Sync

A small COSMIC desktop application that syncs COSMIC theming to GTK applications, making them look consistent with native COSMIC apps.

## Features

- **Remove window button backgrounds** — Strips the circular backgrounds behind close/minimize/maximize buttons, matching COSMIC's clean style
- **Accent-colored window controls** — Window control icons use the COSMIC accent color when focused, with proper hover/pressed states using COSMIC's component colors
- **Font sync** — Applies COSMIC's interface font (`Noto Sans` or whatever you've configured) to GTK apps
- **Flatpak support** — Applies all theming to Flatpak app config dirs and grants them access to system fonts
- **Auto-sync** — Watches COSMIC theme config for changes and re-applies automatically (accent color change, dark/light switch, etc.)
- **GTK3 + GTK4** — Handles both toolkits with the correct selectors

## Screenshots

_Coming soon_

## Installation

### From source

```bash
git clone https://github.com/JacobWLMS/cosmic-ext-gtk-theme.git
cd cosmic-ext-gtk-theme
cargo build --release
sudo make install
```

Or with `just`:

```bash
just build-release
sudo just install
```

### Fedora / RPM

```bash
# From the release .rpm file:
sudo dnf install cosmic-ext-gtk-theme-0.1.0-1.fc43.x86_64.rpm
```

### Debian / Ubuntu

```bash
sudo dpkg -i cosmic-ext-gtk-theme_0.1.0-1_amd64.deb
```

### Arch Linux

```bash
# Using the PKGBUILD from packaging/arch/:
makepkg -si
```

### Flatpak

```bash
flatpak install cosmic-ext-gtk-theme.flatpak
```

### Portable binary

Download the tarball from [Releases](https://github.com/JacobWLMS/cosmic-ext-gtk-theme/releases), extract, and run `install.sh`:

```bash
tar xzf cosmic-ext-gtk-theme-0.1.0-x86_64-linux.tar.gz
cd cosmic-ext-gtk-theme-0.1.0
./install.sh
```

## Usage

Launch from your app launcher or run:

```bash
cosmic-ext-gtk-theme
```

The app opens a settings window with toggles:

| Section | Setting | Default |
|---|---|---|
| **General** | Enable theme sync | On |
| **Appearance** | Remove window button backgrounds | On |
| | Accent-colored window controls on focus | On |
| | Sync COSMIC font to GTK apps | On |
| **Flatpak** | Apply theme to Flatpak apps | On |
| | Grant Flatpak apps access to system fonts | On |

Changes apply immediately. The app watches `~/.config/cosmic/` for theme changes and re-applies automatically.

### Restore defaults

Click **Restore Defaults** in the app to put back COSMIC's original symlinks and remove all custom CSS.

## How it works

1. Reads COSMIC's generated CSS from `~/.config/gtk-4.0/cosmic/{dark,light}.css` (preserving exact palette colors)
2. Reads accent, background, and component colors from COSMIC theme RON files in `~/.config/cosmic/com.system76.CosmicTheme.{Dark,Light}/v1/`
3. Reads the interface font from `~/.config/cosmic/com.system76.CosmicTk/v1/interface_font`
4. Generates GTK4 CSS (with `windowcontrols` selectors) and GTK3 CSS (with `.titlebutton` selectors)
5. Writes to `~/.config/gtk-{3,4}.0/gtk.css` for native apps
6. Writes to `~/.var/app/<id>/config/gtk-{3,4}.0/gtk.css` for each Flatpak app
7. Uses `flatpak override` to grant font access to Flatpak apps
8. Watches the COSMIC config directory with `inotify` and re-applies on changes

## Building

Requires Rust 1.80+ and the following system libraries:

- `libxkbcommon-dev` / `libxkbcommon-devel`
- `libwayland-dev` / `wayland-devel`
- `libegl-dev` / `mesa-libEGL-devel`
- `libvulkan-dev` / `vulkan-loader-devel`
- `libinput-dev` / `libinput-devel`

## License

GPL-3.0
