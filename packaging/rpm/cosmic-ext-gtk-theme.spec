Name:           cosmic-ext-gtk-theme
Version:        0.1.0
Release:        1%{?dist}
Summary:        Sync COSMIC desktop theming to GTK applications
License:        GPL-3.0
URL:            https://github.com/JacobWLMS/cosmic-ext-gtk-theme
Source0:        %{name}-%{version}.tar.gz

BuildRequires:  rust >= 1.80
BuildRequires:  cargo
BuildRequires:  gcc
BuildRequires:  pkg-config
BuildRequires:  libxkbcommon-devel
BuildRequires:  wayland-devel
BuildRequires:  mesa-libEGL-devel
BuildRequires:  vulkan-loader-devel
BuildRequires:  libinput-devel

%description
A COSMIC desktop application that syncs COSMIC theming to GTK applications.
Removes window button backgrounds, applies accent colors to window controls,
syncs fonts, and handles Flatpak apps. Automatically re-applies when the
COSMIC theme changes.

%prep
%autosetup

%build
cargo build --release

%install
install -Dm0755 target/release/%{name} %{buildroot}%{_bindir}/%{name}
install -Dm0644 res/dev.jacobwlms.CosmicExtGtkTheme.desktop %{buildroot}%{_datadir}/applications/dev.jacobwlms.CosmicExtGtkTheme.desktop

%files
%license LICENSE
%{_bindir}/%{name}
%{_datadir}/applications/dev.jacobwlms.CosmicExtGtkTheme.desktop

%changelog
* Tue Feb 25 2026 Jacob WLMS - 0.1.0-1
- Initial release
