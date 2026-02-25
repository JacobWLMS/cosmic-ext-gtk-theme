name := 'cosmic-ext-gtk-theme'
appid := 'dev.jacobwlms.CosmicExtGtkTheme'

rootdir := ''
prefix := '/usr'

base-dir := absolute_path(clean(rootdir / prefix))
bin-dir := base-dir / 'bin'
share-dir := base-dir / 'share'
applications-dir := share-dir / 'applications'

default: build-release

build-debug *args:
    cargo build {{args}}

build-release *args:
    cargo build --release {{args}}

install:
    install -Dm0755 target/release/{{name}} {{bin-dir}}/{{name}}
    install -Dm0644 res/{{appid}}.desktop {{applications-dir}}/{{appid}}.desktop

uninstall:
    rm -f {{bin-dir}}/{{name}}
    rm -f {{applications-dir}}/{{appid}}.desktop

clean:
    cargo clean
