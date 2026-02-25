PREFIX ?= /usr
BINDIR ?= $(PREFIX)/bin
DATADIR ?= $(PREFIX)/share
APPID = dev.jacobwlms.CosmicExtGtkTheme

all: build

build:
	cargo build --release

install: build
	install -Dm0755 target/release/cosmic-ext-gtk-theme $(DESTDIR)$(BINDIR)/cosmic-ext-gtk-theme
	install -Dm0644 res/$(APPID).desktop $(DESTDIR)$(DATADIR)/applications/$(APPID).desktop

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/cosmic-ext-gtk-theme
	rm -f $(DESTDIR)$(DATADIR)/applications/$(APPID).desktop

clean:
	cargo clean

.PHONY: all build install uninstall clean
