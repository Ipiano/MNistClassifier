DATADIR=data

FILENAMES=t10k-images-idx3-ubyte t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte

DATAFILES=$(patsubst %,$(DATADIR)/%,$(FILENAMES))
DATAZIPS=$(patsubst %,%.gz,$(DATAFILES))

# Cunn will fail to install if CUDA toolkit not installed
# GraphicsMagick will install, but won't run without graphicsmagick executable in PATH
LUADEPS = struct cunn graphicsmagick

$(info "$(DATAFILES)")
$(info "$(DATAZIPS)")

.PHONY: mkdirs installdeps

all: $(DATAFILES) | installdeps

installdeps: $(LUADEPS)

$(LUADEPS): %:
	./luarocks_install.bash $@

mkdirs:
	@-mkdir -p $(DATADIR)

$(DATAFILES): $(DATADIR)/%: $(DATADIR)/%.gz
	gunzip -d -k $@

$(DATAZIPS): $(DATADIR)/%: | mkdirs
	@-rm "./$(patsubst $(DATADIR)/%,%,$@)"
	wget http://yann.lecun.com/exdb/mnist/$(patsubst $(DATADIR)/%,%,$@) -nc
	mv $(patsubst $(DATADIR)/%,%,$@) ./$@ -f
