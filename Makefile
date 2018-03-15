DATADIR=data

FILENAMES=t10k-images-idx3-ubyte t10k-labels-idx1-ubyte train-images-idx3-ubyte train-labels-idx1-ubyte

DATAFILES=$(patsubst %,$(DATADIR)/%,$(FILENAMES))
DATAZIPS=$(patsubst %,%.gz,$(DATAFILES))

$(info "$(DATAFILES)")
$(info "$(DATAZIPS)")

all: $(DATAFILES)

.PHONY: mkdirs

mkdirs:
	@-mkdir -p $(DATADIR)

$(DATAFILES): $(DATADIR)/%: $(DATADIR)/%.gz | mkdirs
	gunzip -d $@

$(DATAZIPS): $(DATADIR)/%: | mkdirs
	wget http://yann.lecun.com/exdb/mnist/$(patsubst $(DATADIR)/%,%,$@) -nc
	mv $(patsubst $(DATADIR)/%,%,$@) ./$@ -f