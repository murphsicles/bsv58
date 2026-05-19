# bsv58 — Zeta build
# b58: Ultra-fast Base58 codec for Bitcoin SV
# Pure Zeta implementation (main branch).
# Rust version is on the `rust` branch.

ZETA_HOME ?= /home/zeta/.openclaw/workspace/zeta
ZETAC ?= $(ZETA_HOME)/bin/zetac

.PHONY: all test clean

all: test

# Compile and run tests
test:
	@echo "Testing bsv58.z..."
	$(ZETAC) src/bsv58.z

clean:
	rm -f *.o *.out
