.PHONY: default clean mock

default: build

build:
	@echo "\nBuilding library...\n"
	RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
	@echo "\nLibrary built at target/release\n"

clean:
	@echo "Cleaning project...\n"
	cargo clean
	@echo "\nProject cleaned.\n"

mock: build
	@echo "\nBuilding mock script...\n"
	gcc -o examples/abacus_mock examples/abacus_mock.c -L./target/release/ -lbosque -O3 -march=native
	@echo "\nExecutable built at examples/abacus_mock\n"
