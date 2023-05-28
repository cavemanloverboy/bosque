.PHONY: default clean mock

default: build

build:
	@echo "\n--- Building library... ---\n"
	RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
	@echo "\n--- Library built at target/release ---\n"

clean:
	@echo "\n--- Cleaning project... ---\n"
	cargo clean
	@echo "\n--- Project cleaned ---\n"

mock: build
	@echo "\n--- Building mock script... ---\n"
	gcc -o abacus_mock.out examples/abacus_mock.c -L./target/release/ -lbosque -O3 -march=native
	@echo "\n--- Executable built at abacus_mock.out ---\n"
