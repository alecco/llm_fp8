.PHONY: all build clean profile

CMAKE := cmake

BUILD_DIR := build

all: build

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && $(CMAKE) -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_BUILD_TYPE=Release ..
	@$(MAKE) -j 10 -C $(BUILD_DIR)

clean:
	@rm -rf $(BUILD_DIR)
