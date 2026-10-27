CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -fPIC -Iinclude
LDFLAGS ?= -shared

BUILD_DIR = bin
SRC_DIR = src
TEST_DIR = tests

LIB = $(BUILD_DIR)/libint_spectrum.so
DEMO_BRESENHAM = $(BUILD_DIR)/bresenham_demo
TEST_CPP = $(BUILD_DIR)/test_int_spectrum

.PHONY: all clean test gui eval benchmark

all: $(LIB) $(DEMO_BRESENHAM) $(TEST_CPP)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(LIB): $(SRC_DIR)/int_spectrum_c_api.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@

$(DEMO_BRESENHAM): bresenham_spectrum.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -Iinclude $< -o $@

$(TEST_CPP): $(TEST_DIR)/test_int_spectrum.cpp $(LIB) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -L$(BUILD_DIR) -lint_spectrum -Wl,-rpath,$(BUILD_DIR) -o $@

test: $(LIB) $(TEST_CPP)
	@echo "--- Running C++ Unit Tests ---"
	@./$(TEST_CPP)
	@echo "--- Running Python Unit Tests ---"
	@python3 $(TEST_DIR)/test_int_spectrum.py

eval: $(LIB)
	@echo "--- Running Evaluation Suite ---"
	@python3 $(TEST_DIR)/run_evaluation_suite.py

benchmark: $(LIB)
	@echo "--- Running FFT Comparison Benchmark ---"
	@python3 $(TEST_DIR)/benchmark_vs_fft.py

gui: $(LIB)
	@python3 $(SRC_DIR)/int_spectrum_gui.py

clean:
	rm -rf $(BUILD_DIR)
