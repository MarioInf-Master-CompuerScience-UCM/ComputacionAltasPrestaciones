CXX := icpx
SRCDIR := ./
BUILDDIR := ./
CFLAGS := -O3 -ipo -std=c++11 -fiopenmp -fopenmp-targets=spir64
EXTRA_CFLAGS :=
LIBFLAGS := -fiopenmp -fopenmp-targets=spir64 -ipo

ifdef perf_num
	EXTRA_CFLAGS += -D PERF_NUM
endif

TARGET := mandelbrot

icpx: $(TARGET)

SOURCES := $(wildcard $(SRCDIR)/*.cpp)
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.cpp=.o))

$(TARGET): $(OBJECTS)
	@echo " Linking..."
	$(CXX) $^ $(LIBFLAGS) -o $(TARGET)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(BUILDDIR)
	$(CXX) -c $(CFLAGS) $(EXTRA_CFLAGS) -o $@ $< 

run: $(TARGET)
ifeq ($(shell uname -s),Darwin)
	@export  DYLD_LIBRARY_PATH="$(LIBRARY_PATH)"; ./$(TARGET) $(option)
else
	./$(TARGET) $(option)
endif

clean:
	@echo " Cleaning..."
	@rm -fr $(BUILDDIR) $(TARGET) 2>/dev/null || true
	@rm -f *.o
	@rm -f *.png

.PHONY: clean
