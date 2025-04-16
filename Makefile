CC=gcc
UNAME_S := $(shell uname -s)

# Force disable Vulkan for easier building
CC_FLAGS_EXTRA=
# Add definitions to disable problematic Vulkan video extension headers
CC_FLAGS_EXTRA=-DVK_ENABLE_BETA_EXTENSIONS=0 -DVK_VIDEO_H_=1 -DVK_VIDEO_CODEC_H264STD_H_=1 -DVK_VIDEO_CODEC_H265STD_H_=1 -DVK_VIDEO_CODECS_COMMON_H_=1 -DVK_VIDEO_CODEC_AV1STD_H_=1 -DVK_USE_PLATFORM_MACOS_MVK=1

# Set Vulkan SDK path directly
VULKAN_SDK_PATH=/Users/botond/VulkanSDK/1.4.309.0
VULKAN_SDK_ENV := $(shell echo $$VULKAN_SDK)

# If environment variable is set, use it instead
ifneq ($(VULKAN_SDK_ENV),)
  VULKAN_SDK_PATH=$(VULKAN_SDK_ENV)
endif

# Flag to indicate if Vulkan is available
HAVE_VULKAN=1

ifeq ($(UNAME_S),Darwin)
    # Check for Apple Silicon
    UNAME_P := $(shell uname -p)
    IS_ARM := $(findstring arm,$(UNAME_P))
    
    # macOS - Check for Homebrew OpenMP installation
    BREW_OMP_PATH := $(shell brew --prefix libomp 2>/dev/null)
    ifneq ($(BREW_OMP_PATH),)
        # If Homebrew libomp is installed
        CC=clang
        
        # For Apple Silicon, use OpenMP only for now
        ifneq ($(IS_ARM),)
            CC_FLAGS=-g -Wall -Xpreprocessor -fopenmp -I$(BREW_OMP_PATH)/include -I. $(CC_FLAGS_EXTRA)
            CC_LIBS=-lm -L$(BREW_OMP_PATH)/lib -lomp -framework OpenGL -framework GLUT -L$(VULKAN_SDK_PATH)/macOS/lib -lvulkan -Wl,-rpath,$(VULKAN_SDK_PATH)/macOS/lib
        else
            # For Intel Macs, OpenGL can still be used
            CC_FLAGS=-g -Wall -Xpreprocessor -fopenmp -I$(BREW_OMP_PATH)/include -I. $(CC_FLAGS_EXTRA)
            CC_LIBS=-lm -L$(BREW_OMP_PATH)/lib -lomp -framework OpenGL -framework GLUT -L$(VULKAN_SDK_PATH)/macOS/lib -lvulkan -Wl,-rpath,$(VULKAN_SDK_PATH)/macOS/lib
        endif
    else
        # Check for MacPorts OpenMP
        PORT_OMP_PATH := /opt/local
        ifneq ($(wildcard $(PORT_OMP_PATH)/include/libomp/omp.h),)
            CC=clang
            CC_FLAGS=-g -Wall -Xpreprocessor -fopenmp -I$(PORT_OMP_PATH)/include/libomp -I. $(CC_FLAGS_EXTRA)
            CC_LIBS=-lm -L$(PORT_OMP_PATH)/lib -lomp -framework OpenGL -framework GLUT -L$(VULKAN_SDK_PATH)/macOS/lib -lvulkan -Wl,-rpath,$(VULKAN_SDK_PATH)/macOS/lib
        else
            # Fallback to non-OpenMP and non-GPU version
            CC=clang
            CC_FLAGS=-g -Wall -DDISABLE_OPENMP -I. $(CC_FLAGS_EXTRA)
            CC_LIBS=-lm -framework OpenGL -framework GLUT -L$(VULKAN_SDK_PATH)/macOS/lib -lvulkan -Wl,-rpath,$(VULKAN_SDK_PATH)/macOS/lib
            $(warning "OpenMP support not found. Installing libomp via Homebrew or MacPorts is recommended.")
            $(warning "Falling back to non-parallelized version.")
        endif
    endif
else
    # Linux
    CC_FLAGS=-g -Wall -fopenmp -I. $(CC_FLAGS_EXTRA)
    CC_LIBS=-lm -fopenmp -lGL -lGLU -lglut -lvulkan
endif

SRC_DIR=src
HDR_DIR=include/
OBJ_DIR=obj
GL_SRC_DIR=src/gl
VULKAN_SRC_DIR=src/vulkan

# Check if we should use Vulkan or OpenGL
UNAME_P := $(shell uname -p)
IS_ARM := $(findstring arm,$(UNAME_P))
IS_APPLE := $(findstring Darwin,$(UNAME_S))

ifeq ($(IS_APPLE),Darwin)
    ifeq ($(IS_ARM),arm)
        # For Apple Silicon, we want to use Vulkan/Metal
    endif
endif

# source and object files
SRC_FILES=$(wildcard $(SRC_DIR)/*.c) $(wildcard $(GL_SRC_DIR)/*.c) $(wildcard $(VULKAN_SRC_DIR)/*.c)
OBJ_FILES=$(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%.o, $(filter-out $(GL_SRC_DIR)/% $(VULKAN_SRC_DIR)/%, $(SRC_FILES))) \
          $(patsubst $(GL_SRC_DIR)/%.c, $(OBJ_DIR)/gl/%.o, $(filter $(GL_SRC_DIR)/%, $(SRC_FILES))) \
          $(patsubst $(VULKAN_SRC_DIR)/%.c, $(OBJ_DIR)/vulkan/%.o, $(filter $(VULKAN_SRC_DIR)/%, $(SRC_FILES)))

# Filter out gl_dummy.c from the source files if using real GL implementation
DUMMY_GL_SRC=$(SRC_DIR)/gl_dummy.c
DUMMY_GL_OBJ=$(OBJ_DIR)/gl_dummy.o

# Only use gl_dummy if DISABLE_GPU is defined
ifdef DISABLE_GPU
ADDITIONAL_OBJS=$(DUMMY_GL_OBJ)
else
ADDITIONAL_OBJS=
OBJ_FILES:=$(filter-out $(DUMMY_GL_OBJ), $(OBJ_FILES))
endif

BIN_FILE=cnavier

all: directories dummy_gl $(BIN_FILE)

$(BIN_FILE): $(OBJ_FILES) $(ADDITIONAL_OBJS)
	$(CC) $(CC_FLAGS) $^ -I$(HDR_DIR) -o $@ $(CC_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -c $^ -I$(HDR_DIR) -o $@ $(LFLAGS)

$(OBJ_DIR)/gl/%.o: $(GL_SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR)/gl
	$(CC) $(CC_FLAGS) -c $^ -I$(HDR_DIR) -o $@

$(OBJ_DIR)/vulkan/%.o: $(VULKAN_SRC_DIR)/%.c
	mkdir -p $(OBJ_DIR)/vulkan
	$(CC) $(CC_FLAGS) -c $^ -I$(HDR_DIR) -o $@

directories:
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/gl
	mkdir -p $(OBJ_DIR)/vulkan
	mkdir -p output
	mkdir -p shaders

# Only create dummy GL if DISABLE_GPU is defined
ifdef DISABLE_GPU
dummy_gl:
	@echo "#include \"gl_solver.h\"" > $(DUMMY_GL_SRC)
	@echo "#include \"poisson.h\"" >> $(DUMMY_GL_SRC)
	@echo "int init_gl_solver(int nx, int ny) { return 0; }" >> $(DUMMY_GL_SRC)
	@echo "void cleanup_gl_solver() {}" >> $(DUMMY_GL_SRC)
	@echo "mtrx poisson_gpu(mtrx f, double dx, double dy, int itmax, double tol) { return poisson(f, dx, dy, itmax, tol); }" >> $(DUMMY_GL_SRC)
	@echo "mtrx poisson_SOR_gpu(mtrx f, double dx, double dy, int itmax, double tol, double beta) { return poisson_SOR(f, dx, dy, itmax, tol, beta); }" >> $(DUMMY_GL_SRC)
	@echo "mtrx poisson_gpu_with_object(mtrx f, double dx, double dy, int itmax, double tol, cell_properties **grid) { return poisson_with_object(f, dx, dy, itmax, tol, grid); }" >> $(DUMMY_GL_SRC)
	@echo "mtrx poisson_SOR_gpu_with_object(mtrx f, double dx, double dy, int itmax, double tol, double beta, cell_properties **grid) { return poisson_SOR_with_object(f, dx, dy, itmax, tol, beta, grid); }" >> $(DUMMY_GL_SRC)

$(DUMMY_GL_OBJ): $(DUMMY_GL_SRC)
	$(CC) $(CC_FLAGS) -c $^ -I$(HDR_DIR) -o $@
else
dummy_gl:
	@echo "Using real OpenGL implementation"
endif

clean:
	rm -rf $(BIN_FILE) $(OBJ_DIR) $(DUMMY_GL_SRC) output/*.vtk