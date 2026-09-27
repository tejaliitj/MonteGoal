NVCC = nvcc
NVCC_FLAGS = -rdc=true -O2

SRCS = main.cu \
       rng.cu \
       loader/csv_loader.cpp \
       simulation/bracket.cu \
       simulation/match_engine.cu \
       simulation/group_stage.cu \
       simulation/tournament.cu \
       distributions/uniform.cu \
       distributions/normal.cu \
       distributions/poisson.cu

TARGET = montegoal_wc

all: $(TARGET)

$(TARGET): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) champion_probabilities_cuda.csv

.PHONY: all run clean
