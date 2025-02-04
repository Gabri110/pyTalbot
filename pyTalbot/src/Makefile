CC = gcc

# For Mac
# CFLAGS = -shared -fPIC -Xpreprocessor -fopenmp
# LIBS = -lm -lgsl -lgslcblas -L/usr/local/opt/libomp/lib -lomp4

# For Linux
CFLAGS = -shared -fPIC -Xpreprocessor -fopenmp
LIBS = -lm -lgsl -lgslcblas -L/usr/local/opt/libomp/lib -lomp

TARGET = integrals.so
SRC = integrals.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)