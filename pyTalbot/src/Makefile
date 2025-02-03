CC = gcc
CFLAGS = -shared -fPIC -fopenmp
LIBS = -lm -lgsl -lgslcblas
TARGET = integrals.so
SRC = integrals.c

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC) $(LIBS)

clean:
	rm -f $(TARGET)