
all: v1 v2 v3 v4

v1: v1.c
	$(CC) $(CFLAGS) -Wall -o v1 v1.c -lm
	
v2: v2.c
	$(CC) $(CFLAGS) -Wall -o v2 v2.c -lm
	
v3: v3.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -fopenmp -Wall -o v3 v3.c -lm
	
v4: v4.c
	$(CC) $(CFLAGS) -mavx2 -mfma -Wall -o v4 v4.c -lm

run: v1 v2 v3 v4
	./v1 $(ARGS)
	./v2 $(ARGS)
	./v3 $(ARGS)
	./v4 $(ARGS)

clean:
	rm -f v1 v2 v3 v4