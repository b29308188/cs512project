CC=g++
LDFLAGS=-o3 -std=c++14 
CPPFLAGS=-o3 -std=c++14
CFLAGS=-o3 -c -std=c++14 -Wall


SRCS=Train.cpp main.cpp
OBJS=$(SRCS:.cpp=.o)
EXES=Train

all: $(SRCS) $(EXES)

$(EXES): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJS) $(EXES) Train
