CC=g++
LDFLAGS=-g -std=c++14
CPPFLAGS=-g -std=c++14
CFLAGS=-c -std=c++14 -Wall


SRCS=utils.cpp Train.cpp main.cpp
OBJS=$(SRCS:.cpp=.o)
EXES=Train

all: $(SRCS) $(EXES)

$(EXES): $(OBJS)
	$(CC) $(LDFLAGS) $(OBJS) -o $@

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm -rf $(OBJS) $(EXES)
