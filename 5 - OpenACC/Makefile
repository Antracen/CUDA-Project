# From https://x.momo86.net/?p=29

CXX=/opt/pgi/linux86-64/19.10/bin/pgc++
CXXFLAGS=-I./include -O3 -acc -Minfo

SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR=src
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))

BIN := ./bin
TARGET=sputniPIC.out

all: dir $(BIN)/$(TARGET)

dir: ${BIN}

${BIN}:
	mkdir -p $(BIN)

$(BIN)/$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $+ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir $(OBJDIR)
	$(CXX) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(TARGET)
