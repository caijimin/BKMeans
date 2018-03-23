#g++ BKMeans.cpp test_BKMeans.cpp -o test_BKMeans -I../include -msse4.2
#CPPFLAGS = -Wall -pedantic
CPPFLAGS += -g
CPPFLAGS += -O3
CPPFLAGS += -Wno-long-long
#CPPFLAGS += -fPIC
CPPFLAGS += -fopenmp
CPPFLAGS += -msse4.2
CPPFLAGS += -lpthread

bkmeans_obj=BKMeans.o test_BKMeans.o

TARGET=bkmeans.bin

all: ${TARGET}

INCLUDES=-I../include

%.o: %.cpp
	${CXX} -c ${CPPFLAGS} ${INCLUDES} $< -o $@

bkmeans.bin : $(bkmeans_obj)
	$(CXX) $(bkmeans_obj)  $(CPPFLAGS)  -o $@ 


clean:
	rm -f ${TARGET}  $(bkmeans_obj)

